import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from accelerate import Accelerator
from diffusers import __version__ as diffusers_version
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

from dataset.dresscode import DressCodeDataset
from dataset.preprecesser import VitonHDDataset
# from models.AutoencoderKL import AutoencoderKL
from src.utils.encode_text_word_embedding import encode_text_word_embedding
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics
from vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline

import modules.ModelLoader as ModelLoader

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

# 获取cuda设备
my_cuda = torch.device('cuda')

# 获取cpu设备
my_cpu = torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Full inference script")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )

    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size to use.")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument("--num_workers", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.")

    parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
    parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"])
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')
    parser.add_argument("--use_png", default=False, action="store_true")
    parser.add_argument("--num_inference_steps", default=25, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--compute_metrics", default=False, action="store_true")


    parser.add_argument(
        # "--lowvram", action="store_true", help="Whether or not to use lowvram mode to save GPU memory."
        "--lowvram", default=False, action="store_true", help="Whether or not to use lowvram mode to save GPU memory."
    )
    parser.add_argument(
        # "--lowvram", action="store_true", help="Whether or not to use lowvram mode to save GPU memory."
        "--medvram", default=False, action="store_true", help="Whether or not to use medvram mode to save GPU memory."
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


@torch.inference_mode()
def main():

    # 释放之前的显存
    torch.cuda.empty_cache()
    
    args = parse_args()

    # Check if the dataset dataroot is provided
    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")
    if args.dataset == "dresscode" and args.dresscode_dataroot is None:
        raise ValueError("DressCode dataroot must be provided")

    # Setup accelerator and device.
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    models = ModelLoader.load_all_models(dataset=args.dataset, model_path="models/", scheduler_type="ddim")
    text_encoder        =   models['text_encoder']          # stabilityai/stable-diffusion-2-inpainting
    tokenizer           =   models['tokenizer']             # stabilityai/stable-diffusion-2-inpainting
    vae                 =   models['vae']                   # stabilityai/stable-diffusion-2-inpainting
    scheduler           =   models['scheduler']             # stabilityai/stable-diffusion-2-inpainting
    vision_encoder      =   models['vision_encoder']        # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    processor           =   models['processor']             # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    unet                =   models['unet']                  # miccunifi/ladi-vton
    emasc               =   models['emasc']                 # miccunifi/ladi-vton
    inversion_adapter   =   models['inversion_adapter']     # miccunifi/ladi-vton
    (tps, refinement)   =   models['warping_module']        # miccunifi/ladi-vton


    int_layers = [1, 2, 3, 4, 5]

    # Enable xformers memory efficient attention if requested
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Load the datasets
    if args.category != 'all':
        category = [args.category]
    else:
        category = ['dresses', 'upper_body', 'lower_body']

    outputlist = ['image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'cloth']
    if args.dataset == "dresscode":
        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            outputlist=outputlist,
            category=category,
            size=(512, 384)
        )
    elif args.dataset == "vitonhd":
        test_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            outputlist=outputlist,
            size=(512, 384),
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Cast to weight_dtype
    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16

    lowvram = args.lowvram
    medvram = args.medvram
    
    if lowvram:
        print("Using low VRAM mode")
        # 将一些小模型存放在cpu上
        emasc.to(device, dtype=weight_dtype)                # 22 MB
        refinement.to(device, dtype=weight_dtype)           # 42 MB
        tps.to(device, dtype=weight_dtype)                  # 44 MB
        vae.to('cpu', dtype=weight_dtype)                   # 168 MB
        inversion_adapter.to('cpu', dtype=weight_dtype)     # 252 MB
        text_encoder.to('cpu', dtype=weight_dtype)          # 658 MB
        vision_encoder.to('cpu', dtype=weight_dtype)        # 1308 MB
        unet.to(device, dtype=weight_dtype)                 # 1722 MB
    elif medvram:
        print("Using medium VRAM mode")
        text_encoder.to(device, dtype=weight_dtype)         # 658 MB
        vae.to(device, dtype=weight_dtype)                  # 168 MB
        emasc.to(device, dtype=weight_dtype)                # 22 MB
        inversion_adapter.to(device, dtype=weight_dtype)    # 252 MB
        tps.to(device, dtype=weight_dtype)                  # 44 MB
        refinement.to(device, dtype=weight_dtype)           # 42 MB
        vision_encoder.to(device, dtype=weight_dtype)       # 1308 MB
        unet.to(device, dtype=weight_dtype)                 # 1722 MB
    else:
        print("Using full VRAM mode")
        text_encoder.to(device, dtype=weight_dtype)         # 658 MB
        vae.to(device, dtype=weight_dtype)                  # 168 MB
        emasc.to(device, dtype=weight_dtype)                # 22 MB
        inversion_adapter.to(device, dtype=weight_dtype)    # 252 MB
        tps.to(device, dtype=weight_dtype)                  # 44 MB
        refinement.to(device, dtype=weight_dtype)           # 42 MB
        vision_encoder.to(device, dtype=weight_dtype)       # 1308 MB
        unet.to(device, dtype=weight_dtype)                 # 1722 MB

    # Set to eval mode
    text_encoder.eval()
    vae.eval()
    emasc.eval()
    inversion_adapter.eval()
    unet.eval()
    tps.eval()
    refinement.eval()
    vision_encoder.eval()

    # Create the pipeline
    val_pipe = StableDiffusionTryOnePipeline(
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        emasc=emasc,
        emasc_int_layers=int_layers,
    ).to(device)                                            # 800 MB

    # Prepare the dataloader and create the output directory
    test_dataloader = accelerator.prepare(test_dataloader)
    save_dir = os.path.join(args.output_dir, args.test_order)
    os.makedirs(save_dir, exist_ok=True)
    generator = torch.Generator("cuda").manual_seed(args.seed)
        

    # Generate the images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch.get("image").to(weight_dtype)
        mask_img = batch.get("inpaint_mask").to(weight_dtype)
        if mask_img is not None:
            mask_img = mask_img.to(weight_dtype)
        pose_map = batch.get("pose_map").to(weight_dtype)
        category = batch.get("category")
        cloth = batch.get("cloth").to(weight_dtype)
        im_mask = batch.get('im_mask').to(weight_dtype)


        # Generate the warped cloth
        # For sake of performance, the TPS parameters are predicted on a low resolution image

        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        
        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)  # 30 MB     

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(512, 384),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)

        if diffusers_version == "0.14.0":
            warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')
        else:
            warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border', align_corners=False)

        # Refine the warped cloth using the refinement network
        warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)      # 400 MB
        warped_cloth = refinement(warped_cloth)
        warped_cloth = warped_cloth.clamp(-1, 1)

        # Get the visual features of the in-shop cloths
        input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                               antialias=True).clamp(0, 1)
        processed_images = processor(images=input_image, return_tensors="pt")

        # Compute the predicted PTEs
        if lowvram:
            vision_encoder.to(device, dtype=weight_dtype)
            clip_cloth_features = vision_encoder(
                processed_images.pixel_values.to(model_img.device, dtype=weight_dtype)).last_hidden_state
            vision_encoder.to('cpu', dtype=weight_dtype)
            torch.cuda.empty_cache()
            
            inversion_adapter.to(device, dtype=weight_dtype)
            word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
            inversion_adapter.to('cpu', dtype=weight_dtype)
            torch.cuda.empty_cache()
        else:
            clip_cloth_features = vision_encoder(
                processed_images.pixel_values.to(model_img.device, dtype=weight_dtype)).last_hidden_state
            word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
            
            
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], args.num_vstar, -1))

        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',

        }
        text = [f'a photo of a model wearing {category_text[category]} {" $ " * args.num_vstar}' for
                category in batch['category']]

        # Tokenize text
        tokenized_text = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length",
                                   truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(word_embeddings.device)

        # Encode the text using the PTEs extracted from the in-shop cloths
        if lowvram:
            text_encoder.to(device, dtype=weight_dtype)
            encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text,
                                                               word_embeddings, args.num_vstar).last_hidden_state
            del clip_cloth_features, word_embeddings, tokenized_text
            torch.cuda.empty_cache()
        else:
            encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text,
                                                           word_embeddings, args.num_vstar).last_hidden_state

        # vae 
        if lowvram:
            # unet.to(device, dtype=weight_dtype)
            vae.to(device, dtype=weight_dtype)
        '''
            # Generate images
            generated_images = val_pipe(
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                warped_cloth=warped_cloth,
                prompt_embeds=encoder_hidden_states,
                height=512,
                width=384,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                cloth_input_type='warped',
                num_inference_steps=args.num_inference_steps
            ).images                                                        # 400 MB
        '''
        # Generate images
        generated_images = val_pipe(
            image=model_img,
            mask_image=mask_img,
            pose_map=pose_map,
            warped_cloth=warped_cloth,
            prompt_embeds=encoder_hidden_states,
            height=512,
            width=384,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
            cloth_input_type='warped',
            num_inference_steps=args.num_inference_steps,
        ).images                                                        # 400 MB
        
        if lowvram:
            # unet.to('cpu', dtype=weight_dtype)
            vae.to('cpu', dtype=weight_dtype)
            text_encoder.to('cpu', dtype=weight_dtype)
            torch.cuda.empty_cache()


        # Save images
        for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
            if not os.path.exists(os.path.join(save_dir, cat)):
                os.makedirs(os.path.join(save_dir, cat))

            if args.use_png:
                name = name.replace(".jpg", ".png")
                gen_image.save(
                    os.path.join(save_dir, cat, name))
            else:
                gen_image.save(
                    os.path.join(save_dir, cat, name), quality=95)

        # Free up memory
        del model_img
        del mask_img
        del pose_map
        del cloth
        del im_mask
        del low_cloth
        del low_im_mask
        del low_pose_map
        del agnostic
        del low_grid
        del theta
        del rx
        del ry
        del cx
        del cy
        del rg
        del cg
        del warped_cloth
        del input_image
        del processed_images
        
        if not lowvram and not medvram:
            del clip_cloth_features
            del word_embeddings
            del tokenized_text
        
        del encoder_hidden_states
        del generated_images

        # Empty cache to free up memory
        torch.cuda.empty_cache()                    # free 800 MB

    # Free up memory
    del val_pipe
    del text_encoder
    del vae
    del emasc
    del unet
    del tps
    del refinement
    del vision_encoder
    torch.cuda.empty_cache()

    if args.compute_metrics:
        metrics = compute_metrics(save_dir, args.test_order, args.dataset, args.category, ['all'],
                                  args.dresscode_dataroot, args.vitonhd_dataroot)

        with open(os.path.join(save_dir, f"metrics_{args.test_order}_{args.category}.json"), "w+") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
