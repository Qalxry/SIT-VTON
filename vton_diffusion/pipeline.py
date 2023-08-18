import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from diffusers import __version__ as diffusers_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

from PIL.Image import Image
from typing import List, Optional, Union, Literal

from .src.utils.encode_text_word_embedding import encode_text_word_embedding
from .src.vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline
from .src.modules.ModelLoader import load_all_models
from .src.modules.ImagePreprocessor import vitonhd_preprocesser



class VTONDiffusionPipeline:
    def __init__(
        self,
        model_path: str,
        scheduler_type: Literal["ddim", "dpm", "euler", "euler_a", "unipc"]="ddim",
        VRAM_mode: Literal["fullvram", "medvram", "lowvram"]="lowvram",
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_xformers: bool = False,
    ):
        self.model_loaded       =   False
        self.pipeline           =   None
        self.text_encoder       =   None
        self.tokenizer          =   None
        self.vae                =   None
        self.scheduler          =   None
        self.vision_encoder     =   None
        self.processor          =   None
        self.unet               =   None
        self.emasc              =   None
        self.inversion_adapter  =   None
        self.enable_xformers    =   enable_xformers
        self.VRAM_mode          =   VRAM_mode
        self.device             =   torch.device(device)
        self.dtype              =   dtype
        self.model_path         =   model_path
        self.scheduler_type     =   scheduler_type
        self.num_vstar          =   16
        
        
    def load_model(self):
        if self.VRAM_mode == "fullvram":
            load_mode = 'cuda'
        elif self.VRAM_mode == "medvram":
            load_mode = 'auto'
        else:
            load_mode = 'cpu'

        models = load_all_models(
            dataset='vitonhd', 
            model_path=self.model_path, 
            scheduler_type=self.scheduler_type, 
            device=load_mode,
        )
        self.text_encoder               =   models['text_encoder']          # stabilityai/stable-diffusion-2-inpainting
        self.tokenizer                  =   models['tokenizer']             # stabilityai/stable-diffusion-2-inpainting
        self.vae                        =   models['vae']                   # stabilityai/stable-diffusion-2-inpainting
        self.scheduler                  =   models['scheduler']             # stabilityai/stable-diffusion-2-inpainting
        self.vision_encoder             =   models['vision_encoder']        # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
        self.processor                  =   models['processor']             # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
        self.unet                       =   models['unet']                  # miccunifi/ladi-vton
        self.emasc                      =   models['emasc']                 # miccunifi/ladi-vton
        self.inversion_adapter          =   models['inversion_adapter']     # miccunifi/ladi-vton
        (self.tps, self.refinement)     =   models['warping_module']        # miccunifi/ladi-vton
                
        if self.enable_xformers:
            self.enable_xformers_memory_efficient_attention()
        
        # Set to eval mode
        self.text_encoder.eval()
        self.vae.eval()
        self.emasc.eval()
        self.inversion_adapter.eval()
        self.unet.eval()
        self.tps.eval()
        self.refinement.eval()
        self.vision_encoder.eval()

        # Create the pipeline
        self.pipeline = StableDiffusionTryOnePipeline(
            text_encoder=self.text_encoder,
            vae=self.vae,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            emasc=self.emasc,
            emasc_int_layers=[1, 2, 3, 4, 5],
        )
        
        self.model_loaded = True
        
        print("Model loaded successfully!")
    
    
    def enable_xformers_memory_efficient_attention(self):
        if is_xformers_available():
            if self.unet is not None:
                self.unet.enable_xformers_memory_efficient_attention()
            self.enable_xformers = True
            print("xformers is available. Memory efficient attention enabled.")
        else:
            print("xformers is not available. Make sure it is installed correctly!")
            print("Running on default attention.")

    
    ### WARNING: 这里必须加上@torch.inference_mode()，
    ###          否则无法回收vision_encoder、text_encoder、inversion_adapter、vae、unet的显存！！！
    ###          但似乎tps和refinement不需要加上这个，也能回收显存，不知道为什么。可能是因为这两个模型的显存占用量比较小，或者是BUG。
    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image_path: str = None,
        cloth_path: str = None,
        output_path: str = None,
        height: Optional[int] = 512,
        width: Optional[int] = 384,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = 0,
    ) -> Image:
        torch.cuda.empty_cache()
        
        sample = vitonhd_preprocesser(image_path, cloth_path, output_path)
        for key in sample.keys():
            if key != 'im_name' and key != 'category':
                sample[key].unsqueeze_(0)
            else:
                sample[key] = [sample[key]]
                
        if self.model_loaded == False:
            self.load_model()
        
        if self.VRAM_mode == "fullvram":
            self.to(device=self.device, dtype=self.dtype)

        # Generate the images
        generator = torch.Generator("cuda").manual_seed(seed)
        
        model_img = sample.get("image").to(device=self.device, dtype=self.dtype)
        mask_img = sample.get("inpaint_mask").to(device=self.device, dtype=self.dtype)
        pose_map = sample.get("pose_map").to(device=self.device, dtype=self.dtype)
        category = sample.get("category")
        cloth = sample.get("cloth").to(device=self.device, dtype=self.dtype)
        im_mask = sample.get('im_mask').to(device=self.device, dtype=self.dtype)  

   
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
        
        if self.VRAM_mode == "lowvram":
            self.tps.to(self.device, dtype=self.dtype)
            
        low_grid, _, _, _, _, _, _, _ = self.tps(low_cloth, agnostic)  # 30 MB     
        
        if self.VRAM_mode == "lowvram":
            self.tps.to('cpu')
            torch.cuda.empty_cache()    # free up 30 MB

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
        
        if self.VRAM_mode == "lowvram":
            self.refinement.to(self.device, dtype=self.dtype)
            del low_grid, low_cloth, agnostic, low_im_mask, low_pose_map, highres_grid
            torch.cuda.empty_cache()
        
        warped_cloth = self.refinement(warped_cloth)
        warped_cloth = warped_cloth.clamp(-1, 1)

        # Get the visual features of the in-shop cloths
        input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                                antialias=True).clamp(0, 1)
        processed_images = self.processor(images=input_image, return_tensors="pt")

        # Compute the predicted PTEs
        if self.VRAM_mode == "lowvram":
            self.refinement.to('cpu')
            torch.cuda.empty_cache()    # free up 200 MB
            self.vision_encoder.to(self.device, dtype=self.dtype)
            
        clip_cloth_features = self.vision_encoder(processed_images.pixel_values.to(device=self.device,dtype=self.dtype)).last_hidden_state
        
        if self.VRAM_mode == "lowvram":
            self.vision_encoder.to('cpu')
            del input_image, processed_images
            torch.cuda.empty_cache()
            self.inversion_adapter.to(self.device, dtype=self.dtype)
        
        word_embeddings = self.inversion_adapter(clip_cloth_features.to(model_img.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], self.num_vstar, -1))
        
        if self.VRAM_mode == "lowvram":
            self.inversion_adapter = self.inversion_adapter.to('cpu')
            torch.cuda.empty_cache()
            self.text_encoder.to(self.device, dtype=self.dtype)

        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',

        }
        text = [f'a photo of a model wearing {category_text[category]} {" $ " * self.num_vstar}' for
                category in sample['category']]

        # Tokenize text
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                    truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(self.device)

        # Encode the text using the PTEs extracted from the in-shop cloths
        encoder_hidden_states = encode_text_word_embedding(self.text_encoder, tokenized_text,
                                                            word_embeddings, self.num_vstar).last_hidden_state
       
        if self.VRAM_mode == "lowvram":
            self.text_encoder.to('cpu') 
            del clip_cloth_features, word_embeddings, tokenized_text
            torch.cuda.empty_cache()
            # self.unet.to(self.device, dtype=self.dtype)
            self.vae.to(self.device, dtype=self.dtype)
            self.unet.to(self.device, dtype=self.dtype)
            self.pipeline.to(self.device)
            self.emasc.to(self.device, dtype=self.dtype)
            
        # Generate images
        generated_images = self.pipeline(
            image=model_img,
            mask_image=mask_img,
            pose_map=pose_map,
            warped_cloth=warped_cloth,
            prompt_embeds=encoder_hidden_states,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
            cloth_input_type='warped',
            num_inference_steps=num_inference_steps,
        ).images                                                        # 400 MB
        
        
        if self.VRAM_mode == "lowvram":
            self.unet.to('cpu')
            self.vae.to('cpu')
            self.pipeline.to('cpu')
            torch.cuda.empty_cache()

        # Save images
        for gen_image in generated_images:
            if not os.path.exists(os.path.join(output_path, "result")):
                os.makedirs(os.path.join(output_path, "result"))
            gen_image.save(os.path.join(output_path, "result", 'result.png'))

        del sample, model_img, mask_img, pose_map, warped_cloth, encoder_hidden_states, generated_images
        torch.cuda.empty_cache()
           
            
    def unload_model(self):
        # Free up memory
        self.model_loaded = False
        del self.pipeline
        del self.text_encoder
        del self.vae
        del self.emasc
        del self.unet
        del self.tps
        del self.refinement
        del self.vision_encoder
        self.pipeline = None
        self.text_encoder = None
        self.vae = None
        self.emasc = None
        self.unet = None
        self.tps = None
        self.refinement = None
        self.vision_encoder = None
        torch.cuda.empty_cache()
    
    
    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        if self.model_loaded:
            self.pipeline.to(device, dtype)
            self.text_encoder.to(device, dtype)
            self.vae.to(device, dtype)
            self.emasc.to(device, dtype)
            self.unet.to(device, dtype)
            self.tps.to(device, dtype)
            self.refinement.to(device, dtype)
            self.vision_encoder.to(device, dtype)
            self.inversion_adapter.to(device, dtype)
            torch.cuda.empty_cache()