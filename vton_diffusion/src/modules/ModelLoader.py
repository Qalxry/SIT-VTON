import torch
import os.path
from typing import Literal
from transformers import (
    CLIPTextModel, 
    AutoConfig, 
    CLIPTokenizer, 
    CLIPVisionModelWithProjection, 
    AutoProcessor,
    CLIPVisionModel,
)
from diffusers import (
    UNet2DConditionModel,
    DDIMScheduler, 
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
from ..models.AutoencoderKL import AutoencoderKL
from ..models.ConvNet_TPS import ConvNet_TPS
from ..models.UNet import UNetVanilla
from ..models.emasc import EMASC
from ..models.inversion_adapter import InversionAdapter


all_model_path = "./models/"


def set_model_path(path):
    global all_model_path
    all_model_path = path


def inversion_adapter(dataset: Literal['dresscode', 'vitonhd']):
    # config = AutoConfig.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    # text_encoder_config =  UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
    # inversion_adapter = InversionAdapter(input_dim=config.vision_config.hidden_size,
    #                                      hidden_dim=config.vision_config.hidden_size * 4,
    #                                      output_dim=text_encoder_config['hidden_size'] * 16,
    #                                      num_encoder_layers=1,
    #                                      config=config.vision_config)
    # checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/inversion_adapter_{dataset}.pth"
    # inversion_adapter.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    # return inversion_adapter
    global all_model_path
    config = AutoConfig.from_pretrained(os.path.join(all_model_path, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"))
    text_encoder_config =  UNet2DConditionModel.load_config(os.path.join(all_model_path, "stabilityai/stable-diffusion-2-inpainting"), subfolder="text_encoder")
    inversion_adapter = InversionAdapter(input_dim=config.vision_config.hidden_size,
                                         hidden_dim=config.vision_config.hidden_size * 4,
                                         output_dim=text_encoder_config['hidden_size'] * 16,
                                         num_encoder_layers=1,
                                         config=config.vision_config)
    checkpoint_path = os.path.join(all_model_path, f"miccunifi/ladi-vton/inversion_adapter_{dataset}.pth")
    inversion_adapter.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return inversion_adapter


def extended_unet(dataset: Literal['dresscode', 'vitonhd']):
    # config = UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
    # config['in_channels'] = 31
    # unet = UNet2DConditionModel.from_config(config)
    # checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/unet_{dataset}.pth"
    # unet.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    # return unet
    global all_model_path
    config = UNet2DConditionModel.load_config(os.path.join(all_model_path, "stabilityai/stable-diffusion-2-inpainting"), subfolder="unet")
    config['in_channels'] = 31
    unet = UNet2DConditionModel.from_config(config)
    checkpoint_path = os.path.join(all_model_path, f"miccunifi/ladi-vton/unet_{dataset}.pth")
    unet.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return unet


def emasc(dataset: Literal['dresscode', 'vitonhd']):
    # in_feature_channels = [128, 128, 256, 512]
    # out_feature_channels = [256, 512, 512, 512]
    # in_feature_channels.insert(0, 128)
    # out_feature_channels.insert(0, 128)
    # emasc = EMASC(in_feature_channels,
    #               out_feature_channels,
    #               kernel_size=3,
    #               padding=1,
    #               stride=1,
    #               type='nonlinear')
    # checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/emasc_{dataset}.pth"
    # emasc.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    # return emasc
    global all_model_path
    in_feature_channels = [128, 128, 256, 512]
    out_feature_channels = [256, 512, 512, 512]
    in_feature_channels.insert(0, 128)
    out_feature_channels.insert(0, 128)
    emasc = EMASC(in_feature_channels,
                  out_feature_channels,
                  kernel_size=3,
                  padding=1,
                  stride=1,
                  type='nonlinear')
    checkpoint_path = os.path.join(all_model_path, f"miccunifi/ladi-vton/emasc_{dataset}.pth")
    emasc.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return emasc


def warping_module(dataset: Literal['dresscode', 'vitonhd']):
    # tps = ConvNet_TPS(256, 192, 21, 3)
    # refinement = UNetVanilla(n_channels=24, n_classes=3, bilinear=True)
    # checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/warping_{dataset}.pth"
    # tps.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['tps'])
    # refinement.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['refinement'])
    # return tps, refinement
    global all_model_path
    tps = ConvNet_TPS(256, 192, 21, 3)
    refinement = UNetVanilla(n_channels=24, n_classes=3, bilinear=True)
    checkpoint_path = os.path.join(all_model_path, f"miccunifi/ladi-vton/warping_{dataset}.pth")
    tps.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['tps'])
    refinement.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['refinement'])
    return tps, refinement


def text_encoder():
    return CLIPTextModel.from_pretrained(os.path.join(all_model_path, "stabilityai/stable-diffusion-2-inpainting"), subfolder="text_encoder")


def tokenizer():
    return CLIPTokenizer.from_pretrained(os.path.join(all_model_path, "stabilityai/stable-diffusion-2-inpainting"), subfolder="tokenizer")


# def scheduler(scheduler:Literal[DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler]=DDIMScheduler):
def scheduler(scheduler_type: Literal['ddim', 'dpm', 'unipc'] = 'ddim'):
    if scheduler_type == 'ddim':
        return DDIMScheduler.from_pretrained(os.path.join(all_model_path, "stabilityai/stable-diffusion-2-inpainting"), subfolder="scheduler")
    elif scheduler_type == 'dpm':
        return DPMSolverMultistepScheduler.from_pretrained(os.path.join(all_model_path, "stabilityai/stable-diffusion-2-inpainting"), subfolder="scheduler")
    elif scheduler_type == 'unipc':
        return UniPCMultistepScheduler.from_pretrained(os.path.join(all_model_path, "stabilityai/stable-diffusion-2-inpainting"), subfolder="scheduler")
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not supported")
    

def vae():
    return AutoencoderKL.from_pretrained(os.path.join(all_model_path, "stabilityai/stable-diffusion-2-inpainting"), subfolder="vae")


def vision_encoder():
    CLIPVisionModelWithProjection.to
    
    return CLIPVisionModelWithProjection.from_pretrained(os.path.join(all_model_path, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"))
    # return CLIPVisionModel.from_pretrained(os.path.join(all_model_path, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"))

def processor():
    return AutoProcessor.from_pretrained(os.path.join(all_model_path, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"))


def load_all_models(
    dataset: Literal['dresscode', 'vitonhd'], 
    model_path: str=None, 
    scheduler_type: Literal['ddim', 'dpm', 'unipc'] = 'ddim',
    device: Literal['cpu', 'cuda', 'auto'] = 'cpu',
):
    if model_path is not None:
        set_model_path(model_path)
    if device == 'cpu':
        return {
            'text_encoder': text_encoder(),                     # stabilityai/stable-diffusion-2-inpainting
            'tokenizer': tokenizer(),                           # stabilityai/stable-diffusion-2-inpainting
            'scheduler': scheduler(scheduler_type),             # stabilityai/stable-diffusion-2-inpainting
            'vae': vae(),                                       # stabilityai/stable-diffusion-2-inpainting
            'vision_encoder': vision_encoder(),                 # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
            'processor': processor(),                           # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
            'unet': extended_unet(dataset),                     # miccunifi/ladi-vton 
            'emasc': emasc(dataset),                            # miccunifi/ladi-vton
            'inversion_adapter': inversion_adapter(dataset),    # miccunifi/ladi-vton
            'warping_module': warping_module(dataset)           # miccunifi/ladi-vton
        }
    elif device == 'cuda':
        return {
            'text_encoder': text_encoder().to('cuda'),                     # stabilityai/stable-diffusion-2-inpainting
            'tokenizer': tokenizer(),                                      # stabilityai/stable-diffusion-2-inpainting
            'scheduler': scheduler(scheduler_type).to('cuda'),             # stabilityai/stable-diffusion-2-inpainting
            'vae': vae().to('cuda'),                                       # stabilityai/stable-diffusion-2-inpainting
            'vision_encoder': vision_encoder().to('cuda'),                 # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
            'processor': processor(),                                      # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
            'unet': extended_unet(dataset).to('cuda'),                     # miccunifi/ladi-vton 
            'emasc': emasc(dataset).to('cuda'),                            # miccunifi/ladi-vton
            'inversion_adapter': inversion_adapter(dataset).to('cuda'),    # miccunifi/ladi-vton
            'warping_module': warping_module(dataset)[0].to('cuda'),       # miccunifi/ladi-vton
            'refinement': warping_module(dataset)[1].to('cuda')            # miccunifi/ladi-vton
        }
    elif device == 'auto':
        return {
            'text_encoder': text_encoder(),                             # stabilityai/stable-diffusion-2-inpainting
            'tokenizer': tokenizer(),                                   # stabilityai/stable-diffusion-2-inpainting
            'scheduler': scheduler(scheduler_type),                     # stabilityai/stable-diffusion-2-inpainting
            'vae': vae(),                                               # stabilityai/stable-diffusion-2-inpainting
            'vision_encoder': vision_encoder().to('cuda'),              # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
            'processor': processor(),                                   # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
            'unet': extended_unet(dataset),                             # miccunifi/ladi-vton 
            'emasc': emasc(dataset),                                    # miccunifi/ladi-vton
            'inversion_adapter': inversion_adapter(dataset).to('cuda'), # miccunifi/ladi-vton
            'warping_module': warping_module(dataset)[0].to('cuda'),    # miccunifi/ladi-vton
            'refinement': warping_module(dataset)[1].to('cuda')         # miccunifi/ladi-vton
        }

'''
# def load_ladi_vton_models(dataset: Literal['dresscode', 'vitonhd'], model_path: str=None):
#     if model_path is not None:
#         global all_model_path
#         all_model_path = model_path
#     return {
#         'unet': extended_unet(dataset),                     # ladi-vton 
#         'emasc': emasc(dataset),                            # ladi-vton
#         'inversion_adapter': inversion_adapter(dataset),    # ladi-vton
#         'warping_module': warping_module(dataset)           # ladi-vton
#     }
# def load_stable_diffusion_models(model_path: str=None):
#     if model_path is not None:
#         global all_model_path
#         all_model_path = model_path
#     return {
#         'text_encoder': text_encoder(),                     # stabilityai/stable-diffusion-2-inpainting
#         'tokenizer': tokenizer(),                           # stabilityai/stable-diffusion-2-inpainting
#         'scheduler': scheduler(),                           # stabilityai/stable-diffusion-2-inpainting
#         'vae': vae(),                                       # stabilityai/stable-diffusion-2-inpainting
#     }
'''
