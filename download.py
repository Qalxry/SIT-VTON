import wget, zipfile
import os
import tqdm


FILE_DIR = os.path.dirname(os.path.abspath(__file__))


save_paths = [
    os.path.join(FILE_DIR, 'annotator/openpose/model/'),
    os.path.join(FILE_DIR, 'vton_diffusion/models/miccunifi/ladi-vton'),
    os.path.join(FILE_DIR, 'vton_diffusion/models/miccunifi/ladi-vton'),
    os.path.join(FILE_DIR, 'vton_diffusion/models/miccunifi/ladi-vton'),
    os.path.join(FILE_DIR, 'vton_diffusion/models/miccunifi/ladi-vton'),
    os.path.join(FILE_DIR, 'vton_diffusion/models/stabilityai'),
    os.path.join(FILE_DIR, 'vton_diffusion/models/stabilityai'),
    os.path.join(FILE_DIR, 'vton_diffusion/models/stabilityai'),
    os.path.join(FILE_DIR, 'annotator/CIHP_PGN/checkpoint'),
]

file_names = [
    'pose_iter_584000.caffemodel.pt',       # Openpose body 25 model
    'emasc_vitonhd.pth',                    # EMASC model
    'inversion_adapter_vitonhd.pth',        # Inversion adapter model
    'unet_vitonhd.pth',                     # U-Net model
    'warping_vitonhd.pth',                  # Warping model
    'pytorch_model.bin',                    # Text encoder model
    'diffusion_pytorch_model.bin',          # VAE model
    'pytorch_model.bin',                    # CLIP model
    'CIHP_pgn.zip',                         # CIHP_PGN model
]

file_urls = [
    # 'https://drive.google.com/uc?id=1ij6KxxeBkOjox-kc6RFCQMU_3QY2krR3&export=download&confirm=t&uuid=96966ebb-4c84-4fa0-baa0-38ad425bd773&at=AB6BwCDjSoOUHd2Xo6XIcaPZpxPA:1692348756164',
    'https://github.com/Qalxry/SIT-VTON/releases/download/models/pose_iter_584000.caffemodel.pt',
    'https://github.com/miccunifi/ladi-vton/releases/download/weights/emasc_vitonhd.pth',
    'https://github.com/miccunifi/ladi-vton/releases/download/weights/inversion_adapter_vitonhd.pth',
    'https://github.com/miccunifi/ladi-vton/releases/download/weights/unet_vitonhd.pth',
    'https://github.com/miccunifi/ladi-vton/releases/download/weights/warping_vitonhd.pth',
    'https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/text_encoder/pytorch_model.bin',
    'https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/vae/diffusion_pytorch_model.bin',
    'https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/pytorch_model.bin',
    'https://github.com/Qalxry/SIT-VTON/releases/download/models/CIHP_pgn.zip'
]

file_urls_backup = []

for save_path, file_name, file_url in tqdm(zip(save_paths, file_names, file_urls)):
    if not os.path.exists(os.path.join(save_path, file_name)):
        print(f'Downloading {file_name} to {save_path}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        wget.download(file_url, out=os.path.join(save_path, file_name))
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(save_path, file_name), 'r') as zip_ref:
                zip_ref.extractall(save_path)
            os.remove(os.path.join(save_path, file_name))


