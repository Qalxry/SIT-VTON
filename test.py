from vton_diffusion.src.modules.ImagePreprocessor import vitonhd_preprocesser
from vton_diffusion.pipeline import VTONDiffusionPipeline
import os

image_path = './tmp/person/'
cloth_path = './tmp/cloth/'
output_path = './tmp/'

sample = vitonhd_preprocesser(
    image_path=image_path,
    cloth_path=cloth_path,
    output_path=output_path,
)

for key in sample.keys():
    if key != 'im_name' and key != 'category':
        sample[key].unsqueeze_(0)
        print(sample[key].shape)
    else:
        sample[key] = [sample[key]]
        print(sample[key])


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FILE_DIR, "vton_diffusion", "models")

pipe = VTONDiffusionPipeline(
    model_path=MODEL_PATH,
    scheduler_type="unipc",
    VRAM_mode="lowvram",
    enable_xformers=True,
)

pipe(
    image_path=image_path,
    cloth_path=cloth_path,
    output_path=output_path,
)
