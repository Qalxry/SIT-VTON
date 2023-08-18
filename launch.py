from vton_diffusion.pipeline import VTONDiffusionPipeline
from pathlib import Path
from diffusers.utils import check_min_version
from flask import Flask, request, jsonify
import torch
import os
import io
from PIL import Image

app = Flask(__name__)


PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# 取出Path(__file__)的路径
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FILE_DIR, "vton_diffusion", "models")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

pipeline = VTONDiffusionPipeline(
    model_path=MODEL_PATH,
    scheduler_type="unipc",
    VRAM_mode="lowvram",
    device="cuda",
    dtype=torch.float16,
)

pipeline.enable_xformers_memory_efficient_attention()


def check_models():
    pass
    

@app.route('/run', methods=['POST'])
def run():
    global pipeline
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        cloth_path = data.get('cloth_path')
        output_path = data.get('output_path')
        return_image = data.get('return_image')
        
        if image_path is None:
            return jsonify({'error': 'image_path is missing in JSON'}), 400
        
        if output_path is None:
            return jsonify({'error': 'output_path is missing in JSON'}), 400
        
        if cloth_path is None:
            return jsonify({'error': 'cloth_path is missing in JSON'}), 400
        
        if return_image != True and return_image != False:
            return_image = False
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    try:
        pipeline(
            image_path=image_path,
            cloth_path=cloth_path,
            output_path=output_path,
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    if return_image == False:
        return jsonify({'status': 'success'}), 200
    
    # 保存生成的图片到内存中
    image_io = io.BytesIO()
    image = Image.open(os.path.join(output_path, "result","result.png"))
    image.save(image_io, format='PNG')
    image_io.seek(0)
    
    # 构建响应数据，包含图片和编号
    response_data = {
        'status': 'success',
        'image': image_io.getvalue()
    }
    
    return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
    
    