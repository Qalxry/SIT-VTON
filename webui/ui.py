import gradio as gr


def fittingRoom(inputs: list):
    image_user, image_cloth1, image_cloth2, settings = inputs
    image_user = image_user
    return image_user

def switchVisibility(x):
    x.visible = not x.visible
    x.update()
    return x

# if __name__ == "__main__":
    
with gr.Blocks() as webui:
    gr.Markdown("VTON - 虚拟试衣 WebUI 面板")

    # 试衣间 Fitting Room
    with gr.Tab("试衣间"):
        with gr.Row():
            image_user = gr.Image(label="你的全身照片")
            button_openWardrobe = gr.Button("我的衣柜", scale=0)
            image_cloth1 = gr.Image(label="选择的上衣")
            image_cloth2 = gr.Image(label="选择的下衣")
        
        with gr.Tabs(visible=False) as wardrobe:
            with gr.Tab("上衣") as tab_cloth1:
                pass
            with gr.Tab("下衣") as tab_cloth2:
                pass

        with gr.Row():
            with gr.Column():
                with gr.Blocks() as settings: 
                    with gr.Row():
                        scheduler = gr.Dropdown(["DDIM", "PNDM", "Euler a" , "DPM-Slover++", "UniPC"], label="采样方法", value="DDIM", interactive=True)
                        step = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="迭代步数", interactive=True)
                    mode = gr.Radio(["Full VRAM", "Medium VRAM", "Low VRAM"], label="模式", value="Medium VRAM", interactive=True)
                    with gr.Row():
                        batch_count = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="生成批次 Batch count", interactive=True)
                        batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="每批数量 Batch size", interactive=True)
                button_generation = gr.Button("生成试穿图像")
            with gr.Blocks():
                image_output = gr.Image(label="试衣效果图")  
    
    button_openWardrobe.click(switchVisibility, inputs=wardrobe, outputs=wardrobe)
    button_generation.click(fittingRoom, inputs=[image_user, image_cloth1, image_cloth2, settings], outputs=image_output)

    # 衣柜 Wardrobe
    with gr.Tab("衣柜"):
        pass

    # 设置 Settings
    with gr.Tab("设置"):
        pass
    
webui.launch(share=False)