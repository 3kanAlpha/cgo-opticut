import gradio as gr
from opt_solver import process_image

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="切り抜きたい画像をアップロード"),
        gr.Image(type="pil", label="編集済み画像をアップロード (黒で背景をマーク、白で前景をマーク)")
    ],
    outputs=[
        gr.Image(type="pil", label="生成されたマスク画像"),
        gr.Image(type="pil", label="分離された前景画像")
    ],
    title="OptiCut: Automatic Image Segmentation",
    description="元の画像と、背景（黒）または前景（白）としてマークされた編集済み画像をアップロードすると、前景が分離されます。",
    flagging_mode="never",
    submit_btn="Run"
)

iface.launch()