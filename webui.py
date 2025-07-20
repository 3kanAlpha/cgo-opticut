import gradio as gr
from opt_solver_cielab import process_image

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="切り抜きたい画像をアップロード"),
        gr.Image(type="pil", label="編集済み画像をアップロード (黒で背景をマーク、白で前景をマーク)")
    ],
    outputs=[
        gr.Image(type="pil", label="生成されたマスク画像", format="png"),
        gr.Image(type="pil", label="分離された前景画像", format="png")
    ],
    title="OptiCut: Automatic Image Segmentation",
    description="切り抜きたい画像と、背景を黒/前景を白でマークされた編集済み画像をアップロードすると、前景が分離されます。",
    flagging_mode="never",
    submit_btn="Run"
)

iface.launch(inbrowser=True)