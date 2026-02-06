import gradio as gr
import torch
from pathlib import Path

from linum_v2.models.text2video import Linum_v2_Text2Video

FPS = 24
RESOLUTIONS = {
    "360p": (360, 640),
    "720p": (720, 1280),
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_video(prompt, resolution):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height, width = RESOLUTIONS[resolution]

    model = Linum_v2_Text2Video(
        resolution=resolution,
        device=device,
    )

    video = model.generate(
        prompt=prompt,
        height=height,
        width=width,
        fps=FPS,
    )

    output_path = OUTPUT_DIR / "linum_output.mp4"
    model.save_video(video, output_path)

    return str(output_path)

with gr.Blocks(title="Linum v2 – Text to Video") as demo:
    gr.Markdown("# 🎥 Linum v2 – Text to Video")
    gr.Markdown("Generate AI videos from text using Linum v2")

    prompt = gr.Textbox(
        label="Text Prompt",
        placeholder="A cinematic shot of a fox running through snow, ultra realistic",
        lines=3,
    )

    resolution = gr.Radio(
        ["360p", "720p"],
        value="360p",
        label="Resolution",
    )

    generate_btn = gr.Button("Generate Video 🚀")
    output_video = gr.Video(label="Generated Video")

    generate_btn.click(
        fn=generate_video,
        inputs=[prompt, resolution],
        outputs=output_video,
    )

demo.launch()
