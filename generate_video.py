# Copyright 2026 Linum Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File: generate_video.py
Description: Streamlined script for generating a single video from a text prompt.

Weights are automatically downloaded from HuggingFace Hub on first run.
"""

import argparse
import os

import torch
import torchvision.io
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

from linum_v2.models.text2video import Linum_v2_Text2Video


# ----------------------------------------------------------------------------
# HuggingFace Hub Configuration
# ----------------------------------------------------------------------------
HF_REPO_360P = "Linum-AI/linum-v2-360p"
HF_REPO_720P = "Linum-AI/linum-v2-720p"

# Files in HF repos
DIT_FILE = {
    "360p": "dit/360p.safetensors",
    "720p": "dit/720p.safetensors",
}
VAE_FILE = "vae/vae.safetensors"
T5_ENCODER_DIR = "t5/text_encoder"
T5_TOKENIZER_DIR = "t5/tokenizer"

FPS = 24

RESOLUTIONS = {
    "360p": (360, 640),
    "720p": (720, 1280),
}


# ----------------------------------------------------------------------------
# Weight Download
# ----------------------------------------------------------------------------
def download_weights(resolution: str) -> dict:
    """
    Download model weights from HuggingFace Hub.

    Args:
        resolution: Either "360p" or "720p"

    Returns:
        Dictionary with paths to downloaded weights
    """
    hf_repo = HF_REPO_720P if resolution == "720p" else HF_REPO_360P

    print(f"Downloading Linum V2 {resolution} weights from HuggingFace Hub...")
    print(f"Repository: {hf_repo}")

    # Download DiT checkpoint
    print("  - Downloading DiT model...")
    dit_path = hf_hub_download(
        repo_id=hf_repo,
        filename=DIT_FILE[resolution],
    )

    # Download VAE checkpoint
    print("  - Downloading VAE...")
    vae_path = hf_hub_download(
        repo_id=hf_repo,
        filename=VAE_FILE,
    )

    # Download T5 encoder (directory with multiple files)
    print("  - Downloading T5 encoder...")
    t5_encoder_path = snapshot_download(
        repo_id=hf_repo,
        allow_patterns=f"{T5_ENCODER_DIR}/*",
    )
    t5_encoder_path = os.path.join(t5_encoder_path, T5_ENCODER_DIR)

    # Download T5 tokenizer (directory with multiple files)
    print("  - Downloading T5 tokenizer...")
    t5_tokenizer_path = snapshot_download(
        repo_id=hf_repo,
        allow_patterns=f"{T5_TOKENIZER_DIR}/*",
    )
    t5_tokenizer_path = os.path.join(t5_tokenizer_path, T5_TOKENIZER_DIR)

    print("Download complete!")

    return {
        "dit": dit_path,
        "vae": vae_path,
        "t5_encoder": t5_encoder_path,
        "t5_tokenizer": t5_tokenizer_path,
    }


# ----------------------------------------------------------------------------
# Argument Parsing
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a single video from a text prompt."
    )

    # Prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="In a charming hand-drawn 2D animation style, a rust-orange fox with cream chest fur and alert triangular ears grips a cherry-red steering wheel with both paws, its bushy tail curled on the passenger seat. Stylized trees and pastel houses whoosh past the windows in smooth parallax layers. The fox's golden eyes focus intently ahead, whiskers twitching as it navigates a winding country road rendered in soft watercolor textures.",
        help="Text prompt describing the video to generate.",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video file path. Default: output.mp4",
    )

    # Resolution
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=["360p", "720p"],
        help="Video resolution. Default: 720p",
    )

    # Duration
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Video duration in seconds. Default: 2.0",
    )

    # Generation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=20,
        help="Random seed for reproducibility. Default: 20",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=10.0,
        help="Classifier-free guidance scale. Default: 10.0",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of diffusion steps. Default: 50",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt. Default: empty string",
    )
    parser.add_argument(
        "--apg_rescale",
        type=float,
        default=20.0,
        help="APG rescaling factor. Default: 20.0",
    )

    # Local weight paths (optional - skip HF download if provided)
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to local DiT checkpoint (skips HuggingFace download)",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        help="Path to local VAE weights (skips HuggingFace download)",
    )
    parser.add_argument(
        "--t5-encoder-path",
        type=str,
        help="Path to local T5 encoder (skips HuggingFace download)",
    )
    parser.add_argument(
        "--t5-tokenizer-path",
        type=str,
        help="Path to local T5 tokenizer (skips HuggingFace download)",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------
def main() -> None:
    """Main function for video generation."""
    args = parse_args()

    # Check for CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. No GPU detected.")
        return

    # Get resolution
    height, width = RESOLUTIONS[args.resolution]

    # Determine weight paths (download if not provided locally)
    if args.model_path and args.vae_path and args.t5_encoder_path and args.t5_tokenizer_path:
        # Use local weights
        weight_paths = {
            "dit": args.model_path,
            "vae": args.vae_path,
            "t5_encoder": args.t5_encoder_path,
            "t5_tokenizer": args.t5_tokenizer_path,
        }
        print("Using local weights")
    else:
        # Download from HuggingFace Hub
        weight_paths = download_weights(args.resolution)

    # Calculate frame count from duration
    # Frame count must satisfy: (num_frames - 1) % fps == 0
    num_frames = int(FPS * args.duration) + 1
    if (num_frames - 1) % FPS != 0:
        num_frames = ((num_frames - 1) // FPS) * FPS + 1

    print(f"\n{'='*60}")
    print(f"Linum V2 Video Generation")
    print(f"{'='*60}")
    print(f"Resolution: {args.resolution} ({height}x{width})")
    print(f"Duration: {args.duration}s ({num_frames} frames at {FPS} FPS)")
    print(f"Seed: {args.seed}")
    print(f"CFG: {args.cfg}")
    print(f"Steps: {args.num_steps}")
    print(f"APG Rescale: {args.apg_rescale}")
    print(f"{'='*60}")
    print(f"\nPrompt: {args.prompt}\n")

    # Load model
    print("Loading model...")
    model = Linum_v2_Text2Video.from_pretrained(
        checkpoint_path=weight_paths["dit"],
    )
    model = model.to('cuda').eval()
    print("Model loaded successfully!")

    # Generate video
    print("Generating video...")
    with torch.inference_mode():
        video_tensors = model.generate(
            input_prompt=args.prompt,
            size=(height, width),
            frame_num=num_frames,
            sampling_steps=args.num_steps,
            guide_scale=args.cfg,
            n_prompt=args.negative_prompt,
            t5_tokenizer_path=weight_paths["t5_tokenizer"],
            t5_model_path=weight_paths["t5_encoder"],
            vae_weights_path=weight_paths["vae"],
            seeds=[args.seed],
            apg_rescale=args.apg_rescale,
            device='cuda',
            quiet=False,
        )

    # Get the video tensor and save
    video_tensor = video_tensors[0]  # Shape: (C, T, H, W), uint8

    # Rearrange from (C, T, H, W) to (T, H, W, C) for torchvision
    video_tensor = video_tensor.permute(1, 2, 3, 0)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save video
    torchvision.io.write_video(
        str(output_path),
        video_tensor.cpu(),
        fps=FPS,
        video_codec="h264",
        options={"crf": "18"},
    )

    print(f"\nVideo saved to: {output_path}")


if __name__ == '__main__':
    main()
