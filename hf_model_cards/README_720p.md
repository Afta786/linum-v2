---
license: apache-2.0
tags:
  - text-to-video
  - video-generation
  - diffusion
  - dit
pipeline_tag: text-to-video
library_name: linum-v2
---

# Linum v2 - 720p

Small text-to-video generation model trained from scratch by [Linum AI](https://linum.ai). [Read the launch blog post](https://www.linum.ai/field-notes/launch-linum-v2).

## Model Description

Linum V2 is a 2B parameter Diffusion Transformer (DiT) based text-to-video model that generates 720p (1280x720) videos at 24 FPS from text prompts.

| Property | Value |
|----------|-------|
| Resolution | 1280x720 (720p) |
| Frame Rate | 24 FPS |
| Duration | 2-5 seconds |
| Parameters | 2B |
| Architecture | DiT + T5-XXL + WAN 2.1 VAE |

## Quick Start

**See the full documentation at: [GitHub - Linum-AI/linum-v2](https://github.com/Linum-AI/linum-v2)**

First, install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and generate your first video:

```bash
git clone https://github.com/Linum-AI/linum-v2.git
cd linum-v2
uv sync
uv run python generate_video.py \
    --prompt "In a charming hand-drawn 2D animation style, a rust-orange fox with cream chest fur and alert triangular ears grips a cherry-red steering wheel with both paws, its bushy tail curled on the passenger seat. Stylized trees and pastel houses whoosh past the windows in smooth parallax layers. The fox's golden eyes focus intently ahead, whiskers twitching as it navigates a winding country road rendered in soft watercolor textures." \
    --output fox.mp4 \
    --seed 20 \
    --cfg 7.0
```

<video src="https://huggingface.co/Linum-AI/linum-v2-720p/resolve/main/fox_720p_demo.mov" controls autoplay muted loop width="100%"></video>

Weights are downloaded automatically on first run (~20GB).

### Speed Benchmarks (H100, 50 steps)

| Resolution | Duration | Generation Time |
|------------|----------|-----------------|
| 360p | 2 seconds | ~40 seconds |
| 360p | 5 seconds | ~2 minutes |
| 720p | 2 seconds | ~4 minutes |
| 720p | 5 seconds | ~15 minutes |

For lower VRAM, use the [360p model](https://huggingface.co/Linum-AI/linum-v2-360p).

## Files

```
├── dit/
│   └── 720p.safetensors      # DiT model weights
├── vae/
│   └── vae.safetensors       # WAN 2.1 Video VAE
└── t5/
    ├── text_encoder/         # T5-XXL encoder
    └── tokenizer/            # T5 tokenizer
```

## License

[Apache 2.0](LICENSE)

## Citation

```bibtex
@software{linum_v2_2026,
  title = {Linum V2: Text-to-Video Generation},
  author = {Linum AI},
  year = {2026},
  url = {https://github.com/Linum-AI/linum-v2}
}
```
