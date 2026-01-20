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

# Linum v2 - 360p

Small text-to-video generation model trained from scratch by [Linum AI](https://linum.ai). Lower VRAM requirements than the 720p variant. [Read the launch blog post](https://www.linum.ai/field-notes/launch-linum-v2).

## Model Description

Linum V2 is a 2B parameter Diffusion Transformer (DiT) based text-to-video model that generates 360p (640x360) videos at 24 FPS from text prompts.

| Property | Value |
|----------|-------|
| Resolution | 640x360 (360p) |
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
    --prompt "A cute 3D animated baby goat with shaggy gray fur, a fluffy white chin tuft, and stubby curved horns perches on a round wooden stool. Warm golden studio lights bounce off its glossy cherry-red acoustic guitar as it rhythmically strums with a confident hoof, hind legs dangling. Framed family portraits of other barnyard animals line the cream-colored walls, a leafy potted ficus sits in the back corner, and dust motes drift through the cozy, sun-speckled room." \
    --output goat.mp4 \
    --seed 16 \
    --cfg 7.0 \
    --resolution 360p
```

<video src="https://huggingface.co/Linum-AI/linum-v2-360p/resolve/main/goat_360p_demo.mp4" controls autoplay muted loop width="100%"></video>

Weights are downloaded automatically on first run (~20GB).

For higher quality, use the [720p model](https://huggingface.co/Linum-AI/linum-v2-720p) (requires more VRAM).

## Files

```
├── dit/
│   └── 360p.safetensors      # DiT model weights
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
