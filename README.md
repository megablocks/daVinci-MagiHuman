![cover](assets/cover.png)


-----

<div align="center">

# daVinci-MagiHuman

### Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model

<p align="center">
  <a href="https://plms.ai">SII-GAIR</a> &nbsp;&amp;&nbsp; <a href="https://sand.ai">Sand.ai</a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2603.21986-b31b1b.svg)](https://arxiv.org/abs/2603.21986)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-HuggingFace-orange)](https://huggingface.co/spaces/SII-GAIR/daVinci-MagiHuman)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-HuggingFace-yellow)](https://huggingface.co/GAIR/daVinci-MagiHuman)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-ee4c2c.svg)](https://pytorch.org/)

</div>

## ✨ Highlights

- 🧠 **Single-Stream Transformer** — A unified 15B-parameter, 40-layer Transformer that jointly processes text, video, and audio via self-attention only. No cross-attention, no multi-stream complexity.
- 🎭 **Exceptional Human-Centric Quality** — Expressive facial performance, natural speech-expression coordination, realistic body motion, and accurate audio-video synchronization.
- 🌍 **Multilingual** — Supports Chinese (Mandarin & Cantonese), English, Japanese, Korean, German, and French.
- ⚡ **Blazing Fast Inference** — Generates a 5-second 256p video in **2 seconds** and a 5-second 1080p video in **38 seconds** on a single H100 GPU.
- 🏆 **State-of-the-Art Results** — Achieves **80.0%** win rate vs Ovi 1.1 and **60.9%** vs LTX 2.3 in pairwise human evaluation over 2,000 comparisons.
- 📦 **Fully Open Source** — We release the complete model stack: base model, distilled model, super-resolution model, and inference code.

## 🎬 Demo

https://github.com/user-attachments/assets/7050a191-38ef-4e36-8b48-0084ccc694f1

https://github.com/user-attachments/assets/c6cc056f-56ca-4285-80f3-bb6052228d23

<table>
<tr valign="top">
<td width="33%"><video src="https://github.com/user-attachments/assets/584d4e13-9956-4ef0-8867-2c78efeac5aa" controls muted width="100%"></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/c5f87f3a-f121-4f34-8d41-8c4b1c24b5e6" controls muted width="100%"></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/0fb467e8-e3a4-4155-9d6b-10b2e018bd7f" controls muted width="100%"></video></td>
</tr>
</table>
<table>
<tr valign="top">
<td width="50%"><video src="https://github.com/user-attachments/assets/800ef0e2-cece-4dc0-9e8f-1524b1c6d326" controls muted width="100%"></video></td>
<td width="50%"><video src="https://github.com/user-attachments/assets/440bf0b9-6c6a-4482-a9a5-90f56e0c0e4d" controls muted width="100%"></video></td>
</tr>
</table>

## 🏗️ Architecture

<div align="center">
<img src="assets/architecture.png" width="90%">
</div>

daVinci-MagiHuman uses a single-stream Transformer that takes text tokens, a reference image latent, and noisy video and audio tokens as input, and jointly denoises the video and audio within a unified token sequence.

Key design choices:

| Component | Description |
|---|---|
| 🥪 **Sandwich Architecture** | First and last 4 layers use modality-specific projections; middle 32 layers share parameters across modalities |
| 🕐 **Timestep-Free Denoising** | No explicit timestep embeddings — the model infers the denoising state directly from input latents |
| 🔀 **Per-Head Gating** | Learned scalar gates with sigmoid activation on each attention head for training stability |
| 🔗 **Unified Conditioning** | Denoising and reference signals handled through a minimal unified interface — no dedicated conditioning branches |

## 📊 Performance

### Quantitative Quality Benchmark

| Model | Visual Quality ↑ | Text Alignment ↑ | Physical Consistency ↑ | WER ↓ |
|---|:---:|:---:|:---:|:---:|
| OVI 1.1 | 4.73 | 4.10 | 4.41 | 40.45% |
| LTX 2.3 | 4.76 | 4.12 | **4.56** | 19.23% |
| **daVinci-MagiHuman** | **4.80** | **4.18** | 4.52 | **14.60%** |

### Human Evaluation (2,000 Pairwise Comparisons)

| Matchup | daVinci-MagiHuman Win | Tie | Opponent Win |
|---|:---:|:---:|:---:|
| vs Ovi 1.1 | **80.0%** | 8.2% | 11.8% |
| vs LTX 2.3 | **60.9%** | 17.2% | 21.9% |

### Inference Speed (5-second video, on a single H100 GPU)

| Resolution | Base (s) | Super-Res (s) | Decode (s) | **Total (s)** |
|---|:---:|:---:|:---:|:---:|
| 256p | 1.6 | — | 0.4 | **2.0** |
| 540p | 1.6 | 5.1 | 1.3 | **8.0** |
| 1080p | 1.6 | 31.0 | 5.8 | **38.4** |

## 🚀 Efficient Inference Techniques

- ⚡ **Latent-Space Super-Resolution** — Two-stage pipeline: generate at low resolution, then refine in latent space (not pixel space), avoiding an extra VAE decode-encode round trip.
- 🔄 **Turbo VAE Decoder** — A lightweight re-trained decoder that substantially reduces decoding overhead.
- 🔧 **Full-Graph Compilation** — [MagiCompiler](https://github.com/SandAI-org/MagiCompiler) fuses operators across Transformer layers for ~1.2x speedup.
- 💨 **Distillation** — DMD-2 distillation enables generation with only 8 denoising steps (no CFG), without sacrificing quality.

## 📦 Getting Started

### Option 1: Docker (Recommended)

```bash
# Pull the MagiCompiler Docker image
docker pull sandai/magi-compiler:latest

# Launch container
docker run -it --gpus all -v /path/to/models:/models sandai/magi-compiler:latest bash

# Install MagiCompiler
git clone https://github.com/SandAI-org/MagiCompiler.git
cd MagiCompiler
pip install -r requirements.txt
pip install .
cd ..

# Clone daVinci-MagiHuman
git clone https://github.com/GAIR-NLP/daVinci-MagiHuman
cd daVinci-MagiHuman
```

### Option 2: Conda

```bash
# Create environment
conda create -n davinci python=3.12
conda activate davinci

# Install PyTorch
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

# Install Flash Attention (Hopper)
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper && python setup.py install && cd ../..

# Install MagiCompiler
git clone https://github.com/SandAI-org/MagiCompiler.git
cd MagiCompiler
pip install -r requirements.txt
pip install .
cd ..

# Clone and install daVinci-MagiHuman
git clone https://github.com/GAIR-NLP/daVinci-MagiHuman
cd daVinci-MagiHuman
pip install -r requirements.txt
```

### Download Model Checkpoints

Download the complete model stack from [HuggingFace](https://huggingface.co/GAIR/daVinci-MagiHuman) and update the paths in the config files under `example/`.

You will also need the following external models:

| Model | Source |
|---|---|
| Text Encoder | [t5gemma-9b-9b-ul2](https://huggingface.co/google/t5gemma-9b-9b-ul2) |
| Audio Model | [stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) |
| VAE | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |

## 🎯 Usage

Before running, update the checkpoint paths in the config files (`example/*/config.json`) to point to your local model directory.

> **Note:** The first run will be slower due to model compilation and cache warmup. Subsequent runs will match the reported inference speeds.

**Base Model (256p)**
```bash
bash example/base/run.sh
```

**Distilled Model (256p, 8 steps, no CFG)**
```bash
bash example/distill/run.sh
```

**Super-Resolution to 540p**
```bash
bash example/sr_540p/run.sh
```

**Super-Resolution to 1080p**
```bash
bash example/sr_1080p/run.sh
```


## ☁️ Run on Modal (GPU)

If you want to run inference on remote GPUs without managing your own CUDA host, you can use [Modal](https://modal.com/) to package this repo and execute jobs on H100/A100 instances.

### Prerequisites

```bash
pip install modal
modal setup
```

`modal setup` opens a browser flow to authenticate your CLI.

### Create and Attach Modal Volumes

Use two persistent volumes:
- `davinci-checkpoints`: all model weights/checkpoints referenced by `example/*/config.json`.
- `davinci-outputs`: generated videos/logs.

```bash
# Create once
modal volume create davinci-checkpoints
modal volume create davinci-outputs

# Optional: inspect
modal volume ls
```

Mount them in your Modal app, for example:

```python
volumes={
  "/checkpoints": modal.Volume.from_name("davinci-checkpoints", create_if_missing=True),
  "/outputs": modal.Volume.from_name("davinci-outputs", create_if_missing=True),
}
```

### Expected Checkpoint Directory Layout

The config files in `example/*/config.json` expect paths like `/path/to/checkpoints/...`. On Modal, map those to `/checkpoints/...` and keep this layout:

```text
/checkpoints/
├── base/                         # engine_config.load for base/sr configs
├── distill/                      # engine_config.load for distill config
├── 540p_sr/                      # evaluation_config.sr_model_path (540p)
├── 1080p_sr/                     # evaluation_config.sr_model_path (1080p)
├── stable-audio-open-1.0/        # evaluation_config.audio_model_path
├── t5/
│   └── t5gemma-9b-9b-ul2/        # evaluation_config.txt_model_path
├── wan_vae/
│   └── Wan2.2-TI2V-5B/           # evaluation_config.vae_model_path
└── turbo_vae/
    ├── TurboV3-Wan22-TinyShallow_7_7.json
    └── checkpoint-340000.ckpt
```

Before running on Modal, update each `example/*/config.json` path from `/path/to/checkpoints/...` to `/checkpoints/...`.

### Example Commands (Local Entrypoint vs Remote Function)

Assuming you define a Modal app file (for example `modal_app.py`) with both a local entrypoint and a remote function:

```bash
# Run local entrypoint logic (submits/coordinates jobs)
modal run modal_app.py::main

# Run remote GPU function directly
modal run modal_app.py::run_inference
```

A practical pattern is:
- `main`: validates args/config and dispatches.
- `run_inference`: executes `torchrun ... inference/pipeline/entry.py ...` inside the GPU container.

### Recommended GPU Profiles

- **H100 (recommended for best latency):** best throughput and shortest wall-clock time for base + SR runs, but highest hourly cost.
- **A100 80GB (balanced):** slower than H100 but typically more available and cheaper; good default for production batches.
- **A100 40GB / smaller GPUs:** possible for base/distill, but SR (especially 1080p) is more likely to hit memory pressure or require longer runtimes.

Rule of thumb: use **H100** for interactive turnaround and **A100** for cost-sensitive batch workloads.

### Troubleshooting

- **Missing CUDA libraries** (`libcuda.so`, `libnvrtc.so`, etc.)
  - Use a CUDA-enabled base image in Modal and install GPU-compatible PyTorch/Flash-Attention builds.
  - Confirm your function is declared with a GPU resource (CPU-only functions will fail on CUDA imports).

- **CUDA OOM / out-of-memory**
  - Start with `example/distill/config.json` (fewer steps) or base 256p first.
  - Reduce parallelism (`GPUS_PER_NODE=1`, no extra workers), shorten clip length, and avoid SR until stable.
  - For SR, prefer H100 or A100 80GB.

- **Path mismatch between config and mounted volumes**
  - Ensure config paths use `/checkpoints/...` exactly as mounted in Modal.
  - Verify expected files exist in the volume before launch (`modal volume get` / shell inspection).

- **Slow first run (cold start + compile warmup)**
  - First invocation is slower due to container start, weight load, and model compile/cache initialization.
  - Keep a warm container (periodic pings) for latency-sensitive workloads; subsequent runs should be much faster.


## ✍️ Prompt Guidance
 
daVinci-MagiHuman uses an **Enhanced Prompt** system that rewrites user inputs into detailed performance directions optimized for avatar-style video generation. For the full system prompt specification, see [`prompts/enhanced_prompt_design.md`](prompts/enhanced_prompt_design.md).

Below is a quick reference for writing effective prompts.

### Output Structure
 
Every enhanced prompt has **three parts**:
 
1. **Main Body** (150–200 words) — A clinical, chronological description of the character's appearance, facial dynamics, vocal delivery, and static cinematography. Written in English regardless of dialogue language.
 
2. **Dialogue** — Repeats all spoken lines in a structured format:
   ```
   Dialogue:
   <character description, language>: "Line content"
   ```
 
3. **Background Sound** — Specifies the most prominent ambient sound:
   ```
   Background Sound:
   <Description of the background sound>
   ```
   Use `<No prominent background sound>` if none.

### Quick Example
 
**User input:** A man in a yellow shirt says "有的人在一起生活一辈子，还带着假面具呢"
 
**Enhanced prompt (abbreviated):**
 
> A young man with short dark hair, wearing a bright yellow polo shirt, sits stationary. His disposition is earnest and slightly agitated... He speaks with a rapid, emphatic tone, his mouth opening wide as he says, "有 的 人 在 一 起 生 活 一 辈 子，还 带 着 假 面 具 呢..." His brow furrows, lip muscles showing distinct dynamics...
>
> Dialogue:
> \<Young man in yellow polo, Mandarin\>: "有 的 人 在 一 起 生 活 一 辈 子，还 带 着 假 面 具 呢..."
>
> Background Sound:
> \<No prominent background sound\>

## 🙏 Acknowledgements

We thank the open-source community, and in particular [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Turbo-VAED](https://github.com/hustvl/Turbo-VAED), for their valuable contributions.

## 📄 License

This project is released under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

## 📖 Citation

```bibtex
@misc{davinci-magihuman-2026,
  title   = {Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model},
  author  = {SII-GAIR and Sand.ai},
  year    = {2026},
  url     = {https://github.com/GAIR-NLP/daVinci-MagiHuman}
}
```
