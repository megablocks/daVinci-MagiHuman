from __future__ import annotations

import shlex
import time
from pathlib import Path

import modal

APP_NAME = "davinci-magihuman"
REPO_DIR = "/root/daVinci-MagiHuman"
CHECKPOINTS_DIR = "/vol/checkpoints"
OUTPUTS_DIR = "/vol/outputs"

# Modal Volumes for persistent model checkpoints and generated artifacts.
checkpoints_volume = modal.Volume.from_name("davinci-magihuman-checkpoints", create_if_missing=True)
outputs_volume = modal.Volume.from_name("davinci-magihuman-outputs", create_if_missing=True)

# CUDA 12.4 wheel index aligns with current Modal NVIDIA runtime.
TORCH_PACKAGES = [
    "torch==2.9.0",
    "torchvision==0.24.0",
    "torchaudio==2.9.0",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsndfile1",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .add_local_dir(".", remote_path=REPO_DIR, copy=True)
    .workdir(REPO_DIR)
    .pip_install_from_requirements("requirements.txt")
    .pip_install(*TORCH_PACKAGES, extra_index_url="https://download.pytorch.org/whl/cu124")
)

app = modal.App(APP_NAME, image=image)


def _upload_if_local(path: str | None, volume: modal.Volume, remote_subdir: str) -> str | None:
    if not path:
        return None

    p = Path(path).expanduser().resolve()
    if not p.exists():
        # Treat as an in-container path that already exists in a mounted volume.
        return path

    remote_path = f"{remote_subdir}/{p.name}"
    with volume.batch_upload() as batch:
        batch.put_file(str(p), remote_path)
    volume.commit()
    return str(Path(OUTPUTS_DIR) / remote_path)


@app.function(
    gpu="H100",
    timeout=60 * 60,
    volumes={
        CHECKPOINTS_DIR: checkpoints_volume,
        OUTPUTS_DIR: outputs_volume,
    },
)
def run_inference(
    prompt: str,
    image_path: str,
    output_path: str,
    audio_path: str | None = None,
    config_path: str | None = None,
) -> str:
    import subprocess

    cmd = [
        "python",
        "-m",
        "inference.pipeline.entry",
        "--prompt",
        prompt,
        "--image_path",
        image_path,
        "--output_path",
        output_path,
    ]

    if audio_path:
        cmd.extend(["--audio_path", audio_path])
    if config_path:
        cmd.extend(["--config-load-path", config_path])

    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=REPO_DIR)

    outputs_volume.commit()
    return output_path


@app.local_entrypoint()
def main(
    prompt: str,
    image_path: str,
    audio_path: str | None = None,
    config_path: str | None = None,
    output_path: str | None = None,
):
    """
    Trigger remote inference from local CLI.

    Example:
      modal run modal_app.py \
        --prompt "A person talking naturally" \
        --image-path ./example/assets/image.png \
        --config-path ./example/base/config.json \
        --output-path /vol/outputs/runs/demo
    """
    run_id = str(int(time.time()))

    remote_image_path = _upload_if_local(image_path, outputs_volume, f"inputs/{run_id}") or image_path
    remote_audio_path = _upload_if_local(audio_path, outputs_volume, f"inputs/{run_id}") if audio_path else None
    remote_config_path = _upload_if_local(config_path, outputs_volume, f"inputs/{run_id}") if config_path else None

    resolved_output_path = output_path or str(Path(OUTPUTS_DIR) / "runs" / run_id)
    if not resolved_output_path.startswith("/"):
        resolved_output_path = str(Path(OUTPUTS_DIR) / resolved_output_path)

    result = run_inference.remote(
        prompt=prompt,
        image_path=remote_image_path,
        audio_path=remote_audio_path,
        config_path=remote_config_path,
        output_path=resolved_output_path,
    )

    print(f"Inference complete. Outputs written under: {result}")
    print(f"Checkpoints volume mount path in container: {CHECKPOINTS_DIR}")
    print(f"Artifacts volume mount path in container: {OUTPUTS_DIR}")

