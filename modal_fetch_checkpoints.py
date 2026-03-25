from __future__ import annotations

import modal

CHECKPOINTS_DIR = "/vol/checkpoints"
checkpoints_volume = modal.Volume.from_name("davinci-magihuman-checkpoints", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")
app = modal.App("davinci-magihuman-fetch", image=image)


@app.function(
    timeout=60 * 60 * 3,
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
)
def fetch_gair_repo() -> str:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="GAIR/daVinci-MagiHuman",
        repo_type="model",
        local_dir=CHECKPOINTS_DIR,
        allow_patterns=[
            "base/*",
            "turbo_vae/*",
            "config.json",
            "README.md",
        ],
    )

    checkpoints_volume.commit()
    return "Fetched GAIR base + turbo_vae to /vol/checkpoints"


@app.function(
    timeout=60 * 60 * 6,
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
)
def fetch_external_models() -> str:
    import os
    from huggingface_hub import snapshot_download

    targets = [
        {
            "repo": "stabilityai/stable-audio-open-1.0",
            "local_dir": f"{CHECKPOINTS_DIR}/stable-audio-open-1.0",
            "allow_patterns": None,
        },
        {
            "repo": "google/t5gemma-9b-9b-ul2",
            "local_dir": f"{CHECKPOINTS_DIR}/t5/t5gemma-9b-9b-ul2",
            "allow_patterns": None,
        },
        {
            "repo": "Wan-AI/Wan2.2-TI2V-5B",
            "local_dir": f"{CHECKPOINTS_DIR}/wan_vae/Wan2.2-TI2V-5B",
            "allow_patterns": ["Wan2.2_VAE.pth", "README.md", "config.json"],
        },
    ]

    results = []
    for t in targets:
        os.makedirs(t["local_dir"], exist_ok=True)
        try:
            snapshot_download(
                repo_id=t["repo"],
                repo_type="model",
                local_dir=t["local_dir"],
                allow_patterns=t["allow_patterns"],
            )
            results.append(f"OK:{t['repo']}")
        except Exception as e:
            results.append(f"FAIL:{t['repo']}:{type(e).__name__}:{e}")

    checkpoints_volume.commit()
    return " | ".join(results)


@app.local_entrypoint()
def main(fetch_external: bool = True):
    print(fetch_gair_repo.remote())
    if fetch_external:
        print(fetch_external_models.remote())
