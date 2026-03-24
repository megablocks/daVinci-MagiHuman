#!/usr/bin/env python3
"""Run daVinci-MagiHuman inference via a Modal remote function.

This script keeps user-facing knobs aligned with existing `example/*/run.sh` scripts,
then translates example config.json values into a single payload for a Modal function.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "example/assets/prompt.txt"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "example/assets/image.png"

PRESET_CONFIGS = {
    "base": PROJECT_ROOT / "example/base/config.json",
    "distill": PROJECT_ROOT / "example/distill/config.json",
    "sr_540p": PROJECT_ROOT / "example/sr_540p/config.json",
    "sr_1080p": PROJECT_ROOT / "example/sr_1080p/config.json",
}

PRESET_DEFAULTS = {
    "base": {
        "seconds": 10,
        "br_width": 448,
        "br_height": 256,
        "sr_width": None,
        "sr_height": None,
        "output_prefix": "output_example_base",
        "env": {},
    },
    "distill": {
        "seconds": 10,
        "br_width": 448,
        "br_height": 256,
        "sr_width": None,
        "sr_height": None,
        "output_prefix": "output_example_distill",
        "env": {},
    },
    "sr_540p": {
        "seconds": 10,
        "br_width": 448,
        "br_height": 256,
        "sr_width": 896,
        "sr_height": 512,
        "output_prefix": "output_example_sr_540p",
        "env": {"CPU_OFFLOAD": "true"},
    },
    "sr_1080p": {
        "seconds": 10,
        "br_width": 448,
        "br_height": 256,
        "sr_width": 1920,
        "sr_height": 1088,
        "output_prefix": "output_example_sr_1080p",
        "env": {"SR2_1080": "true", "CPU_OFFLOAD": "true"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on Modal using modal_app.py remote function.")
    parser.add_argument(
        "--example",
        choices=sorted(PRESET_CONFIGS.keys()),
        default="base",
        help="Example preset matching the existing shell scripts.",
    )
    parser.add_argument(
        "--config-load-path",
        type=Path,
        default=None,
        help="Override config path. Defaults to the selected preset's config.json.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt. Defaults to example/assets/prompt.txt contents.")
    parser.add_argument("--image_path", type=Path, default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--audio_path", type=Path, default=None, help="Optional input audio path.")

    parser.add_argument("--seconds", type=int, default=None)
    parser.add_argument("--br_width", type=int, default=None)
    parser.add_argument("--br_height", type=int, default=None)
    parser.add_argument("--sr_width", type=int, default=None)
    parser.add_argument("--sr_height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--modal-app-path", type=Path, default=PROJECT_ROOT / "modal_app.py")
    parser.add_argument("--modal-function", type=str, default="run_inference")
    parser.add_argument(
        "--modal-bin",
        type=str,
        default="modal",
        help="Modal CLI executable (default: modal).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated payload and Modal command without executing remote call.",
    )

    return parser.parse_args()


def _load_prompt(user_prompt: str | None) -> str:
    if user_prompt is not None:
        return user_prompt
    return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _output_path(example: str, user_output: str | None) -> str:
    if user_output:
        return user_output
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{PRESET_DEFAULTS[example]['output_prefix']}_{timestamp}"


def _resolve_runtime_value(example: str, field: str, user_value: int | None) -> int | None:
    return user_value if user_value is not None else PRESET_DEFAULTS[example][field]


def build_payload(args: argparse.Namespace) -> dict:
    config_path = (args.config_load_path or PRESET_CONFIGS[args.example]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    runtime = {
        "prompt": _load_prompt(args.prompt),
        "image_path": str(args.image_path.resolve()),
        "audio_path": str(args.audio_path.resolve()) if args.audio_path else None,
        "seconds": _resolve_runtime_value(args.example, "seconds", args.seconds),
        "br_width": _resolve_runtime_value(args.example, "br_width", args.br_width),
        "br_height": _resolve_runtime_value(args.example, "br_height", args.br_height),
        "sr_width": _resolve_runtime_value(args.example, "sr_width", args.sr_width),
        "sr_height": _resolve_runtime_value(args.example, "sr_height", args.sr_height),
        "seed": args.seed,
        "output_path": _output_path(args.example, args.output_path),
    }

    payload = {
        "preset": args.example,
        "config_load_path": str(config_path),
        "engine_config": config.get("engine_config", {}),
        "evaluation_config": config.get("evaluation_config", {}),
        "runtime": {k: v for k, v in runtime.items() if v is not None},
        "env": PRESET_DEFAULTS[args.example]["env"],
    }
    return payload


def run_modal(args: argparse.Namespace, payload: dict) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False)
    app_function = f"{args.modal_app_path}::{args.modal_function}"
    cmd = [args.modal_bin, "run", app_function, "--payload-json", payload_json]

    print("[modal] command:", " ".join(cmd))
    if args.dry_run:
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    print("[modal] payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    run_modal(args, payload)


if __name__ == "__main__":
    main()
