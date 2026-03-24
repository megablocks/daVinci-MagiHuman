# Copyright (c) 2026 SandAI. All Rights Reserved.
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

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModalRunContext:
    run_id: str
    timestamp: str
    staging_dir: Path
    local_output_dir: Path
    save_path_prefix: str
    volume_run_dir: Path


@dataclass(frozen=True)
class ModalStagedInputs:
    image_path: str
    audio_path: str | None
    config_path: str | None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_volume_file(volume_root: Path, path_or_name: str | None) -> Path | None:
    if path_or_name is None:
        return None
    candidate = Path(path_or_name)
    if candidate.is_absolute():
        return candidate
    return volume_root / candidate


def _copy_input(source: Path | None, destination_dir: Path, label: str) -> str | None:
    if source is None:
        return None
    if not source.exists():
        raise FileNotFoundError(f"Missing {label} file: {source}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{label}{source.suffix}"
    shutil.copy2(source, destination)
    return str(destination)


def _resolve_save_path_prefix(
    save_path_prefix: str | None,
    output_path: str | None,
    local_output_dir: Path,
) -> str:
    resolved = save_path_prefix or output_path
    if not resolved:
        raise ValueError("--save_path_prefix (or --output_path) is required.")

    resolved_path = Path(resolved)
    if resolved_path.is_absolute():
        return str(resolved_path)
    return str(local_output_dir / resolved_path)


def build_run_context(
    volume_root: str,
    save_path_prefix: str | None = None,
    output_path: str | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
    local_work_root: str = "/tmp/modal_inference_runs",
) -> ModalRunContext:
    ts = timestamp or _utc_timestamp()
    generated_id = run_id
    if generated_id is None:
        digest = hashlib.sha256(ts.encode("utf-8")).hexdigest()[:8]
        generated_id = f"run_{ts}_{digest}"

    staging_dir = Path(local_work_root) / generated_id
    local_output_dir = staging_dir / "outputs"
    volume_run_dir = Path(volume_root) / "runs" / generated_id
    save_prefix = _resolve_save_path_prefix(save_path_prefix, output_path, local_output_dir)

    local_output_dir.mkdir(parents=True, exist_ok=True)
    return ModalRunContext(
        run_id=generated_id,
        timestamp=ts,
        staging_dir=staging_dir,
        local_output_dir=local_output_dir,
        save_path_prefix=save_prefix,
        volume_run_dir=volume_run_dir,
    )


def stage_modal_inputs(
    volume_root: str,
    run_context: ModalRunContext,
    image_path: str,
    audio_path: str | None = None,
    config_path: str | None = None,
) -> ModalStagedInputs:
    volume = Path(volume_root)
    input_dir = run_context.staging_dir / "inputs"

    staged_image = _copy_input(_resolve_volume_file(volume, image_path), input_dir, "image")
    staged_audio = _copy_input(_resolve_volume_file(volume, audio_path), input_dir, "audio")
    staged_config = _copy_input(_resolve_volume_file(volume, config_path), input_dir, "config")
    if staged_image is None:
        raise ValueError("image_path is required")
    return ModalStagedInputs(image_path=staged_image, audio_path=staged_audio, config_path=staged_config)


def persist_modal_outputs(
    run_context: ModalRunContext,
    output_files: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir = run_context.volume_run_dir / "outputs"
    log_dir = run_context.volume_run_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    persisted_files: list[str] = []
    for output_file in output_files:
        source = Path(output_file)
        if not source.exists():
            continue
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        persisted_files.append(str(destination))

    metadata_path = log_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "persisted_output_files": persisted_files,
        "metadata_path": str(metadata_path),
    }
