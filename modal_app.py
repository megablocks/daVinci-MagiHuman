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

from typing import Any

from inference.common import parse_config
from inference.infra import initialize_infra
from inference.infra.modal_io import build_run_context, persist_modal_outputs, stage_modal_inputs
from inference.model.dit import get_dit
from inference.pipeline import MagiPipeline


def run_modal_inference(
    prompt: str,
    image_path: str,
    volume_root: str,
    audio_path: str | None = None,
    config_path: str | None = None,
    save_path_prefix: str | None = None,
    output_path: str | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    """Run inference from Modal volume inputs and persist artifacts back to volume.

    save_path_prefix/output_path aliasing follows inference/pipeline/entry.py behavior.
    """

    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")

    run_context = build_run_context(
        volume_root=volume_root,
        save_path_prefix=save_path_prefix,
        output_path=output_path,
        run_id=run_id,
        timestamp=timestamp,
    )
    staged_inputs = stage_modal_inputs(
        volume_root=volume_root,
        run_context=run_context,
        image_path=image_path,
        audio_path=audio_path,
        config_path=config_path,
    )

    initialize_infra()
    config = parse_config()
    model = get_dit(config.arch_config, config.engine_config)
    pipeline = MagiPipeline(model, config.evaluation_config)

    optional_kwargs = {key: value for key, value in pipeline_kwargs.items() if value is not None and value is not False}
    output_file = pipeline.run_offline(
        prompt=prompt,
        image=staged_inputs.image_path,
        audio=staged_inputs.audio_path,
        save_path_prefix=run_context.save_path_prefix,
        **optional_kwargs,
    )

    metadata = {
        "run_id": run_context.run_id,
        "timestamp": run_context.timestamp,
        "status": "succeeded",
        "output_file": output_file,
        "save_path_prefix": run_context.save_path_prefix,
    }
    persisted = persist_modal_outputs(run_context, output_files=[output_file], metadata=metadata)

    return {
        "status": "succeeded",
        "run_id": run_context.run_id,
        "timestamp": run_context.timestamp,
        "output_files": {
            "local": [output_file],
            "volume": persisted["persisted_output_files"],
        },
        "metadata": {
            "metadata_path": persisted["metadata_path"],
            "staged_inputs": {
                "image_path": staged_inputs.image_path,
                "audio_path": staged_inputs.audio_path,
                "config_path": staged_inputs.config_path,
            },
        },
    }
