from __future__ import annotations

import shutil
from pathlib import Path
import re
from textwrap import dedent

from pipeline.model_catalog import ModelSpec


_BACKEND_TEMPLATE_ROOT = Path(__file__).resolve().parent / "backend_templates"
_TRITON_TENSOR_SPEC_RE = re.compile(r"^(?P<name>[^:]+): (?P<dtype>[A-Z0-9]+)\[(?P<dims>[^\]]+)\](?: (?P<description>.*))?$")


def _parse_triton_tensor_spec(entry: str) -> tuple[str, str, list[int], str]:
    match = _TRITON_TENSOR_SPEC_RE.match(entry)
    if match is None:
        raise RuntimeError(f"Could not parse Triton tensor contract entry: {entry}")

    dims = [
        int(dim.strip()) if dim.strip().lstrip("-").isdigit() else -1
        for dim in match.group("dims").split(",")
    ]
    return (
        match.group("name").strip(),
        match.group("dtype").strip(),
        dims,
        (match.group("description") or "").strip(),
    )


def _render_pbtxt_tensor_block(kind: str, entry: str) -> str:
    name, dtype, dims, description = _parse_triton_tensor_spec(entry)
    dim_str = ", ".join(str(d) for d in dims)
    comment = f"# {description}\n" if description else ""
    return (
        f"{comment}{kind} [\n"
        "  {\n"
        f'    name: "{name}"\n'
        f"    data_type: TYPE_{dtype}\n"
        f"    dims: [ {dim_str} ]\n"
        "  }\n"
        "]\n"
    )


def _render_runtime_config(spec: ModelSpec) -> str:
    input_blocks = "".join(_render_pbtxt_tensor_block("input", entry) for entry in spec.triton_inputs)
    output_blocks = "".join(_render_pbtxt_tensor_block("output", entry) for entry in spec.triton_outputs)
    backend = spec.triton_backend or "python"
    return dedent(
        f"""\
        name: "{spec.repository_model_name}"
        backend: "{backend}"
        max_batch_size: 0
        instance_group [
          {{
            kind: KIND_CPU
            count: 1
          }}
        ]

        {input_blocks}{output_blocks}parameters: {{
          key: "prepared_by"
          value: {{
            string_value: "pipeline.manual_runtime"
          }}
        }}
        """
    )


def _render_runtime_model(spec: ModelSpec) -> str:
    template_path = _BACKEND_TEMPLATE_ROOT / f"{spec.model_id}.py"
    if not template_path.is_file():
        raise RuntimeError(f"No runnable backend template exists for {spec.model_id} at {template_path}")
    return template_path.read_text(encoding="utf-8")


def _snapshot_download(spec: ModelSpec, target_dir: Path, cache_dir: Path | None) -> None:
    if spec.hf_repo_id is None or spec.revision is None:
        raise RuntimeError(f"{spec.model_id} does not define a downloadable upstream snapshot.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to materialize manual model snapshots. "
            "Run uv sync or bun install after pulling this change."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=spec.hf_repo_id,
        revision=spec.revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(spec.snapshot_allow_patterns) or None,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def materialize_manual_runtime_model(
    output_root: Path,
    spec: ModelSpec,
    cache_dir: Path | None,
) -> dict[str, object]:
    if spec.repository_model_name is None:
        raise RuntimeError(f"{spec.model_id} does not define a Triton repository name.")

    model_root = output_root / spec.repository_model_name
    version_root = model_root / "1"
    upstream_root = version_root / "upstream"
    version_root.mkdir(parents=True, exist_ok=True)

    _snapshot_download(spec, upstream_root, cache_dir)

    config_path = model_root / "config.pbtxt"
    config_path.write_text(_render_runtime_config(spec), encoding="utf-8")

    model_path = version_root / "model.py"
    model_path.write_text(_render_runtime_model(spec), encoding="utf-8")

    requirements_path = model_root / "requirements.txt"
    requirements_path.write_text(
        "\n".join(spec.runtime_pip_packages) + ("\n" if spec.runtime_pip_packages else ""),
        encoding="utf-8",
    )

    return {
        "installed": True,
        "materialization_mode": "opt_in_manual_prepare",
        "repository_path": str(version_root.relative_to(output_root)),
        "runtime_files": [
            str(config_path.relative_to(output_root)),
            str(model_path.relative_to(output_root)),
            str(requirements_path.relative_to(output_root)),
            str(upstream_root.relative_to(output_root)),
        ],
        "snapshot_allow_patterns": list(spec.snapshot_allow_patterns),
        "runtime_pip_packages": list(spec.runtime_pip_packages),
    }
