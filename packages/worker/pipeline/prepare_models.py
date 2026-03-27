from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Callable
from urllib.parse import quote
from urllib.error import URLError
from urllib.request import Request, urlopen

from pipeline.model_catalog import (
    AUTO_DOWNLOAD_LANE,
    HOLD_LANE,
    MANUAL_PLANNED_LANE,
    ModelArtifact,
    ModelSpec,
    get_model_spec,
    get_profile_model_ids,
)
from pipeline.manual_runtime import materialize_manual_runtime_model

DownloadFn = Callable[[ModelSpec, ModelArtifact, Path | None], Path]
MANIFEST_SCHEMA_VERSION = 2
_TRITON_TENSOR_SPEC_RE = re.compile(r"^(?P<name>[^:]+): (?P<dtype>[A-Z0-9]+)\[(?P<dims>[^\]]+)\](?: (?P<description>.*))?$")


class ModelPreparationError(RuntimeError):
    pass


def _default_downloader(spec: ModelSpec, artifact: ModelArtifact, cache_dir: Path | None) -> Path:
    if spec.hf_repo_id is None or spec.revision is None:
        raise ModelPreparationError(f"{spec.model_id} does not define a downloadable upstream artifact.")

    cache_base = cache_dir or Path(".cache") / "model-downloads"
    target_path = cache_base / spec.hf_repo_id.replace("/", "--") / spec.revision / artifact.hf_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        return target_path

    url = f"https://huggingface.co/{spec.hf_repo_id}/resolve/{spec.revision}/{quote(artifact.hf_path, safe='/')}"
    request = Request(url, headers={"User-Agent": "triton-playground/0.1"})

    temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")

    try:
        with urlopen(request) as response, temporary_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except URLError:
        subprocess.run(
            ["curl", "-sS", "-L", url, "-o", str(temporary_path)],
            check=True,
        )

    temporary_path.replace(target_path)
    return target_path


def _sha256_for(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return

    actual = _sha256_for(path)
    if actual != expected_sha256:
        raise ModelPreparationError(
            f"SHA256 mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )


def _skip_reason(spec: ModelSpec) -> str:
    if spec.serve_status == MANUAL_PLANNED_LANE:
        return "model is cataloged for manual download and planned serving"

    if spec.serve_status == HOLD_LANE:
        return "model is on hold pending provenance review"

    return "model is not approved for automatic download"


def _prepare_triton_model(
    output_root: Path,
    spec: ModelSpec,
    cache_dir: Path | None,
    downloader: DownloadFn,
) -> dict[str, object]:
    if spec.repository_model_name is None:
        raise ModelPreparationError(f"{spec.model_id} does not define a Triton repository name.")

    install_dir = output_root / spec.repository_model_name / "1"
    install_dir.mkdir(parents=True, exist_ok=True)

    downloaded_artifacts: list[dict[str, object]] = []
    for artifact in spec.artifacts:
        downloaded_path = downloader(spec, artifact, cache_dir)
        _verify_sha256(downloaded_path, artifact.sha256)

        target_path = install_dir / artifact.target_name
        shutil.copy2(downloaded_path, target_path)
        downloaded_artifacts.append(
            {
                "hf_path": artifact.hf_path,
                "target_path": str(target_path.relative_to(output_root)),
                "sha256": artifact.sha256,
            }
        )

    return {
        "installed": True,
        "repository_path": str(install_dir.relative_to(output_root)),
        "downloaded_artifacts": downloaded_artifacts,
    }


def _parse_triton_tensor_spec(entry: str) -> tuple[str, str, list[int], str]:
    match = _TRITON_TENSOR_SPEC_RE.match(entry)
    if match is None:
        raise ModelPreparationError(f"Could not parse Triton tensor contract entry: {entry}")

    dims: list[int] = []
    for raw_dim in match.group("dims").split(","):
        stripped = raw_dim.strip()
        dims.append(int(stripped) if stripped.lstrip("-").isdigit() else -1)

    return (
        match.group("name").strip(),
        match.group("dtype").strip(),
        dims,
        (match.group("description") or "").strip(),
    )


def _render_pbtxt_tensor_block(kind: str, entry: str) -> str:
    name, dtype, dims, description = _parse_triton_tensor_spec(entry)
    dim_lines = "\n".join(f"      {dim}" for dim in dims)
    comment = f"# {description}\n" if description else ""
    return (
        f"{comment}{kind} [\n"
        "  {\n"
        f'    name: "{name}"\n'
        f"    data_type: TYPE_{dtype}\n"
        "    dims: [\n"
        f"{dim_lines}\n"
        "    ]\n"
        "  }\n"
        "]\n"
    )


def _render_manual_config_template(spec: ModelSpec) -> str:
    input_blocks = "".join(_render_pbtxt_tensor_block("input", entry) for entry in spec.triton_inputs)
    output_blocks = "".join(_render_pbtxt_tensor_block("output", entry) for entry in spec.triton_outputs)
    backend = spec.triton_backend or "python"
    return dedent(
        f"""\
        # Manual Triton backend template for {spec.model_id}
        # Copy this file to config.pbtxt inside a real Triton model repository only after
        # you have replaced 1/model.py.template with a working backend implementation.
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
          key: "manual_setup"
          value: {{
            string_value: "Replace model.py.template with a real backend implementation before starting Triton."
          }}
        }}
        """
    )


def _render_manual_model_template(spec: ModelSpec) -> str:
    input_contract = "\n".join(f"    # - {entry}" for entry in spec.triton_inputs) or "    # - no inputs recorded"
    output_contract = "\n".join(f"    # - {entry}" for entry in spec.triton_outputs) or "    # - no outputs recorded"
    return dedent(
        f'''\
        """
        Manual Triton Python backend template for {spec.model_id}.

        Replace this file with the real adapter before renaming it to model.py in a mounted
        Triton repository.
        """

        import triton_python_backend_utils as pb_utils


        class TritonPythonModel:
            def initialize(self, args):
                raise pb_utils.TritonModelException(
                    "Manual backend stub for {spec.model_id} is not implemented yet. "
                    "Wire the upstream weights and replace 1/model.py.template first."
                )

            def execute(self, requests):
                raise pb_utils.TritonModelException(
                    "Manual backend stub for {spec.model_id} is not implemented yet."
                )


        # Expected inputs:
        {input_contract}
        #
        # Expected outputs:
        {output_contract}
        '''
    )


def _write_manual_stub(stub_root: Path, spec: ModelSpec) -> dict[str, object]:
    if spec.repository_model_name is None:
        raise ModelPreparationError(f"{spec.model_id} does not define a Triton repository name.")

    model_root = stub_root / spec.repository_model_name
    version_root = model_root / "1"
    version_root.mkdir(parents=True, exist_ok=True)

    readme_path = model_root / "README.md"
    readme_path.write_text(
        dedent(
            f"""\
            # {spec.model_id}

            Manual Triton backend scaffold for `{spec.repository_model_name}`.

            - Upstream: `{spec.hf_repo_id}`
            - Revision: `{spec.revision}`
            - Backend: `{spec.triton_backend or "python"}`
            - Lane: `{spec.serve_status}`

            Next action:
            {spec.next_action}

            Expected input contract:
            {chr(10).join(f"- `{entry}`" for entry in spec.triton_inputs) or "- none recorded"}

            Expected output contract:
            {chr(10).join(f"- `{entry}`" for entry in spec.triton_outputs) or "- none recorded"}

            Bring-up steps:
            1. Copy `config.pbtxt.template` to `config.pbtxt` in a real Triton model repository.
            2. Replace `1/model.py.template` with the actual backend implementation and rename it to `model.py`.
            3. Mount the required weights and any tokenizer assets under this model directory.
            4. Start Triton only after the backend can satisfy the recorded tensor contract.
            """
        ),
        encoding="utf-8",
    )

    config_template_path = model_root / "config.pbtxt.template"
    config_template_path.write_text(_render_manual_config_template(spec), encoding="utf-8")

    model_template_path = version_root / "model.py.template"
    model_template_path.write_text(_render_manual_model_template(spec), encoding="utf-8")

    return {
        "manual_stub_path": str(model_root),
        "manual_stub_files": [
            str(readme_path.relative_to(stub_root)),
            str(config_template_path.relative_to(stub_root)),
            str(model_template_path.relative_to(stub_root)),
        ],
    }


def prepare_model_repository(
    output_root: Path,
    model_ids: list[str],
    cache_dir: Path | None = None,
    downloader: DownloadFn = _default_downloader,
    manual_stub_root: Path | None = None,
    materialize_manual_models: bool = False,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    if manual_stub_root is not None:
        manual_stub_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "baseline_profile": "baseline",
            "planning_profiles": ["stt", "localize", "catalog"],
            "allowed_serve_statuses": [AUTO_DOWNLOAD_LANE, MANUAL_PLANNED_LANE, HOLD_LANE],
            "manual_materialization_opt_in": True,
        },
        "models": [],
    }
    if manual_stub_root is not None:
        manifest["manual_stub_root"] = str(manual_stub_root)

    for model_id in model_ids:
        spec = get_model_spec(model_id)
        record: dict[str, object] = spec.to_dict()

        if not spec.approved_for_auto_download:
            if spec.serve_status == MANUAL_PLANNED_LANE and materialize_manual_models:
                record.update(materialize_manual_runtime_model(output_root, spec, cache_dir))
            else:
                record["installed"] = False
                record["reason"] = _skip_reason(spec)
                if spec.serve_status == MANUAL_PLANNED_LANE and manual_stub_root is not None:
                    record.update(_write_manual_stub(manual_stub_root, spec))
            manifest["models"].append(record)
            continue

        if spec.serve_status != AUTO_DOWNLOAD_LANE:
            raise ModelPreparationError(
                f"{spec.model_id} is marked for automatic download but is not in the auto-serve lane"
            )

        install_record = _prepare_triton_model(output_root, spec, cache_dir, downloader)
        record.update(install_record)
        manifest["models"].append(record)

    return manifest


def write_manifest(output_root: Path, manifest: dict[str, object]) -> Path:
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Triton model repository.")
    parser.add_argument("--profile", default="baseline", help="approved model profile to materialize")
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="extra approved model ids to include in addition to the profile",
    )
    parser.add_argument("--output-root", default="model_repository", help="target Triton model repository")
    parser.add_argument("--cache-dir", default=None, help="optional Hugging Face cache directory")
    parser.add_argument(
        "--manual-stub-root",
        default=None,
        help="optional directory for generated manual Triton backend scaffolds",
    )
    parser.add_argument(
        "--materialize-manual-models",
        action="store_true",
        help="explicitly install manual planned models into the Triton repository instead of generating metadata only",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    selected_model_ids = list(dict.fromkeys([*get_profile_model_ids(args.profile), *args.model_id]))
    output_root = Path(args.output_root).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    manual_stub_root = Path(args.manual_stub_root).resolve() if args.manual_stub_root else None

    manifest = prepare_model_repository(
        output_root=output_root,
        model_ids=selected_model_ids,
        cache_dir=cache_dir,
        manual_stub_root=manual_stub_root,
        materialize_manual_models=args.materialize_manual_models,
    )
    manifest["profile"] = args.profile
    manifest["selected_model_ids"] = selected_model_ids
    manifest["materialize_manual_models"] = args.materialize_manual_models

    manifest_path = write_manifest(output_root, manifest)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "selected_model_ids": selected_model_ids,
                "manual_stub_root": str(manual_stub_root) if manual_stub_root is not None else None,
                "materialize_manual_models": args.materialize_manual_models,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
