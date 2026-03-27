from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
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

DownloadFn = Callable[[ModelSpec, ModelArtifact, Path | None], Path]
MANIFEST_SCHEMA_VERSION = 2


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


def prepare_model_repository(
    output_root: Path,
    model_ids: list[str],
    cache_dir: Path | None = None,
    downloader: DownloadFn = _default_downloader,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "baseline_profile": "baseline",
            "planning_profiles": ["stt", "catalog"],
            "allowed_serve_statuses": [AUTO_DOWNLOAD_LANE, MANUAL_PLANNED_LANE, HOLD_LANE],
        },
        "models": [],
    }

    for model_id in model_ids:
        spec = get_model_spec(model_id)
        record: dict[str, object] = spec.to_dict()

        if not spec.approved_for_auto_download:
            record["installed"] = False
            record["reason"] = _skip_reason(spec)
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    selected_model_ids = list(dict.fromkeys([*get_profile_model_ids(args.profile), *args.model_id]))
    output_root = Path(args.output_root).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    manifest = prepare_model_repository(output_root=output_root, model_ids=selected_model_ids, cache_dir=cache_dir)
    manifest["profile"] = args.profile
    manifest["selected_model_ids"] = selected_model_ids

    manifest_path = write_manifest(output_root, manifest)
    print(json.dumps({"manifest_path": str(manifest_path), "selected_model_ids": selected_model_ids}, indent=2))


if __name__ == "__main__":
    main()
