"""Download model weights for Triton serving without materializing the full repository.

Usage:
    python -m pipeline.download_weights                          # download all downloadable models
    python -m pipeline.download_weights --model-id whisper_large_v3_turbo
    python -m pipeline.download_weights --profile localize
    python -m pipeline.download_weights --list                   # show downloadable models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.model_catalog import (
    HOLD_LANE,
    MODEL_CATALOG,
    ModelSpec,
    get_model_spec,
    get_profile_model_ids,
)


def _downloadable_specs() -> list[ModelSpec]:
    return [
        spec
        for spec in MODEL_CATALOG.values()
        if spec.hf_repo_id is not None
        and spec.revision is not None
        and spec.serve_status != HOLD_LANE
        and spec.snapshot_allow_patterns
    ]


def _download_snapshot(spec: ModelSpec, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required. Run: uv pip install huggingface-hub"
        ) from exc

    assert spec.hf_repo_id is not None
    assert spec.revision is not None

    return Path(
        snapshot_download(
            repo_id=spec.hf_repo_id,
            revision=spec.revision,
            allow_patterns=list(spec.snapshot_allow_patterns) or None,
            cache_dir=str(cache_dir),
        )
    )


def download_weights(
    model_ids: list[str] | None = None,
    cache_dir: Path | None = None,
) -> list[dict[str, str]]:
    if model_ids:
        specs = [get_model_spec(mid) for mid in model_ids]
    else:
        specs = _downloadable_specs()

    resolved_cache = cache_dir or Path(".cache/huggingface")
    resolved_cache.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, str]] = []
    for spec in specs:
        if not spec.snapshot_allow_patterns:
            print(f"  skip {spec.model_id}: no snapshot patterns defined", file=sys.stderr)
            continue
        if spec.hf_repo_id is None or spec.revision is None:
            print(f"  skip {spec.model_id}: no upstream pinned", file=sys.stderr)
            continue

        print(f"  downloading {spec.model_id} ({spec.hf_repo_id}@{spec.revision[:8]})...", file=sys.stderr)
        try:
            snapshot_path = _download_snapshot(spec, resolved_cache)
        except Exception as exc:
            print(f"  FAILED {spec.model_id}: {exc}", file=sys.stderr)
            continue
        print(f"  done -> {snapshot_path}", file=sys.stderr)

        results.append({
            "model_id": spec.model_id,
            "hf_repo_id": spec.hf_repo_id,
            "revision": spec.revision,
            "snapshot_path": str(snapshot_path),
        })

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model weights for Triton serving.")
    parser.add_argument("--model-id", action="append", default=[], help="specific model id(s) to download")
    parser.add_argument("--profile", default=None, help="download all models in a profile (e.g. localize)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--list", action="store_true", dest="list_only", help="list downloadable models and exit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_only:
        for spec in _downloadable_specs():
            print(f"  {spec.model_id:30s} {spec.hf_repo_id}@{spec.revision[:8] if spec.revision else '?'}")
        return

    model_ids: list[str] | None = None
    if args.model_id:
        model_ids = args.model_id
    elif args.profile:
        model_ids = list(get_profile_model_ids(args.profile))

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    results = download_weights(model_ids=model_ids, cache_dir=cache_dir)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
