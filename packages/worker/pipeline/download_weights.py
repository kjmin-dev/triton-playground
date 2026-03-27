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
import os
import time
from fnmatch import fnmatch
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from pipeline.model_catalog import (
    HOLD_LANE,
    MODEL_CATALOG,
    ModelSpec,
    get_model_spec,
    get_profile_model_ids,
)

_console = Console(stderr=True)


def _downloadable_specs() -> list[ModelSpec]:
    return [
        spec
        for spec in MODEL_CATALOG.values()
        if spec.hf_repo_id is not None
        and spec.revision is not None
        and spec.serve_status != HOLD_LANE
        and spec.snapshot_allow_patterns
    ]


def _resolve_expected_files(
    spec: ModelSpec,
    cache_dir: Path,
) -> tuple[list[str], int]:
    """Return (filenames, total_bytes) for the files we expect to download."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(
        repo_id=spec.hf_repo_id,
        revision=spec.revision,
        files_metadata=True,
    )
    patterns = list(spec.snapshot_allow_patterns)
    files: list[str] = []
    total_bytes = 0
    for sibling in info.siblings:
        if not patterns or any(fnmatch(sibling.rfilename, p) for p in patterns):
            files.append(sibling.rfilename)
            total_bytes += sibling.size or 0
    return files, total_bytes


def _download_files(
    spec: ModelSpec,
    cache_dir: Path,
    files: list[str],
    total_bytes: int,
    counter: str,
) -> Path:
    """Download files one-by-one with a live rich progress bar."""
    from huggingface_hub import hf_hub_download

    assert spec.hf_repo_id is not None
    assert spec.revision is not None

    snapshot_path: Path | None = None
    downloaded_bytes = 0

    with Progress(
        SpinnerColumn(),
        TextColumn(f"{counter} [bold]{spec.model_id}[/bold]"),
        BarColumn(bar_width=30),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(compact=True),
        console=_console,
        transient=True,
    ) as progress:
        task = progress.add_task("download", total=total_bytes)

        for filename in files:
            t0 = time.monotonic()
            path = hf_hub_download(
                repo_id=spec.hf_repo_id,
                filename=filename,
                revision=spec.revision,
                cache_dir=str(cache_dir),
            )
            elapsed = time.monotonic() - t0

            file_path = Path(path)
            if snapshot_path is None:
                # Derive snapshot root from first downloaded file
                # e.g. .cache/huggingface/hub/models--org--name/snapshots/rev/file
                parts = file_path.parts
                snap_idx = parts.index("snapshots") if "snapshots" in parts else None
                if snap_idx is not None and snap_idx + 2 < len(parts):
                    snapshot_path = Path(*parts[: snap_idx + 2])

            file_size = file_path.stat().st_size if file_path.exists() else 0
            downloaded_bytes += file_size
            cached = elapsed < 0.1 and file_size > 0
            progress.update(task, completed=downloaded_bytes)

            if not cached:
                progress.console.print(
                    f"       [dim]{filename}[/dim]",
                    highlight=False,
                )

    return snapshot_path or Path(path).parent


def _download_snapshot_simple(spec: ModelSpec, cache_dir: Path) -> Path:
    """Fallback: use snapshot_download directly (no per-file progress)."""
    from huggingface_hub import snapshot_download

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
    # Suppress huggingface_hub's internal tqdm — we show our own progress
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    if model_ids:
        specs = [get_model_spec(mid) for mid in model_ids]
    else:
        specs = _downloadable_specs()

    resolved_cache = cache_dir or Path(".cache/huggingface")
    resolved_cache.mkdir(parents=True, exist_ok=True)

    actionable = [
        s for s in specs
        if s.snapshot_allow_patterns and s.hf_repo_id and s.revision
    ]
    skipped = [s for s in specs if s not in actionable]

    total = len(actionable)
    _console.print()
    _console.print(f"  [bold]Downloading model weights[/bold]  [dim]({total} model{'s' if total != 1 else ''})[/dim]")
    _console.print()

    for spec in skipped:
        reason = "no snapshot patterns" if not spec.snapshot_allow_patterns else "no upstream pinned"
        _console.print(f"  [yellow]–[/yellow] [dim]{spec.model_id}: {reason}[/dim]")

    results: list[dict[str, str]] = []
    for i, spec in enumerate(actionable, 1):
        counter = f"[dim]\\[{i}/{total}][/dim]"
        repo_label = f"[dim]{spec.hf_repo_id}@{spec.revision[:8]}[/dim]"

        try:
            files, total_bytes = _resolve_expected_files(spec, resolved_cache)
        except Exception:
            files, total_bytes = [], 0

        try:
            if files and total_bytes > 0:
                snapshot_path = _download_files(spec, resolved_cache, files, total_bytes, counter)
            else:
                _console.print(f"  {counter} [bold]{spec.model_id}[/bold]  {repo_label}")
                with _console.status("       resolving..."):
                    snapshot_path = _download_snapshot_simple(spec, resolved_cache)
        except Exception as exc:
            _console.print(f"  {counter} [red]✗[/red] [bold]{spec.model_id}[/bold]  [red]{exc}[/red]")
            continue

        _console.print(f"  {counter} [green]✓[/green] [bold]{spec.model_id}[/bold]  {repo_label}")

        results.append({
            "model_id": spec.model_id,
            "hf_repo_id": spec.hf_repo_id or "",
            "revision": spec.revision or "",
            "snapshot_path": str(snapshot_path),
        })

    _console.print()
    ok = len(results)
    if ok == total:
        _console.print(f"  [bold green]✓ {ok} model{'s' if ok != 1 else ''} ready[/bold green]")
    elif ok:
        _console.print(f"  [yellow]✓ {ok}/{total} models ready[/yellow]")
    else:
        _console.print(f"  [red]✗ no models downloaded[/red]")
    _console.print()

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
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Model", style="bold")
        table.add_column("Repository", style="dim")
        table.add_column("Stage")
        for spec in _downloadable_specs():
            rev = spec.revision[:8] if spec.revision else "?"
            table.add_row(spec.model_id, f"{spec.hf_repo_id}@{rev}", spec.stage)
        _console.print()
        _console.print(table)
        _console.print()
        return

    model_ids: list[str] | None = None
    if args.model_id:
        model_ids = args.model_id
    elif args.profile:
        model_ids = list(get_profile_model_ids(args.profile))

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    results = download_weights(model_ids=model_ids, cache_dir=cache_dir)

    if results:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
