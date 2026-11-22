#!/usr/bin/env python3
"""Invoke the Go benchtool for performance and quality regression testing.

The Python layer is intentionally thin: it chooses a profile, builds the Go tool when needed,
merges JSON configs for combined runs, invokes benchtool with reproducible paths and prints a
short summary. All timing, exit-code and baseline logic lives in `cmd/benchtool`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CONFIG_DIR = Path("config/benchtool")
ARTIFACT_DIR = Path("artifacts/bench")
BENCHTOOL_PATH = Path("target/benchtool")

PROFILES: dict[str, tuple[str, ...]] = {
    "quality": ("quality.json",),
    "performance": ("performance.json",),
    "regression": ("quality.json", "performance.json"),
    "all": ("quality.json", "performance.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="regression",
        help="check profile to run (default: regression)",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--benchtool", type=Path, default=BENCHTOOL_PATH, help="path to the built benchtool binary"
    )
    parser.add_argument("--rebuild", action="store_true", help="always rebuild benchtool")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACT_DIR, help="artifact directory")
    parser.add_argument("--baseline", type=Path, help="baseline JSON to compare against")
    parser.add_argument(
        "--update-baseline",
        type=Path,
        nargs="?",
        const=ARTIFACT_DIR / "baseline.json",
        help="copy the fresh results to this baseline path",
    )
    parser.add_argument("--iterations", type=int, help="override iterations for performance checks")
    parser.add_argument("--warmup", type=int, help="override warm-up runs")
    parser.add_argument("--tolerance", type=float, help="override the regression tolerance")
    parser.add_argument("--filter", help="only run checks whose name contains this string")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failure")
    parser.add_argument("--timeout", type=int, help="override per-command timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="print the benchtool command and exit")
    return parser.parse_args()


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        env["PATH"] = f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def build_benchtool(root: Path, binary: Path, env: dict[str, str], force: bool) -> None:
    if binary.exists() and not force:
        return
    binary.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> building {binary}")
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/benchtool"],
        cwd=root,
        env=env,
        check=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def merged_config(root: Path, names: tuple[str, ...], profile: str) -> dict[str, Any]:
    configs = [load_json(root / CONFIG_DIR / name) for name in names]
    merged: dict[str, Any] = {
        "name": profile,
        "description": f"Combined quality and performance profile: {profile}",
        "warmup": max(int(config.get("warmup", 0)) for config in configs),
        "iterations": max(int(config.get("iterations", 1)) for config in configs),
        "timeout_seconds": max(int(config.get("timeout_seconds", 600)) for config in configs),
        "tolerance": max(float(config.get("tolerance", 0)) for config in configs),
        "commands": [],
    }
    seen: set[str] = set()
    for config in configs:
        for check in config.get("commands", []):
            name = check.get("name") or check.get("command")
            if name in seen:
                continue
            seen.add(name)
            merged["commands"].append(check)
    return merged


def resolve_config(root: Path, profile: str) -> tuple[Path, tempfile.NamedTemporaryFile | None]:
    names = PROFILES[profile]
    if len(names) == 1:
        return root / CONFIG_DIR / names[0], None
    config = merged_config(root, names, profile)
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"quarrel-{profile}-", delete=False, encoding="utf-8"
    )
    json.dump(config, handle, indent=2)
    handle.write("\n")
    handle.close()
    return Path(handle.name), handle


def build_command(args: argparse.Namespace, root: Path, config: Path, out: Path, report: Path) -> list[str]:
    cmd = [
        str(root / args.benchtool) if not args.benchtool.is_absolute() else str(args.benchtool),
        "-config", str(config),
        "-root", str(root),
        "-out", str(out),
        "-markdown", str(report),
    ]
    baseline = args.baseline
    if baseline is None:
        default_baseline = root / ARTIFACT_DIR / "baseline.json"
        if default_baseline.exists() and args.profile in {"performance", "regression", "all"}:
            baseline = default_baseline
    if baseline is not None:
        cmd += ["-baseline", str(baseline if baseline.is_absolute() else root / baseline)]
    if args.iterations is not None:
        cmd += ["-iterations", str(args.iterations)]
    if args.warmup is not None:
        cmd += ["-warmup", str(args.warmup)]
    if args.tolerance is not None:
        cmd += ["-tolerance", str(args.tolerance)]
    if args.filter:
        cmd += ["-filter", args.filter]
    if args.fail_fast:
        cmd += ["-fail-fast"]
    if args.timeout is not None:
        cmd += ["-timeout", str(args.timeout)]
    return cmd


def print_summary(results_path: Path) -> None:
    if not results_path.exists():
        return
    payload = load_json(results_path)
    summary = payload.get("summary", {})
    print(
        "\nSummary: {passed}/{total} passed, {failed} failed, {regressed} regressed, "
        "{duration_ms:.1f} ms".format(
            passed=summary.get("passed", 0),
            total=summary.get("total", 0),
            failed=summary.get("failed", 0),
            regressed=summary.get("regressed", 0),
            duration_ms=summary.get("duration_ms", 0),
        )
    )
    for check in payload.get("checks", []):
        if check.get("passed"):
            continue
        print(f"  FAILED {check['name']}: {check.get('failure', 'unknown failure')}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not (root / "go.mod").exists():
        print(f"error: {root} does not look like the quarrel-rs repository root", file=sys.stderr)
        return 2

    env = command_env()
    if not args.dry_run:
        build_benchtool(root, args.benchtool, env, args.rebuild)

    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    report_path = out_dir / "results.md"

    config_path, temp_handle = resolve_config(root, args.profile)
    try:
        cmd = build_command(args, root, config_path, results_path, report_path)
        printable = " ".join(str(part) for part in cmd)
        print(f"==> {printable}")
        if args.dry_run:
            return 0

        completed = subprocess.run(cmd, cwd=root, env=env, check=False)
        print_summary(results_path)

        if completed.returncode == 0 and args.update_baseline is not None:
            baseline = args.update_baseline
            baseline = baseline if baseline.is_absolute() else root / baseline
            baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(results_path, baseline)
            print(f"updated baseline: {baseline}")

        return completed.returncode
    finally:
        if temp_handle is not None:
            Path(temp_handle.name).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
