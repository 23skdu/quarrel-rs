"""Unit tests for scripts/bench.py.

Run with ``python3 -m unittest discover -s scripts -p 'test_*.py' -v`` from the repository root.
The tests use only the standard library and never execute the Go benchtool.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = Path(__file__).resolve().parent / "bench.py"

spec = importlib.util.spec_from_file_location("quarrel_bench_driver", MODULE_PATH)
if spec is None or spec.loader is None:  # pragma: no cover - defensive
    raise RuntimeError(f"cannot load {MODULE_PATH}")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchDriverTests(unittest.TestCase):
    def test_profiles_are_defined(self) -> None:
        self.assertEqual(bench.PROFILES["quality"], ("quality.json",))
        self.assertEqual(bench.PROFILES["performance"], ("performance.json",))
        self.assertEqual(bench.PROFILES["regression"], ("quality.json", "performance.json"))
        self.assertEqual(bench.PROFILES["all"], bench.PROFILES["regression"])

    def test_merged_config_deduplicates_checks(self) -> None:
        config = bench.merged_config(REPO_ROOT, bench.PROFILES["regression"], "regression")
        self.assertEqual(config["profile"] if "profile" in config else config["name"], "regression")
        names = [check["name"] for check in config["commands"]]
        self.assertIn("unit-tests", names)
        self.assertIn("gguf-quant-kernels", names)
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(config["iterations"], 3)
        self.assertGreaterEqual(config["timeout_seconds"], 900)
        self.assertAlmostEqual(config["tolerance"], 0.1)

    def test_resolve_config_single_profile(self) -> None:
        path, handle = bench.resolve_config(REPO_ROOT, "quality")
        self.assertIsNone(handle)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "quality.json")

    def test_resolve_config_combined_profile(self) -> None:
        path, handle = bench.resolve_config(REPO_ROOT, "regression")
        self.assertIsNotNone(handle)
        try:
            payload = bench.load_json(path)
            self.assertEqual(payload["name"], "regression")
            self.assertTrue(payload["commands"])
        finally:
            path.unlink(missing_ok=True)

    def test_build_command_includes_overrides(self) -> None:
        args = Namespace(
            benchtool=Path("target/benchtool"),
            baseline=Path("artifacts/bench/baseline.json"),
            iterations=5,
            warmup=2,
            tolerance=0.2,
            filter="sampler",
            fail_fast=True,
            timeout=120,
            profile="performance",
        )
        cmd = bench.build_command(
            args,
            REPO_ROOT,
            REPO_ROOT / "config/benchtool/performance.json",
            REPO_ROOT / "artifacts/bench/results.json",
            REPO_ROOT / "artifacts/bench/results.md",
        )
        self.assertEqual(cmd[0], str(REPO_ROOT / "target/benchtool"))
        for flag, value in [
            ("-baseline", str(REPO_ROOT / "artifacts/bench/baseline.json")),
            ("-iterations", "5"),
            ("-warmup", "2"),
            ("-tolerance", "0.2"),
            ("-filter", "sampler"),
            ("-timeout", "120"),
        ]:
            self.assertIn(flag, cmd)
            self.assertIn(value, cmd)
        self.assertIn("-fail-fast", cmd)

    def test_build_command_uses_default_baseline_for_performance(self) -> None:
        args = Namespace(
            benchtool=Path("target/benchtool"),
            baseline=None,
            iterations=None,
            warmup=None,
            tolerance=None,
            filter="",
            fail_fast=False,
            timeout=None,
            profile="performance",
        )
        cmd = bench.build_command(
            args,
            REPO_ROOT,
            REPO_ROOT / "config/benchtool/performance.json",
            REPO_ROOT / "artifacts/bench/results.json",
            REPO_ROOT / "artifacts/bench/results.md",
        )
        if (REPO_ROOT / "artifacts/bench/baseline.json").exists():
            self.assertIn("-baseline", cmd)
        else:
            self.assertNotIn("-baseline", cmd)

    def test_build_benchtool_skips_existing_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "benchtool"
            binary.write_text("existing", encoding="utf-8")
            with mock.patch.object(bench.subprocess, "run") as run:
                bench.build_benchtool(REPO_ROOT, binary, os.environ.copy(), force=False)
            run.assert_not_called()
            self.assertEqual(binary.read_text(encoding="utf-8"), "existing")

    def test_command_env_prepends_cargo_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cargo_bin = Path(tmp) / ".cargo" / "bin"
            cargo_bin.mkdir(parents=True)
            with mock.patch("pathlib.Path.home", return_value=Path(tmp)):
                env = bench.command_env()
            self.assertTrue(env["PATH"].startswith(str(cargo_bin)))

    def test_print_summary_reports_failures(self) -> None:
        payload = {
            "summary": {"passed": 1, "total": 2, "failed": 1, "regressed": 0, "duration_ms": 12.5},
            "checks": [
                {"name": "ok", "passed": True},
                {"name": "bad", "passed": False, "failure": "exit code 2"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                bench.print_summary(path)
        output = stream.getvalue()
        self.assertIn("1/2 passed", output)
        self.assertIn("FAILED bad: exit code 2", output)

    def test_main_dry_run(self) -> None:
        argv = [
            "bench.py",
            "--profile",
            "quality",
            "--root",
            str(REPO_ROOT),
            "--dry-run",
        ]
        with mock.patch.object(sys, "argv", argv):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = bench.main()
        self.assertEqual(code, 0)
        self.assertIn("-config", stream.getvalue())

    def test_main_rejects_non_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["bench.py", "--root", tmp, "--dry-run"]
            with mock.patch.object(sys, "argv", argv):
                stream = io.StringIO()
                with contextlib.redirect_stderr(stream):
                    code = bench.main()
            self.assertEqual(code, 2)
            self.assertIn("does not look like", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
