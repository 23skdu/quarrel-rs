package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStats(t *testing.T) {
	t.Parallel()

	mean, min, max, stddev := stats([]float64{1, 2, 3, 4})
	if mean != 2.5 || min != 1 || max != 4 {
		t.Fatalf("unexpected bounds: mean=%v min=%v max=%v", mean, min, max)
	}
	if math.Abs(stddev-1.118033988749895) > 1e-9 {
		t.Fatalf("unexpected stddev: %v", stddev)
	}

	mean, min, max, stddev = stats(nil)
	if mean != 0 || min != 0 || max != 0 || stddev != 0 {
		t.Fatalf("empty stats should be zero: %v %v %v %v", mean, min, max, stddev)
	}
}

func TestSum(t *testing.T) {
	t.Parallel()

	if got := sum([]float64{1.5, 2, 3.25}); got != 6.75 {
		t.Fatalf("sum = %v, want 6.75", got)
	}
	if got := sum(nil); got != 0 {
		t.Fatalf("sum(nil) = %v, want 0", got)
	}
}

func TestTail(t *testing.T) {
	t.Parallel()

	if got := tail("short output"); got != "short output" {
		t.Fatalf("short tail = %q", got)
	}
	long := strings.Repeat("x", outputTailLimit+10)
	got := tail(long)
	if !strings.HasPrefix(got, "...[truncated]...\n") {
		t.Fatalf("tail prefix missing: %q", got[:32])
	}
	if !strings.HasSuffix(got, strings.Repeat("x", outputTailLimit)) {
		t.Fatal("tail should retain the last outputTailLimit bytes")
	}
}

func TestLoadConfig(t *testing.T) {
	t.Parallel()

	path := writeConfig(t, config{
		Iterations:     2,
		TimeoutSeconds: 30,
		Commands:       []command{{Name: "ok", Command: "sh", Args: []string{"-c", "true"}}},
	})
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.Name != filepath.Base(path) {
		t.Fatalf("default name = %q, want %q", cfg.Name, filepath.Base(path))
	}
	if cfg.Iterations != 2 || len(cfg.Commands) != 1 {
		t.Fatalf("unexpected config: %+v", cfg)
	}

	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadConfig(bad); err == nil {
		t.Fatal("malformed JSON should fail")
	}
	if _, err := loadConfig(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("missing config should fail")
	}
}

func TestLoadBaselines(t *testing.T) {
	t.Parallel()

	baselines, err := loadBaselines("")
	if err != nil || len(baselines) != 0 {
		t.Fatalf("empty baseline path: %v %v", baselines, err)
	}

	path := filepath.Join(t.TempDir(), "baseline.json")
	writeResult(t, path, result{Checks: []checkResult{
		{Name: "one", MeanMS: 10},
		{Name: "two", MeanMS: 20},
	}})
	baselines, err = loadBaselines(path)
	if err != nil || len(baselines) != 2 || baselines["two"].MeanMS != 20 {
		t.Fatalf("baseline load: %v %v", baselines, err)
	}

	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte("[]"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadBaselines(bad); err == nil {
		t.Fatal("wrong baseline JSON shape should fail")
	}
	if _, err := loadBaselines(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("missing baseline should fail")
	}
}

func TestExecuteQualityChecks(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	cfg := config{Iterations: 2, TimeoutSeconds: 5}
	success := execute(cfg, command{
		Name:    "success",
		Command: "sh",
		Args:    []string{"-c", "echo hello"},
	}, root, checkResult{})
	// Quality checks are gates, so they always run once regardless of the configured iteration count.
	if !success.Passed || success.ExitCode != 0 || success.Iterations != 1 {
		t.Fatalf("unexpected success result: %+v", success)
	}
	if success.MeanMS <= 0 || success.MinMS > success.MaxMS {
		t.Fatalf("unexpected timings: %+v", success)
	}
	if !strings.Contains(success.OutputTail, "hello") {
		t.Fatalf("output tail missing: %q", success.OutputTail)
	}

	expectedFailure := execute(cfg, command{
		Name:       "expected failure",
		Command:    "sh",
		Args:       []string{"-c", "exit 4"},
		ExpectExit: 4,
	}, root, checkResult{})
	if !expectedFailure.Passed || expectedFailure.ExitCode != 4 {
		t.Fatalf("expected exit code should pass: %+v", expectedFailure)
	}

	failure := execute(cfg, command{
		Name:    "failure",
		Command: "sh",
		Args:    []string{"-c", "exit 3"},
	}, root, checkResult{})
	if failure.Passed || failure.ExitCode != 3 || !strings.Contains(failure.Failure, "expected 0") {
		t.Fatalf("unexpected failure result: %+v", failure)
	}
}

func TestExecutePerformanceBaseline(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	cfg := config{Iterations: 1, TimeoutSeconds: 5, Tolerance: 0.1}
	check := command{
		Name:    "timed",
		Kind:    "performance",
		Command: "sh",
		Args:    []string{"-c", "sleep 0.05"},
	}

	noBaseline := execute(cfg, check, root, checkResult{})
	if !noBaseline.Passed || noBaseline.BaselineMS != 0 || noBaseline.Delta != 0 {
		t.Fatalf("missing baseline should not regress: %+v", noBaseline)
	}

	good := execute(cfg, check, root, checkResult{Name: "timed", MeanMS: 1000})
	if !good.Passed || good.BaselineMS != 1000 || good.Delta >= 0 {
		t.Fatalf("improvement should pass: %+v", good)
	}

	regressed := execute(cfg, check, root, checkResult{Name: "timed", MeanMS: 0.001})
	if regressed.Passed || !strings.Contains(regressed.Failure, "tolerance") {
		t.Fatalf("regression should fail: %+v", regressed)
	}
}

func TestExecuteWorkDirOverride(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	sub := filepath.Join(root, "nested")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	check := command{
		Name:    "pwd",
		Command: "sh",
		Args:    []string{"-c", "pwd"},
		WorkDir: "nested",
	}
	check.TimeoutSeconds = 5
	got := execute(config{Iterations: 1}, check, root, checkResult{})
	if !got.Passed || !strings.Contains(got.OutputTail, sub) {
		t.Fatalf("workdir override failed: %+v", got)
	}
}

func TestRunOnceTimeoutAndMissingCommand(t *testing.T) {
	t.Parallel()

	_, code, output := runOnce("sh", []string{"-c", "sleep 2"}, t.TempDir(), 1)
	if code != -1 || !strings.Contains(output, "benchmark timed out") {
		t.Fatalf("timeout not reported: code=%d output=%q", code, output)
	}

	_, code, output = runOnce("quarrel-benchtool-no-such-command", nil, t.TempDir(), 5)
	if code != -1 || strings.TrimSpace(output) == "" {
		t.Fatalf("missing command not reported: code=%d output=%q", code, output)
	}
}

func TestSummarise(t *testing.T) {
	t.Parallel()

	checks := []checkResult{
		{Name: "pass", Passed: true},
		{Name: "quality-fail"},
		{Name: "perf-fail", Kind: "performance", BaselineMS: 10},
	}
	got := summarise(checks, 1500*time.Millisecond)
	if got.Total != 3 || got.Passed != 1 || got.Failed != 2 || got.Regressed != 1 {
		t.Fatalf("unexpected summary: %+v", got)
	}
	if got.DurationMS != 1500 {
		t.Fatalf("duration = %v, want 1500", got.DurationMS)
	}
}

func TestRunEndToEnd(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	cfgPath := writeConfig(t, config{
		Iterations: 1,
		Commands: []command{
			{Name: "quality-check", Command: "sh", Args: []string{"-c", "echo quality-ok"}},
			{Name: "performance-check", Kind: "performance", Command: "sh", Args: []string{"-c", "echo perf-ok"}},
		},
	})
	out := filepath.Join(dir, "results.json")
	report := filepath.Join(dir, "report.md")

	if err := run([]string{
		"-config", cfgPath,
		"-root", dir,
		"-out", out,
		"-markdown", report,
		"-warmup", "0",
		"-iterations", "1",
	}); err != nil {
		t.Fatalf("run: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var got result
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode results: %v", err)
	}
	if got.Tool != "quarrel-benchtool" || got.Summary.Total != 2 || got.Summary.Passed != 2 {
		t.Fatalf("unexpected result: %+v", got)
	}
	if len(got.Checks) != 2 || got.Checks[0].Name != "quality-check" {
		t.Fatalf("unexpected checks: %+v", got.Checks)
	}

	markdown, err := os.ReadFile(report)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"quality-check", "performance-check", "PASS"} {
		if !strings.Contains(string(markdown), name) {
			t.Fatalf("report missing %q:\n%s", name, markdown)
		}
	}
}

func TestRunFilterAndFailure(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	cfgPath := writeConfig(t, config{
		Iterations: 1,
		Commands: []command{
			{Name: "skip-me", Command: "sh", Args: []string{"-c", "true"}},
			{Name: "fail-me", Command: "sh", Args: []string{"-c", "exit 9"}},
		},
	})
	out := filepath.Join(dir, "results.json")
	report := filepath.Join(dir, "report.md")
	err := run([]string{
		"-config", cfgPath,
		"-root", dir,
		"-out", out,
		"-markdown", report,
		"-warmup", "0",
		"-filter", "fail",
	})
	if err == nil || !strings.Contains(err.Error(), "check(s) failed") {
		t.Fatalf("expected failure error, got %v", err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var got result
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Summary.Total != 1 || got.Checks[0].Name != "fail-me" {
		t.Fatalf("filter was not applied: %+v", got)
	}
}

func TestRunRejectsEmptyConfigAndBadFlags(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	empty := writeConfig(t, config{Iterations: 1})
	out := filepath.Join(dir, "results.json")
	report := filepath.Join(dir, "report.md")
	if err := run([]string{"-config", empty, "-out", out, "-markdown", report}); err == nil {
		t.Fatal("empty command list should fail")
	}
	if err := run([]string{"-not-a-flag"}); err == nil {
		t.Fatal("invalid flag should fail")
	}
}

func writeConfig(t *testing.T, cfg config) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "config.json")
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeResult(t *testing.T, path string, value result) {
	t.Helper()

	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}
