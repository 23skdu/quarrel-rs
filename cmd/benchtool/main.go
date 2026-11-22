// Command benchtool runs reproducible performance and quality checks for quarrel-rs.
//
// It is deliberately dependency-free: the tool shells out to the repository's Makefile and cargo
// commands, records wall-clock timings and exit codes, compares them with an optional baseline and
// emits both JSON and Markdown artifacts for CI to archive.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const outputTailLimit = 4096

type config struct {
	Name           string    `json:"name"`
	Description    string    `json:"description"`
	Warmup         int       `json:"warmup"`
	Iterations     int       `json:"iterations"`
	TimeoutSeconds int       `json:"timeout_seconds"`
	Tolerance      float64   `json:"tolerance"`
	Commands       []command `json:"commands"`
}

type command struct {
	Name           string   `json:"name"`
	Kind           string   `json:"kind"`
	Command        string   `json:"command"`
	Args           []string `json:"args"`
	WorkDir        string   `json:"workdir"`
	ExpectExit     int      `json:"expect_exit"`
	TimeoutSeconds int      `json:"timeout_seconds"`
	Iterations     int      `json:"iterations"`
}

type checkResult struct {
	Name       string    `json:"name"`
	Kind       string    `json:"kind"`
	Command    string    `json:"command"`
	ExitCode   int       `json:"exit_code"`
	Iterations int       `json:"iterations"`
	DurationMS float64   `json:"duration_ms"`
	MeanMS     float64   `json:"mean_ms"`
	MinMS      float64   `json:"min_ms"`
	MaxMS      float64   `json:"max_ms"`
	StddevMS   float64   `json:"stddev_ms"`
	BaselineMS float64   `json:"baseline_ms,omitempty"`
	Delta      float64   `json:"delta,omitempty"`
	Passed     bool      `json:"passed"`
	Failure    string    `json:"failure,omitempty"`
	OutputTail string    `json:"output_tail,omitempty"`
	FinishedAt time.Time `json:"finished_at"`
}

type summary struct {
	Total      int     `json:"total"`
	Passed     int     `json:"passed"`
	Failed     int     `json:"failed"`
	Regressed  int     `json:"regressed"`
	DurationMS float64 `json:"duration_ms"`
}

type result struct {
	Tool       string        `json:"tool"`
	Config     string        `json:"config"`
	StartedAt  time.Time     `json:"started_at"`
	FinishedAt time.Time     `json:"finished_at"`
	Summary    summary       `json:"summary"`
	Checks     []checkResult `json:"checks"`
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "benchtool: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	root, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("resolve working directory: %w", err)
	}

	flags := flag.NewFlagSet("benchtool", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	configPath := flags.String("config", "config/benchtool/quality.json", "JSON check configuration")
	outPath := flags.String("out", "artifacts/bench/results.json", "JSON result path")
	markdownPath := flags.String("markdown", "artifacts/bench/results.md", "Markdown report path")
	baselinePath := flags.String("baseline", "", "baseline JSON result for regression comparison")
	repoRoot := flags.String("root", root, "repository root used as the default command working directory")
	tolerance := flags.Float64("tolerance", 0, "override the configured regression tolerance")
	iterations := flags.Int("iterations", 0, "override the configured iteration count")
	warmup := flags.Int("warmup", -1, "override the configured warm-up count")
	timeout := flags.Int("timeout", 0, "override the configured per-command timeout in seconds")
	filter := flags.String("filter", "", "only run checks whose name contains this substring")
	failFast := flags.Bool("fail-fast", false, "stop after the first failed check")
	if err := flags.Parse(args); err != nil {
		return fmt.Errorf("parse flags: %w", err)
	}

	cfg, err := loadConfig(*configPath)
	if err != nil {
		return err
	}
	if cfg.Iterations <= 0 {
		cfg.Iterations = 1
	}
	if cfg.TimeoutSeconds <= 0 {
		cfg.TimeoutSeconds = 600
	}
	if *tolerance > 0 {
		cfg.Tolerance = *tolerance
	}
	if *iterations > 0 {
		cfg.Iterations = *iterations
	}
	if *warmup >= 0 {
		cfg.Warmup = *warmup
	}
	if *timeout > 0 {
		cfg.TimeoutSeconds = *timeout
	}
	if len(cfg.Commands) == 0 {
		return fmt.Errorf("configuration %q defines no commands", *configPath)
	}

	baselines, err := loadBaselines(*baselinePath)
	if err != nil {
		return err
	}

	started := time.Now()
	res := result{
		Tool:      "quarrel-benchtool",
		Config:    *configPath,
		StartedAt: started,
		Checks:    make([]checkResult, 0, len(cfg.Commands)),
	}

	for _, check := range cfg.Commands {
		if *filter != "" && !strings.Contains(check.Name, *filter) {
			continue
		}
		current := execute(cfg, check, *repoRoot, baselines[check.Name])
		res.Checks = append(res.Checks, current)
		printLine(current)
		if !current.Passed && *failFast {
			break
		}
	}

	res.FinishedAt = time.Now()
	res.Summary = summarise(res.Checks, res.FinishedAt.Sub(started))

	if err := writeJSON(*outPath, res); err != nil {
		return err
	}
	if err := writeMarkdown(*markdownPath, res); err != nil {
		return err
	}

	fmt.Printf("\n%s: %d passed, %d failed, %d regressed, %.2fs total\n",
		res.Config, res.Summary.Passed, res.Summary.Failed, res.Summary.Regressed,
		res.Summary.DurationMS/1000)
	fmt.Printf("json: %s\nmarkdown: %s\n", *outPath, *markdownPath)

	if res.Summary.Failed > 0 || res.Summary.Regressed > 0 {
		return fmt.Errorf("%d check(s) failed", res.Summary.Failed+res.Summary.Regressed)
	}
	return nil
}

func loadConfig(path string) (config, error) {
	var cfg config
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read config %q: %w", path, err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config %q: %w", path, err)
	}
	if strings.TrimSpace(cfg.Name) == "" {
		cfg.Name = filepath.Base(path)
	}
	return cfg, nil
}

func loadBaselines(path string) (map[string]checkResult, error) {
	baselines := map[string]checkResult{}
	if path == "" {
		return baselines, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read baseline %q: %w", path, err)
	}
	var baseline result
	if err := json.Unmarshal(data, &baseline); err != nil {
		return nil, fmt.Errorf("parse baseline %q: %w", path, err)
	}
	for _, check := range baseline.Checks {
		baselines[check.Name] = check
	}
	return baselines, nil
}

func execute(cfg config, check command, root string, baseline checkResult) checkResult {
	out := checkResult{
		Name:       check.Name,
		Kind:       check.Kind,
		Command:    strings.TrimSpace(strings.Join(append([]string{check.Command}, check.Args...), " ")),
		FinishedAt: time.Now(),
	}
	if out.Name == "" {
		out.Name = out.Command
	}
	if out.Kind == "" {
		out.Kind = "quality"
	}

	iterations := check.Iterations
	if iterations <= 0 {
		iterations = cfg.Iterations
	}
	if strings.EqualFold(out.Kind, "quality") {
		iterations = 1
	}
	if iterations <= 0 {
		iterations = 1
	}

	timeout := check.TimeoutSeconds
	if timeout <= 0 {
		timeout = cfg.TimeoutSeconds
	}
	workDir := check.WorkDir
	if workDir == "" {
		workDir = root
	} else if !filepath.IsAbs(workDir) {
		workDir = filepath.Join(root, workDir)
	}

	durations := make([]float64, 0, iterations)
	var tail string
	var lastExit int
	for i := 0; i < iterations+cfg.Warmup; i++ {
		duration, exitCode, output := runOnce(check.Command, check.Args, workDir, timeout)
		tail = output
		lastExit = exitCode
		if i >= cfg.Warmup {
			durations = append(durations, duration)
		}
		if exitCode != check.ExpectExit {
			break
		}
	}

	out.Iterations = len(durations)
	out.ExitCode = lastExit
	out.OutputTail = tail
	out.DurationMS = sum(durations)
	out.MeanMS, out.MinMS, out.MaxMS, out.StddevMS = stats(durations)
	out.Passed = lastExit == check.ExpectExit
	if !out.Passed {
		out.Failure = fmt.Sprintf("exit code %d, expected %d", lastExit, check.ExpectExit)
		return out
	}

	if strings.EqualFold(out.Kind, "performance") && baseline.Name != "" && baseline.MeanMS > 0 {
		out.BaselineMS = baseline.MeanMS
		out.Delta = (out.MeanMS - baseline.MeanMS) / baseline.MeanMS
		if cfg.Tolerance > 0 && out.Delta > cfg.Tolerance {
			out.Passed = false
			out.Failure = fmt.Sprintf("regression %.1f%% exceeds %.1f%% tolerance",
				out.Delta*100, cfg.Tolerance*100)
		}
	}
	return out
}

func runOnce(name string, args []string, workDir string, timeout int) (float64, int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = workDir
	cmd.Env = os.Environ()
	var combined bytes.Buffer
	cmd.Stdout = &combined
	cmd.Stderr = &combined

	started := time.Now()
	err := cmd.Run()
	duration := float64(time.Since(started).Microseconds()) / 1000
	if ctx.Err() == context.DeadlineExceeded {
		return duration, -1, tail(combined.String() + "\nbenchmark timed out")
	}
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return duration, exitErr.ExitCode(), tail(combined.String())
		}
		return duration, -1, tail(combined.String() + "\n" + err.Error())
	}
	return duration, 0, tail(combined.String())
}

func tail(value string) string {
	if len(value) <= outputTailLimit {
		return value
	}
	return "...[truncated]...\n" + value[len(value)-outputTailLimit:]
}

func sum(values []float64) float64 {
	var total float64
	for _, value := range values {
		total += value
	}
	return total
}

func stats(values []float64) (mean, min, max, stddev float64) {
	if len(values) == 0 {
		return 0, 0, 0, 0
	}
	min, max = values[0], values[0]
	for _, value := range values {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
		mean += value
	}
	mean /= float64(len(values))
	for _, value := range values {
		stddev += (value - mean) * (value - mean)
	}
	stddev = math.Sqrt(stddev / float64(len(values)))
	return mean, min, max, stddev
}

func summarise(checks []checkResult, elapsed time.Duration) summary {
	out := summary{Total: len(checks), DurationMS: float64(elapsed.Microseconds()) / 1000}
	for _, check := range checks {
		if check.Passed {
			out.Passed++
			continue
		}
		out.Failed++
		if check.Kind == "performance" && check.BaselineMS > 0 {
			out.Regressed++
		}
	}
	return out
}

func printLine(check checkResult) {
	status := "PASS"
	if !check.Passed {
		status = "FAIL"
	}
	fmt.Printf("[%-4s] %-28s %-12s mean=%8.1fms delta=%+6.1f%%\n",
		status, check.Name, check.Kind, check.MeanMS, check.Delta*100)
	if check.Failure != "" {
		fmt.Printf("       %s\n", check.Failure)
	}
}

func writeJSON(path string, value result) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode result: %w", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write result %q: %w", path, err)
	}
	return nil
}

func writeMarkdown(path string, value result) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# %s benchmark report\n\n", value.Config)
	fmt.Fprintf(&b, "- Started: %s\n- Finished: %s\n- Passed: %d/%d\n- Regressions: %d\n\n",
		value.StartedAt.Format(time.RFC3339), value.FinishedAt.Format(time.RFC3339),
		value.Summary.Passed, value.Summary.Total, value.Summary.Regressed)
	b.WriteString("| Check | Kind | Mean (ms) | Min (ms) | Delta | Result |\n")
	b.WriteString("|-------|------|-----------|----------|-------|--------|\n")
	checks := append([]checkResult(nil), value.Checks...)
	sort.Slice(checks, func(i, j int) bool { return checks[i].Name < checks[j].Name })
	for _, check := range checks {
		status := "PASS"
		if !check.Passed {
			status = "FAIL"
		}
		fmt.Fprintf(&b, "| %s | %s | %.1f | %.1f | %+.1f%% | %s |\n",
			check.Name, check.Kind, check.MeanMS, check.MinMS, check.Delta*100, status)
		if check.Failure != "" {
			fmt.Fprintf(&b, "|\n| _%s_: %s | | | | | |\n", check.Name, check.Failure)
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create report directory: %w", err)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		return fmt.Errorf("write report %q: %w", path, err)
	}
	return nil
}
