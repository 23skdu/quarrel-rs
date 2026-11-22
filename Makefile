# quarrel-rs developer Makefile
#
# Every target is self-documenting: `make help` prints the list below.
# Tool installation is intentionally explicit (`make tools`) so CI and containers do not
# accidentally mutate a developer's Homebrew prefix.

.DEFAULT_GOAL := help
.SHELLFLAGS := -eu -o pipefail -c
SHELL := /bin/bash

CARGO ?= cargo
BREW ?= brew
HELM ?= helm
DOCKER ?= docker
COMPOSE ?= $(DOCKER) compose
JQ ?= jq
YQ ?= yq
VALGRIND ?= valgrind
IAI_RUNNER ?= iai-callgrind-runner
IAI_VERSION ?= 0.16.1
IAI_MANIFEST := benches/iai/Cargo.toml
BASELINE_DIR := benches/baselines
CRITERION_BENCHES := quarrel-gguf:matvec quarrel-gguf:quant quarrel-simd:kernels \
	quarrel-engine:sampler quarrel-flight:arrow_roundtrip
IMAGE ?= quarrel-rs:local
CUDA_VERSION ?= 12.8.1
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || printf dev)
REVISION ?= $(shell git rev-parse HEAD 2>/dev/null || printf unknown)
SOURCE_DATE_EPOCH ?= $(shell git log -1 --format=%ct 2>/dev/null || printf 0)
DOCKER_BUILD_ARGS ?= --build-arg VERSION=$(VERSION) --build-arg REVISION=$(REVISION) \
	--build-arg SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH)

BREW_TOOLS := cargo-nextest cargo-deny cargo-audit bacon cargo-llvm-cov cargo-edit \
	cargo-mutants cargo-bloat cargo-outdated taplo yq jq shellcheck helm
CARGO_TOOLS := cargo-fuzz cargo-udeps cargo-geiger
MIRI_TOOLCHAIN := nightly
FUZZ_TARGETS := gguf_parse sampling_params arrow_ipc
REQUIRED_TOOLS := cargo rustfmt helm docker shellcheck

OPENAPI_FILE := api/openapi.json
HOOKS_DIR := $(shell git rev-parse --git-path hooks 2>/dev/null || printf '.git/hooks')

# iai-callgrind is deliberately opt-in: it needs valgrind and a runner binary
# whose version matches the pinned iai-callgrind crate in benches/iai/Cargo.toml.
define require_iai_tools
	@command -v $(VALGRIND) >/dev/null 2>&1 || { echo "$(VALGRIND) is required for iai benchmarks"; exit 1; }
	@command -v $(IAI_RUNNER) >/dev/null 2>&1 || { echo "$(IAI_RUNNER) $(IAI_VERSION) is required: cargo install iai-callgrind-runner --version $(IAI_VERSION) --locked"; exit 1; }
endef

.PHONY: help tools tools-brew tools-cargo tools-miri check-tools makehooks unmakehooks precommit \
	build build-release run test test-unit test-benchtool nextest fmt fmt-check lint audit deny \
	security shellcheck test-pythonsdk \
	coverage coverage-floors bench bench-iai bench-iai-baseline bench-iai-compare \
	bench-criterion-baseline bench-criterion-compare bench-baselines \
	bench-quality bench-performance bench-regression \
	fuzz fuzz-fmt fuzz-fmt-check fuzz-smoke fuzz-coverage miri mutants outdated doc bloat openapi \
	openapi-check metrics-check \
	docker-build docker-build-cpu docker-build-cuda docker-build-distroless docker-build-all \
	container-check docker-up docker-down docker-logs \
	helm-lint helm-template links ci clean distclean

## Display this help message
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nquarrel-rs — available targets\n\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2} \
		/^## / {printf "\n  %s\n", substr($$0, 4)} END {printf "\n"}' $(MAKEFILE_LIST)

## Install all development tools (Homebrew formulas, cargo subcommands and nightly Miri)
tools: tools-brew tools-cargo tools-miri

## Install Homebrew development tools (idempotent)
tools-brew:
	@command -v $(BREW) >/dev/null 2>&1 || { echo "brew is required: https://brew.sh"; exit 1; }
	$(BREW) install $(BREW_TOOLS)

## Install cargo subcommands that Homebrew does not package
tools-cargo:
	$(CARGO) install $(CARGO_TOOLS) --locked

## Install nightly Rust with the miri component (Miri is a rustup component, not a crate)
tools-miri:
	rustup toolchain install $(MIRI_TOOLCHAIN) --profile minimal --component miri --component rust-src

## Report missing required tools and how to install them
check-tools:
	@missing=0; \
	for tool in $(REQUIRED_TOOLS); do \
		if command -v $$tool >/dev/null 2>&1; then \
			printf '  ok       %s\n' "$$tool"; \
		else \
			printf '  MISSING  %s  (run: make tools)\n' "$$tool"; missing=1; \
		fi; \
	done; \
	exit $$missing

## Install the pre-commit hook that runs unit tests and rustfmt
makehooks:
	@mkdir -p "$(HOOKS_DIR)"
	@install -m 0755 scripts/pre-commit "$(HOOKS_DIR)/pre-commit"
	@echo "installed scripts/pre-commit -> $(HOOKS_DIR)/pre-commit"

## Remove the installed pre-commit hook
unmakehooks:
	@rm -f "$(HOOKS_DIR)/pre-commit"
	@echo "removed $(HOOKS_DIR)/pre-commit"

## Run the pre-commit checks manually
precommit:
	bash scripts/pre-commit

## Debug build of the whole workspace
build:
	$(CARGO) build --workspace --locked

## Optimised release build of the whole workspace
build-release:
	$(CARGO) build --workspace --release --locked

## Build and run the control plane
run:
	$(CARGO) run --locked -p quarrel-server

## Full test suite (nextest when available, cargo test otherwise)
test:
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		$(CARGO) nextest run --workspace --locked; \
	else \
		$(CARGO) test --workspace --locked; \
	fi

## Unit tests only — the same set enforced by the pre-commit hook
test-unit:
	$(CARGO) test --workspace --locked --lib --bins

## Test the pyarrow Flight SDK (skips pyarrow-dependent cases when unavailable)
test-pythonsdk:
	python3 -m unittest discover -s pythonsdk/tests -v

## Test the Go benchtool and its Python driver (no Rust build required)
test-benchtool:
	@test -z "$$(gofmt -l cmd/benchtool)" || { echo "gofmt required for cmd/benchtool"; exit 1; }
	go vet ./cmd/benchtool
	go test ./cmd/benchtool
	python3 -m py_compile scripts/bench.py scripts/test_bench.py
	python3 -m unittest discover -s scripts -p 'test_*.py' -v
	python3 scripts/bench.py --profile quality --dry-run

## Explicit cargo test, useful when debugging a nextest-only failure
nextest:
	$(CARGO) test --workspace --locked

## Apply rustfmt to every crate
fmt:
	$(CARGO) fmt --all

## Fail if any file is not rustfmt-clean
fmt-check:
	$(CARGO) fmt --all -- --check

## Clippy with warnings denied across all targets
lint:
	$(CARGO) clippy --workspace --all-targets --all-features --locked -- -D warnings

## Audit the lockfile for known vulnerabilities
audit:
	$(CARGO) audit

## Check licenses, duplicate versions, bans and advisories
deny:
	$(CARGO) deny check

## Run cargo audit and cargo deny together
security: audit deny

## Shellcheck every tracked shell script
shellcheck:
	@git ls-files '*.sh' scripts/pre-commit | xargs -r shellcheck

## HTML coverage report under target/llvm-cov/html
coverage:
	$(CARGO) llvm-cov --workspace --html

## Enforce the per-area coverage floors from docs/gates.md
coverage-floors:
	$(CARGO) llvm-cov -p quarrel-core -p quarrel-gguf -p quarrel-simd --fail-under-lines 85
	$(CARGO) llvm-cov -p quarrel-engine -p quarrel-api -p quarrel-server --fail-under-lines 80

## Criterion benchmarks only; iai is opt-in via bench-iai
bench:
	$(CARGO) bench --workspace --locked

## Run pinned iai-callgrind SIMD benchmarks (needs valgrind + runner)
bench-iai:
	$(call require_iai_tools)
	$(CARGO) bench --locked --manifest-path $(IAI_MANIFEST) --bench hot_kernels

## Refresh the local iai-callgrind reference baseline artifact
bench-iai-baseline:
	$(call require_iai_tools)
	@mkdir -p $(BASELINE_DIR)
	@{ \
		printf '%s\n' '# Local reference baseline — NOT a cross-machine gate.'; \
		printf '%s\n' '# Refresh only on an idle, documented machine: make bench-iai-baseline'; \
		printf 'recorded_utc: %s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
		printf 'counts_command: %s\n' '$(CARGO) bench --locked --manifest-path $(IAI_MANIFEST) --bench hot_kernels'; \
		printf 'baseline_command: %s\n' '$(CARGO) bench --locked --manifest-path $(IAI_MANIFEST) --bench hot_kernels -- --save-baseline=committed'; \
		printf 'rustc: %s\n' "$$(rustc -V 2>/dev/null || printf unavailable)"; \
		printf 'valgrind: %s\n' "$$($(VALGRIND) --version 2>/dev/null || printf unavailable)"; \
		printf 'iai-callgrind: %s\n' '$(IAI_VERSION) (crate and runner)'; \
		printf 'cpu: %s\n' "$$(sed -n 's/^model name[^:]*: *//p' /proc/cpuinfo | head -1)"; \
		printf 'mem_total: %s\n' "$$(awk '/MemTotal/ {printf "%.1f GiB", $$2 / 1048576}' /proc/meminfo)"; \
		printf '\n'; \
	} > $(BASELINE_DIR)/iai-callgrind-local.txt
	$(CARGO) bench --locked --manifest-path $(IAI_MANIFEST) --bench hot_kernels 2>&1 | tee -a $(BASELINE_DIR)/iai-callgrind-local.txt
	$(CARGO) bench --locked --manifest-path $(IAI_MANIFEST) --bench hot_kernels -- --save-baseline=committed >/dev/null 2>&1
	@printf '\nbaseline: committed (saved under benches/iai/target/iai, not committed)\n' >> $(BASELINE_DIR)/iai-callgrind-local.txt

## Compare current iai-callgrind counts with the last local baseline
bench-iai-compare:
	$(call require_iai_tools)
	$(CARGO) bench --locked --manifest-path $(IAI_MANIFEST) --bench hot_kernels -- --baseline=committed

## Refresh the local Criterion reference baseline artifact
bench-criterion-baseline:
	@mkdir -p $(BASELINE_DIR)
	@{ \
		printf '%s\n' '# Local reference baseline — NOT a cross-machine gate.'; \
		printf '%s\n' '# Refresh only on an idle, documented machine: make bench-criterion-baseline'; \
		printf 'recorded_utc: %s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
		printf 'command: %s\n' 'for each CRITERION_BENCH: $(CARGO) bench --locked -p <crate> --bench <name> -- --save-baseline committed'; \
		printf 'rustc: %s\n' "$$(rustc -V 2>/dev/null || printf unavailable)"; \
		printf 'python3: %s\n' "$$(python3 --version 2>/dev/null || printf unavailable)"; \
		printf 'go: %s\n' "$$(go version 2>/dev/null || printf unavailable)"; \
		printf 'cpu: %s\n' "$$(sed -n 's/^model name[^:]*: *//p' /proc/cpuinfo | head -1)"; \
		printf 'mem_total: %s\n' "$$(awk '/MemTotal/ {printf "%.1f GiB", $$2 / 1048576}' /proc/meminfo)"; \
		printf '\n'; \
	} > $(BASELINE_DIR)/criterion-local.txt
	{ for spec in $(CRITERION_BENCHES); do \
		pkg=$${spec%%:*}; bench=$${spec##*:}; \
		echo "==> $$pkg/$$bench"; \
		$(CARGO) bench --locked -p $$pkg --bench $$bench -- --save-baseline committed; \
	done; } 2>&1 | tee -a $(BASELINE_DIR)/criterion-local.txt

## Compare current Criterion timings with the last local baseline
bench-criterion-compare:
	@for spec in $(CRITERION_BENCHES); do \
		pkg=$${spec%%:*}; bench=$${spec##*:}; \
		echo "==> $$pkg/$$bench"; \
		$(CARGO) bench --locked -p $$pkg --bench $$bench -- --baseline committed; \
	done

## Refresh both committed local baseline artifacts (slow)
bench-baselines: bench-criterion-baseline bench-iai-baseline

## Replay all quality gates through the Go benchtool
bench-quality:
	python3 scripts/bench.py --profile quality

## Run timed performance checks through the Go benchtool
bench-performance:
	python3 scripts/bench.py --profile performance

## Replay quality and performance gates for a release candidate
bench-regression:
	python3 scripts/bench.py --profile regression

## Fuzz every target for 60 s (nightly + cargo-fuzz required)
fuzz:
	@cd fuzz && for target in $(FUZZ_TARGETS); do \
		echo "==> fuzzing $$target for 60s"; \
		$(CARGO) fuzz run $$target -- -max_total_time=60; \
	done

## Short 5 s fuzz pass per target, suitable for pull requests
fuzz-smoke:
	@cd fuzz && for target in $(FUZZ_TARGETS); do \
		echo "==> fuzzing $$target for 5s"; \
		$(CARGO) fuzz run $$target -- -max_total_time=5; \
	done

## Generate coverage reports for every fuzz target corpus
fuzz-coverage:
	@cd fuzz && for target in $(FUZZ_TARGETS); do \
		echo "==> fuzz coverage for $$target"; \
		$(CARGO) fuzz coverage $$target; \
	done

## Apply rustfmt to the excluded fuzz crate with the pinned stable toolchain
fuzz-fmt:
	$(CARGO) fmt --manifest-path fuzz/Cargo.toml -p quarrel-fuzz

## Fail if the fuzz crate is not rustfmt-clean
fuzz-fmt-check:
	$(CARGO) fmt --manifest-path fuzz/Cargo.toml -p quarrel-fuzz -- --check

## Run the workspace under Miri (requires `make tools-miri`)
miri:
	$(CARGO) +$(MIRI_TOOLCHAIN) miri test --workspace

## Mutation testing of core, SIMD and sampler logic
mutants:
	@for crate in quarrel-core quarrel-simd quarrel-engine; do \
		echo "==> mutating $$crate"; \
		$(CARGO) mutants -p $$crate; \
	done

## List outdated dependencies without changing the lockfile
outdated:
	$(CARGO) outdated --workspace

## Build and open the workspace documentation
doc:
	$(CARGO) doc --workspace --no-deps --open

## Report what contributes to the release binary size
bloat:
	$(CARGO) bloat --release --crates

## Regenerate api/openapi.json from the axum handlers
openapi:
	$(CARGO) run --quiet --locked -p quarrel-server -- --print-openapi > $(OPENAPI_FILE)
	@echo "wrote $(OPENAPI_FILE)"

## Fail when the committed OpenAPI contract is stale
openapi-check:
	@tmp=$$(mktemp); \
	$(CARGO) run --quiet --locked -p quarrel-server -- --print-openapi > "$$tmp"; \
	if ! diff -u $(OPENAPI_FILE) "$$tmp"; then \
		echo "OpenAPI drift detected: run 'make openapi'"; rm -f "$$tmp"; exit 1; \
	fi; \
	rm -f "$$tmp"
	@echo "OpenAPI contract is current"

## Check that metric names in code, alerts and the dashboard agree
metrics-check:
	@for metric in quarrel_requests_total quarrel_request_duration_seconds \
		quarrel_build_info quarrel_process_resident_bytes quarrel_engine_errors_total; do \
		grep -q "$$metric" crates/quarrel-server/src/metrics.rs || { echo "missing in code: $$metric"; exit 1; }; \
		grep -q "$$metric" deploy/grafana/dashboards/quarrel-overview.json || { echo "missing in dashboard: $$metric"; exit 1; }; \
	done
	@grep -q 'quarrel_requests_total' deploy/prometheus/alerts/quarrel.rules.yml
	@echo "metric names agree across code, alerts and dashboard"

## Build the CPU container image (default)
docker-build: docker-build-cpu

## Build the CPU container image explicitly
docker-build-cpu:
	$(DOCKER) build $(DOCKER_BUILD_ARGS) -t $(IMAGE) -f Dockerfile .

## Build the CUDA-base container image (GPU kernels land in Phase 6)
docker-build-cuda:
	$(DOCKER) build $(DOCKER_BUILD_ARGS) --build-arg CUDA_VERSION=$(CUDA_VERSION) \
		-t $(IMAGE)-cuda -f Dockerfile.cuda .

## Build the distroless container image
docker-build-distroless:
	$(DOCKER) build $(DOCKER_BUILD_ARGS) -t $(IMAGE)-distroless -f Dockerfile.distroless .

## Build all three container variants
docker-build-all: docker-build-cpu docker-build-cuda docker-build-distroless

## Validate all container definitions without building them
container-check:
	@for file in Dockerfile Dockerfile.cuda Dockerfile.distroless; do \
		echo "==> checking $$file"; \
		$(DOCKER) build --check $(DOCKER_BUILD_ARGS) -f $$file .; \
	done

## Start quarrel-rs, Prometheus and Grafana
docker-up:
	$(COMPOSE) up -d --build

## Stop the local monitoring stack
docker-down:
	$(COMPOSE) down --remove-orphans

## Tail logs from the local stack
docker-logs:
	$(COMPOSE) logs -f

## Lint the Helm chart
helm-lint:
	$(HELM) lint helm/quarrel --strict

## Render the Helm chart (override with VALUES=path/to/values.yaml)
helm-template:
	$(HELM) template quarrel helm/quarrel -f $(or $(VALUES),helm/quarrel/values.yaml)

## Print useful project links
links:
	@echo "upstream:   https://github.com/23skdu/longbow-quarrel"
	@echo "roadmap:    https://github.com/23skdu/quarrel-rs/blob/main/docs/roadmap.md"
	@echo "api docs:   http://localhost:8080/docs"
	@echo "metrics:    http://localhost:8080/metrics"
	@echo "prometheus: http://localhost:9090"
	@echo "grafana:    http://localhost:3000"

## Full local gate sequence used by CI
ci: fmt-check fuzz-fmt-check lint test test-pythonsdk security openapi-check metrics-check helm-lint container-check
	$(COMPOSE) config --quiet

## Remove build artefacts, generated caches and unused Docker objects (named volumes kept)
clean:
	$(CARGO) clean
	rm -rf target/llvm-cov target/criterion .coverage coverage .pytest_cache
	find . -name '*.profraw' -delete
	-$(DOCKER) system prune -af

## clean plus local runtime state (never touches tracked source files)
distclean: clean docker-down
	rm -f .env.local
	rm -rf /tmp/opencode/quarrel-rs-*
