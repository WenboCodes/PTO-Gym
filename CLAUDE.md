# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

PTO-Gym is a VPTO case repository. It primarily contains hand-authored `examples/pto/**` kernels plus the host-validation scripts used to lower, build, and validate those cases. It is not the main PTOAS implementation repo; the wheel workflow intentionally builds PTOAS from `mouliangyu/PTOAS` and uses this repo as a source of VPTO test cases.

## Common commands

### Run one VPTO case locally

```bash
mkdir -p .work/vpto-single
rm -rf .work/vpto-single/*

WORK_SPACE=$PWD/.work/vpto-single \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
CASE_NAME=micro-op/binary-vector/vadd \
DEVICE=SIM \
bash examples/pto/scripts/run_host_vpto_validation.sh
```

### Run many micro-op cases in parallel

```bash
mkdir -p .work/vpto-sim-microop-64
rm -rf .work/vpto-sim-microop-64/*

WORK_SPACE=$PWD/.work/vpto-sim-microop-64 \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
CASE_PREFIX=micro-op \
DEVICE=SIM \
JOBS=64 \
bash examples/pto/scripts/run_host_vpto_validation_parallel.sh
```

### Compile-only stop after kernel shared library

```bash
WORK_SPACE=$PWD/.work/vpto-compile-only \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
CASE_NAME=micro-op/binary-vector/vadd \
DEVICE=SIM \
COMPILE_ONLY=1 \
bash examples/pto/scripts/run_host_vpto_validation.sh
```

### Trigger the wheel workflow manually

```bash
gh workflow run build_wheel.yml
```

### Inspect the latest wheel workflow run

```bash
gh run list --workflow build_wheel.yml --limit 5
gh run view <run-id>
gh run view <run-id> --log-failed
```

## Important environment variables

- `WORK_SPACE`: required scratch/output directory for validation runs.
- `ASCEND_HOME_PATH`: required; scripts source `set_env.sh` from here when present.
- `PTOAS_BIN`: path to the built `ptoas` binary. Defaults to `build/tools/ptoas/ptoas` inside the checked-out source tree.
- `DEVICE`: defaults to `SIM`; controls whether validation runs locally through simulator libraries or through the host runner.
- `SIM_LIB_DIR`: optional override for simulator libs. If unset, the serial runner searches under `ASCEND_HOME_PATH` for `*/simulator/dav_3510/lib`.
- `CASE_NAME`: run a single case by path relative to `examples/pto/`.
- `CASE_PREFIX`: parallel runner filter for a subtree of cases.
- `COMPILE_ONLY=1`: for the serial runner, stop after lowering/building the kernel shared library and skip host execution / compare.

## Architecture overview

### VPTO case layout

Each runnable case under `examples/pto/**` is a directory with a fixed bundle of files:

- `kernel.pto`: source kernel
- `stub.cpp`: host stub used to embed the generated device object
- `launch.cpp`: launch-side wrapper compiled with Bisheng CCE flags
- `main.cpp`: host executable entry point
- `golden.py`: input / expected-output generation
- `compare.py`: result checker

The validation scripts treat that file set as the contract for a runnable case. Case discovery is directory-based rather than registry-based.

### Serial validation flow

`examples/pto/scripts/run_host_vpto_validation.sh` is the main entry point. Its pipeline is:

1. discover a case or set of cases under `examples/pto/`
2. lower `kernel.pto` with `ptoas` using `--pto-arch a5 --pto-backend=vpto --vpto-emit-hivm-llvm`
3. compile the emitted LLVM IR to a device object with `bisheng`
4. build `launch.cpp` and `stub.cpp`, then link `lib<case>_kernel.so`
5. unless `COMPILE_ONLY=1`, build the host executable from `main.cpp`, generate golden data, run the executable, and compare outputs

When debugging failures, first identify which stage failed: PTO lowering, Bisheng compile, kernel `.so` link, host executable build, runtime execution, or compare.

### Parallel validation flow

`examples/pto/scripts/run_host_vpto_validation_parallel.sh` is only a scheduler around the serial script. It:

- discovers cases with the same directory contract
- launches multiple serial validations in parallel
- writes per-case results to `parallel-summary.tsv`
- keeps the authoritative per-case logs under `WORK_SPACE/<case-token>/validation.log`

If a parallel run fails, inspect the individual case log rather than only the summary file.

### Wheel workflow architecture

`.github/workflows/build_wheel.yml` builds PTOAS wheels from the external PTOAS repo, not from PTO-Gym itself.

Key points:

- the workflow checks out `mouliangyu/PTOAS` as the build source
- it separately checks out the current PTO-Gym repo into `current-repo/`
- LLVM/MLIR is built or restored from cache under `/llvm-workspace/llvm-project/build-shared`
- PTOAS is configured and built from the external source checkout
- wheel validation has three layers:
  1. install/import test for the built wheel
  2. CLI lowering test against a case from the external PTOAS checkout
  3. CLI lowering test against a case from this repo (`current-repo/`)

This split is important: if the workflow fails only on the current-repo lowering step, the PTOAS build may still be healthy and the breakage is likely in the PTO-Gym case corpus.

### Release behavior

The wheel workflow is triggered by:

- `pull_request`
- `workflow_dispatch`
- `schedule`
- `release`

It no longer runs on plain `push`.

Scheduled and release runs execute `upload_release_assets`; manual and PR runs keep artifacts in Actions only.
