#!/bin/bash

MICROBENCHMARKS="generative_agents guessing_game multiturn_long multiturn_short sharegpt"
E2E_BENCHMARKS="gptswarm_mmlu guessing_game_e2e guessing_game_e2e_cot"
E2E_BENCHMARK_SERVERS="no-apc apc mt-apc mt-apc-no-sched"

ROOT_RESULT_DIR="$HOME/Workspace/Projects/elmas/results"

# Microbenchmarking experiments
python -O benchmarks/runner.py \
    --benchmarks $MICROBENCHMARKS \
    --num-trials=5\
    --result-dir "$ROOT_RESULT_DIR/microbenchmarks" \
    --clear-result-dir

# End-to-end experiments
python -O benchmarks/runner.py \
    --benchmarks $E2E_BENCHMARKS \
    --servers $E2E_BENCHMARK_SERVERS \
    --num-trials=3\
    --result-dir "$ROOT_RESULT_DIR/e2e" \
    --clear-result-dir
