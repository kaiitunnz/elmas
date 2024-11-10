#!/bin/bash

STANDARD_WORKLOADS="generative_agents guessing_game multiturn_long multiturn_short sharegpt"
MA_WORKLOADS="gptswarm_mmlu guessing_game_e2e guessing_game_e2e_cot"
MA_WORKLOADS_SERVERS="no-apc apc mt-apc mt-apc-no-sched"

ROOT_RESULT_DIR="$HOME/Workspace/Projects/elmas/results"

# Experiments with standard workloads
python -O benchmarks/runner.py \
    --benchmarks $STANDARD_WORKLOADS \
    --num-trials=5 \
    --result-dir "$ROOT_RESULT_DIR/standard" \
    --clear-result-dir

# Experiments with large-scale multi-agent workloads
python -O benchmarks/runner.py \
    --benchmarks $MA_WORKLOADS \
    --servers $MA_WORKLOADS_SERVERS \
    --num-trials=3 \
    --result-dir "$ROOT_RESULT_DIR/multi_agent" \
    --clear-result-dir
