#!/bin/bash
set -e

# Install our version of vLLM
git clone https://github.com/kaiitunnz/vllm.git
cd vllm
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python python_only_dev.py
cd -

# Install our version of GPTSwarm
git clone https://github.com/J1shen/Pruned-Agent-Swarm.git GPTSwarm
cd GPTSwarm
pip install -e .
cd -

# Install this package
pip install -e .
