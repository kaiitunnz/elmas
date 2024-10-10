#!/bin/bash
set -e

# Install local vLLM
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
cd /home/noppanat/Workspace/Projects/vllm
python python_only_dev.py
cd -

pip install -r requirements.txt

