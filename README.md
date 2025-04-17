# Efficient LLM Serving with Multi-Tier, Prefix-Aware KV Cache Sharing for Scalable Multi-Agent Systems

This repository contains the implementation of benchmarks described in the report and scripts for deploying the vLLM server and client programs. The implementation of **MT-APC** is available in a separate repository [here](https://github.com/kaiitunnz/elmas-vllm).

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Dependencies](#dependencies)
3. [Usage](#usage)
   - [Running Benchmark Suite](#running-benchmark-suite)
   - [Running Individual Benchmarks](#running-individual-benchmarks)
   - [Starting the vLLM Server](#starting-the-vllm-server)
   - [Client Programs](#client-programs)
4. [Contributing](#contributing)

---

## Overview

Recent advancements in large language models (LLMs) have significantly enhanced their performance and efficiency, enabling the development of single-agent and multi-agent (MA) systems for complex tasks such as automation and agent-based simulations. However, the high computational time and cost of LLM inference limit their applicability in sophisticated or large-scale tasks, particularly as input size grows with the number of agents in the system.

Since agents in most MA systems often share identical system prompts and engage in multi-turn interactions, KV cache optimization techniques like prefix-aware KV caching can be used to reuse the KV cache across requests with common prefixes, thereby eliminating redundant computations and reducing memory requirements. Nevertheless, as the number of agents scales to thousands or millions, the effectiveness of KV cache sharing diminishes, leading to increased recomputations.

To address this, we introduce MT-APC, a novel prefix-aware KV caching mechanism that leverages a memory hierarchy, including GPU, CPU, and disk storage, to store a large KV cache. Combined with asynchronous KV cache movement and device- and prefix-aware scheduling mechanisms, our system efficiently caches and reuses a larger amount of KV tensors. We evaluate our system on large-scale MA workloads, demonstrating its effectiveness in improving throughput and reducing time to first token.

This repository supports evaluating and benchmarking MT-APC and other baselines under different workloads and includes:

- Scripts to set up and deploy the vLLM server and client programs.
- Tools for evaluating the serving systems' performance with different kinds of workloads, including large-scale multi-agent applications.

## Getting Started

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/kaiitunnz/elmas.git
   cd elmas
   ```

2. Create a Conda environment:

   ```bash
   conda create -n elmas python=3.11 && conda activate elmas
   ```

3. Install dependencies:

   ```bash
   bash scripts/install.sh
   ```

4. Create a `.env` file for default server configuration. See [.env.example](./.env.example) for reference.

Note: If you encounter the following error: `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`, add the following line to `<PYTHON_LIBRARY_PATH>/huggingface_hub/__init__.py`.

```python
cached_download = None
```

### Dependencies

The project depends on the following:

- Python version 3.11 (tested only on this version)
- CUDA-compatible GPU (for GPU acceleration)
- Dependencies such as our versions of vLLM and GPTSwarm. See [scripts/install.sh](./scripts/install.sh) for installation.

## Usage

After installing this package and its dependencies with [scripts/install.sh](./scripts/install.sh), you can perform the following actions.

### Running Benchmark Suite

We provide a script to run the benchmarks we used in the report. See [scripts/experiments.sh](./scripts/experiments.sh) or run the following command.

```bash
bash scripts/experiments.sh
```

### Running Individual Benchmarks

The following command template can be used to run individual benchmarks.

```bash
python -O benchmarks/runner.py \
   --benchmarks <benchmark-name> \
   --servers <server-name> \
   --num-trials=5 \
   --result-dir=</path/to/result/dir> \
   --clear-result-dir
```

See [benchmarks/runner.py](./benchmarks/runner.py) or run the following command to see the lists of benchmarks and servers.

```bash
python benchmarks/runner.py --help
```

### Starting the vLLM Server

We provide a script for starting the vLLM server with various options. Below are some examples. You can remove the `-O` flag to enable assert statements.

1. vLLM server without prefix caching

   ```bash
   python -Om agents.utils.vllm.start_server \
      --preemption-mode=recompute
   ```

2. vLLM server with APC

   ```bash
   python -Om agents.utils.vllm.start_server \
      --enable-prefix-caching \
      --preemption-mode=recompute
   ```

3. vLLM server with MT-APC

   ```bash
   python -Om agents.utils.vllm.start_server \
      --enable-prefix-caching \
      --enable-multi-tier-prefix-caching \
      --enable-async-swapping \
      --enable-prefix-aware-scheduling \
      --enable-async-prefetching \
      --scheduler-window-size=10 \
      --preemption-mode=recompute
   ```

4. vLLM server with MT-APC and profiling enabled

   ```bash
   python -Om agents.utils.vllm.start_server \
      --enable-prefix-caching \
      --enable-multi-tier-prefix-caching \
      --enable-async-swapping \
      --enable-prefix-aware-scheduling \
      --enable-async-prefetching \
      --scheduler-window-size=10 \
      --profiling \
      --preemption-mode=recompute
   ```

### Client Programs

We provide several example client programs listed below.

1. Chatbot applications

   ```bash
   python -m agents.chatbot.chatbot    # Chatbot assistant
   python -m agents.chatbot.completion # LLM completion
   python -m agents.chatbot.profile    # Simple prompts for profiling the server
   ```

2. GPTSwarm's agent applications

   ```bash
   python -m agents.gptswarm.guessing_game --num-participants=20 --num-steps=5   # Guessing Game simulation
   python -m agents.gptswarm.gaia         # GAIA application
   python -m agents.gptswarm.crosswords   # Mini CrossWords application
   ```

### Contributing

We welcome contributions! Please: 1. Fork the repository. 2. Create a new branch for your feature or bugfix. 3. Submit a pull request detailing your changes.

For further questions, please contact the authors listed in the paper or open an issue in this repository.
