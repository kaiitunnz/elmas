"""
Adapted from https://github.com/vllm-project/vllm.
"""

from agents.config import BaseClientConfig

import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, List, Optional, TypeVar

import numpy as np

from tasks.base import VLLMConfigBase
from utils import utils
from utils.benchmarker import AnyTokenizer, Benchmarker, InputRequest, RequestFuncOutput

T = TypeVar("T")


@dataclass
class Config(VLLMConfigBase):
    hf_path = Path(os.environ["HF_HOME"])

    dataset_name: str = "sharegpt"
    dataset_path: Path = hf_path / "datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    num_requests: int = 200
    request_rate: Optional[float] = 4


def sample_sharegpt_requests(
    tokenizer: AnyTokenizer,
    dataset_path: Path,
    num_requests: int,
    fixed_output_len: Optional[int] = None,
) -> List[InputRequest]:
    # Load the dataset.
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[InputRequest] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        prompt_len = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append(InputRequest(prompt, prompt_len, output_len))

    return filtered_dataset


async def generate_workload(
    requests: List[T],
    request_rate: float,
) -> AsyncGenerator[T, None]:
    requests_iter = iter(requests)
    for request in requests_iter:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def _benchmark(server_config: BaseClientConfig, benchmark_config: Config) -> None:
    request_rate = benchmark_config.request_rate or float("inf")

    benchmarker = Benchmarker(server_config, benchmark_config.disabled_pbar)
    input_requests = sample_sharegpt_requests(
        benchmarker.tokenizer,
        benchmark_config.dataset_path,
        benchmark_config.num_requests,
    )
    request_func_inputs = [
        benchmarker.create_request_func_input(request) for request in input_requests
    ]
    benchmarker.set_num_requests(len(request_func_inputs))

    tasks: List[asyncio.Task] = []
    with benchmarker:
        async for request_func_input in generate_workload(
            request_func_inputs, request_rate
        ):
            tasks.append(
                asyncio.create_task(benchmarker.async_request(request_func_input))
            )
        outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    results = benchmarker.calculate_results(
        input_requests,
        outputs,
        benchmark_config.selected_percentile_metrics,
        benchmark_config.selected_percentiles,
    )

    benchmarker.report_results(results, benchmark_config.selected_percentile_metrics)
    benchmarker.save_results(results, benchmark_config.result_file_str)


def benchmark(
    server_config: BaseClientConfig,
    result_file: Optional[Path] = None,
    **kwargs,
) -> None:
    benchmark_config = Config(result_file=result_file, **kwargs)
    utils.set_seed(benchmark_config.seed)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_benchmark(server_config, benchmark_config))


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
