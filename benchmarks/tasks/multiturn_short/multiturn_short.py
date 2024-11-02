from agents.config import BaseClientConfig

import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tasks.base import VLLMConfigBase
from tasks.multiturn_short.data_gen import gen_arguments
from utils import utils
from utils.benchmarker import (
    Benchmarker,
    InputRequest,
    RequestFuncOutput,
)


@dataclass
class Config(VLLMConfigBase):
    parallel: int = 64
    turns: int = 4
    num_qa: int = 20
    min_len_q: int = 256
    max_len_q: int = 512
    min_len_a: int = 4
    max_len_a: int = 8


async def _multi_turns(
    benchmarker: Benchmarker,
    qas: List[Dict[str, Any]],
) -> List[Tuple[InputRequest, RequestFuncOutput]]:
    s = ""
    outputs = []
    for qa in qas:
        s += qa["prompt"]
        input_request = benchmarker.create_input_request(
            prompt=s, max_tokens=qa["new_tokens"]
        )
        request_func_input = benchmarker.create_request_func_input(input_request)
        output = await benchmarker.async_request(request_func_input)
        s += output.generated_text
        outputs.append((input_request, output))
    return outputs


async def _benchmark(server_config: BaseClientConfig, benchmark_config: Config) -> None:
    benchmarker = Benchmarker(server_config, benchmark_config.disabled_pbar)
    multi_qas = gen_arguments(
        turns=benchmark_config.turns,
        num_qa=benchmark_config.num_qa,
        min_len_q=benchmark_config.min_len_q,
        max_len_q=benchmark_config.max_len_q,
        min_len_a=benchmark_config.min_len_a,
        max_len_a=benchmark_config.max_len_a,
        tokenizer=benchmarker.tokenizer,
    )
    num_requests = benchmark_config.turns * benchmark_config.num_qa
    benchmarker.set_num_requests(num_requests)

    executor = ThreadPoolExecutor(max_workers=benchmark_config.parallel)
    loop = asyncio.get_event_loop()
    with benchmarker:
        futures = [
            loop.run_in_executor(
                executor, utils.run_coroutine, _multi_turns, benchmarker, qas["qas"]
            )
            for qas in multi_qas
        ]
        outputs_list: List[List[Tuple[InputRequest, RequestFuncOutput]]] = (
            await asyncio.gather(*futures)
        )

    input_requests: List[InputRequest] = []
    outputs: List[RequestFuncOutput] = []
    for input_request, output in itertools.chain(*outputs_list):
        input_requests.append(input_request)
        outputs.append(output)
    results = benchmarker.calculate_results(
        input_requests,
        outputs,
        benchmark_config.selected_percentile_metrics,
        benchmark_config.selected_percentiles,
    )

    benchmarker.report_results(results, benchmark_config.selected_percentile_metrics)
    benchmarker.save_results(results, benchmark_config.result_file_str)


def benchmark(
    server_config: BaseClientConfig, result_file: Optional[Path] = None, **kwargs
) -> None:
    benchmark_config = Config(result_file=result_file, **kwargs)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_benchmark(server_config, benchmark_config))


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
