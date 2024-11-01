from agents.config import BaseClientConfig

import asyncio
import itertools
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tasks.base import SGLangConfigBase
from tasks.multiturn_short.data_gen import gen_arguments
from utils.benchmarker import (
    Benchmarker,
    InputRequest,
    RequestFuncOutput,
)


@dataclass
class Config(SGLangConfigBase):
    task_root: Path = SGLangConfigBase.benchmark_root / "multi_turn_chat"
    python_file_path: Path = task_root / "bench_other.py"

    disabled_pbar: bool = False
    selected_percentile_metrics: List[str] = field(
        default_factory=lambda: ["ttft", "tpot", "itl"]
    )
    selected_percentiles: List[float] = field(default_factory=lambda: [99])

    parallel: int = 64
    turns: int = 4
    num_qa: int = 20
    min_len_q: int = 256
    max_len_q: int = 512
    min_len_a: int = 4
    max_len_a: int = 8


def run_coroutine(coroutine: Callable[..., Coroutine], *args) -> Any:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine(*args))
    finally:
        loop.close()


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
                executor, run_coroutine, _multi_turns, benchmarker, qas["qas"]
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