from agents.config import BaseClientConfig

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from tasks.generative_agents.agent_functions import (
    action_location_object_prompt,
    action_location_sector_prompt,
    generate_event_triple_prompt,
    generate_pronunciatio_prompt,
    poignancy_event_prompt,
)
from tasks.base import SGLangConfigBase
from utils import utils
from utils.benchmarker import Benchmarker, RequestFuncOutput


@dataclass
class Config(SGLangConfigBase):
    task_root: Path = SGLangConfigBase.benchmark_root / "generative_agents"
    python_file_path: Path = task_root / "bench_other.py"
    data_path: Path = task_root / "agent_calls.jsonl"

    disabled_pbar: bool = False
    selected_percentile_metrics: List[str] = field(
        default_factory=lambda: ["ttft", "tpot", "itl"]
    )
    selected_percentiles: List[float] = field(default_factory=lambda: [99])

    num_events: int = 1000
    backend: str = "vllm"
    parallel: int = 1


def _benchmark(server_config: BaseClientConfig, benchmark_config: Config) -> None:
    lines = list(utils.read_jsonl(str(benchmark_config.data_path)))[
        : benchmark_config.num_events
    ]
    mapping = {
        "poignancy_event": poignancy_event_prompt,
        "generate_event_triple": generate_event_triple_prompt,
        "generate_pronunciatio": generate_pronunciatio_prompt,
        "action_location_sector": action_location_sector_prompt,
        "action_location_object": action_location_object_prompt,
    }

    arguments = [mapping[k](**v) for l in lines for k, v in l.items()]  # type: ignore
    outputs: List[RequestFuncOutput] = []

    benchmarker = Benchmarker(server_config, benchmark_config.disabled_pbar)
    input_requests = [benchmarker.create_input_request(**arg) for arg in arguments]
    request_func_inputs = [
        benchmarker.create_request_func_input(request) for request in input_requests
    ]
    benchmarker.set_num_requests(len(request_func_inputs))

    async def get_one_answer_async(request_func_input):
        answer = await benchmarker.async_request(request_func_input)
        outputs.append(answer)

    loop = asyncio.get_event_loop()
    with benchmarker:
        for arg in request_func_inputs:
            loop.run_until_complete(get_one_answer_async(arg))

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
    _benchmark(server_config, benchmark_config)


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
