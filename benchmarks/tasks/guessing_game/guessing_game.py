from agents.config import BaseClientConfig
from tasks.guessing_game.benchmark_llm import BenchmarkLLM

import asyncio
from dataclasses import dataclass
from typing import Optional

from agents.gptswarm.guessing_game import guessing_game
from swarm.llm import BENCHMARK_MODEL_PREFIX

from tasks.base import VLLMConfigBase
from utils import utils
from utils.benchmarker import Benchmarker


@dataclass
class Config(VLLMConfigBase):
    max_value: int = 100
    ratio: str = "2/3"
    dump_file: Optional[str] = None

    num_participants: int = 20
    num_steps: int = 5

    @property
    def num_requests(self) -> int:
        # 2 requests per participant per step
        return self.num_participants * self.num_steps * 2


async def _benchmark(server_config: BaseClientConfig, benchmark_config: Config) -> None:
    benchmarker = Benchmarker(
        server_config, benchmark_config.disabled_pbar, seed=benchmark_config.seed
    )
    BenchmarkLLM.configure(benchmarker)
    benchmarker.set_num_requests(benchmark_config.num_requests)

    with benchmarker:
        await guessing_game.run(
            num_participants=benchmark_config.num_participants,
            num_steps=benchmark_config.num_steps,
            model_name=BENCHMARK_MODEL_PREFIX + server_config.model,
            prompt_path=guessing_game.DEFAULT_PROMPT_PATH,
            max_value=benchmark_config.max_value,
            ratio=benchmark_config.ratio,
            dump_file=benchmark_config.dump_file,
            async_exec=True,
        )

    input_requests, outputs = zip(*BenchmarkLLM.record)
    results = benchmarker.calculate_results(
        input_requests,
        outputs,
        benchmark_config.selected_percentile_metrics,
        benchmark_config.selected_percentiles,
    )

    benchmarker.report_results(results, benchmark_config.selected_percentile_metrics)
    benchmarker.save_results(results, benchmark_config.result_file_str)


def benchmark(
    server_config: BaseClientConfig, benchmark_config: Optional[Config] = None
) -> None:
    benchmark_config = benchmark_config or Config()
    utils.set_seed(benchmark_config.seed)
    asyncio.run(_benchmark(server_config, benchmark_config))


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
