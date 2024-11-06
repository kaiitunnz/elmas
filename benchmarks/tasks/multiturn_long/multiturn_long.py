from agents.config import BaseClientConfig

import asyncio
from dataclasses import dataclass
from typing import Optional

import tasks.multiturn_short.multiturn_short as multiturn_short
from utils import utils


@dataclass
class Config(multiturn_short.Config):
    num_qa: int = 10
    min_len_a: int = 64
    max_len_a: int = 128


def benchmark(
    server_config: BaseClientConfig, benchmark_config: Optional[Config] = None
) -> None:
    benchmark_config = benchmark_config or Config()
    utils.set_seed(benchmark_config.seed)
    asyncio.run(multiturn_short._benchmark(server_config, benchmark_config))


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
