from agents.config import BaseClientConfig

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tasks.multiturn_short.multiturn_short as multiturn_short
from utils import utils


@dataclass
class Config(multiturn_short.Config):
    num_qa: int = 5
    min_len_a: int = 256
    max_len_a: int = 512


def benchmark(
    server_config: BaseClientConfig, result_file: Optional[Path] = None, **kwargs
) -> None:
    benchmark_config = Config(result_file=result_file, **kwargs)
    utils.set_seed(benchmark_config.seed)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(multiturn_short._benchmark(server_config, benchmark_config))


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
