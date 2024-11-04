from agents.config import BaseClientConfig

import asyncio
from dataclasses import dataclass, replace
from typing import Optional

from tasks.base import VLLMConfigBase
from tasks.guessing_game import guessing_game
from utils import utils


@dataclass
class Config(VLLMConfigBase):
    max_value: int = 100
    ratio: str = "2/3"
    dump_file: Optional[str] = "/home/noppanat/Workspace/Projects/elmas/logs/memory.log"

    num_participants: int = 1000
    num_steps: int = 5

    num_gpu_blocks_override: Optional[int] = None
    num_cpu_blocks_override: Optional[int] = None
    swap_space: int = 16
    max_model_len: Optional[int] = None

    @property
    def num_requests(self) -> int:
        # 2 requests per participant per step
        return self.num_participants * self.num_steps * 2

    def overwrite(self, server_config: BaseClientConfig) -> BaseClientConfig:
        return replace(
            server_config,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            num_cpu_blocks_override=self.num_cpu_blocks_override,
            swap_space=self.swap_space,
            max_model_len=self.max_model_len,
        )  # type: ignore


def benchmark(
    server_config: BaseClientConfig, benchmark_config: Optional[Config] = None
) -> None:
    benchmark_config = benchmark_config or Config()
    utils.set_seed(benchmark_config.seed)
    asyncio.run(guessing_game._benchmark(server_config, benchmark_config))


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
