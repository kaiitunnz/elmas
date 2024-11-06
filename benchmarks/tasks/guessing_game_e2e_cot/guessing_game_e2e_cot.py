from agents.config import BaseClientConfig

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from agents.gptswarm.guessing_game.prompt.prompt_set import PromptSet

from tasks.base import VLLMConfigBase
from tasks.guessing_game import guessing_game
from utils import utils


@dataclass
class Config(VLLMConfigBase):
    max_value: int = 100
    ratio: str = "2/3"
    dump_file: Optional[str] = str(Path(__file__).resolve().parent / "memory.log")

    num_participants: int = 800
    num_steps: int = 5

    num_gpu_blocks_override: Optional[int] = None
    num_cpu_blocks_override: Optional[int] = None
    swap_space: int = 16
    max_model_len: Optional[int] = None

    scheduler_window_size: Optional[int] = 10

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
            scheduler_window_size=self.scheduler_window_size,
        )  # type: ignore


def benchmark(
    server_config: BaseClientConfig, benchmark_config: Optional[Config] = None
) -> None:
    benchmark_config = benchmark_config or Config()
    utils.set_seed(benchmark_config.seed)
    tmp_user_prompt = PromptSet._user_prompts.copy()
    PromptSet._user_prompts[0] = tmp_user_prompt[-1]
    asyncio.run(guessing_game._benchmark(server_config, benchmark_config))
    PromptSet._user_prompts = tmp_user_prompt


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
