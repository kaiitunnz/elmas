from agents.config import BaseProfilingConfig

import subprocess
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict

from agents.utils.vllm.profiling import VLLMProfiling

@dataclass
class ProfilingConfig(BaseProfilingConfig):
    # max_tokens: int = 300
    max_tokens: int = 2
    n: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.8
    echo: bool = True

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "echo": self.echo,
        }
    

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("cmd", type=str, help="Command to run")

    return parser.parse_args()


def main(cmd: str):
    config = ProfilingConfig()
    profiling = VLLMProfiling(config)
    profiling.start()
    subprocess.call(cmd, shell=True)
    profiling.stop()

if __name__ == "__main__":
    args = parse_args()
    main(args.cmd)
