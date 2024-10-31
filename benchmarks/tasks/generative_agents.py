from agents.config import BaseClientConfig

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import SGLangConfigBase


@dataclass
class Config(SGLangConfigBase):
    task_root: Path = SGLangConfigBase.benchmark_root / "generative_agents"
    python_file_path: Path = task_root / "bench_other.py"
    data_path: Path = task_root / "agent_calls.jsonl"

    num_events: int = 1000
    backend: str = "vllm"
    parallel: int = 1


def benchmark(
    server_config: BaseClientConfig, result_file: Optional[Path] = None
) -> None:
    benchmark_config = Config(result_file=result_file)
    host = f"http://{server_config.host}"
    port = server_config.port

    subprocess.run(
        [
            "python",
            str(benchmark_config.python_file_path),
            f"--host={host}",
            f"--port={port}",
            f"--data-path={benchmark_config.data_path}",
            f"--num-events={benchmark_config.num_events}",
            f"--backend={benchmark_config.backend}",
            f"--parallel={benchmark_config.parallel}",
            f"--result-file={benchmark_config.result_file_str}",
        ]
    )


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
