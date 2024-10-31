from agents.config import BaseClientConfig

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import SGLangConfigBase


@dataclass
class Config(SGLangConfigBase):
    task_root: Path = SGLangConfigBase.benchmark_root / "multi_turn_chat"
    python_file_path: Path = task_root / "bench_other.py"

    backend: str = "vllm"


def benchmark(
    server_config: BaseClientConfig, result_file: Optional[Path] = None
) -> None:
    benchmark_config = Config(result_file=result_file)
    host = f"http://{server_config.host}"
    port = server_config.port
    tokenizer = server_config.model

    subprocess.run(
        [
            "python",
            str(benchmark_config.python_file_path),
            f"--host={host}",
            f"--port={port}",
            f"--tokenizer={tokenizer}",
            f"--backend={benchmark_config.backend}",
            f"--result-file={benchmark_config.result_file_str}",
        ]
    )


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
