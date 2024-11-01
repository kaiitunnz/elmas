from agents.config import BaseClientConfig

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tasks.base import BenchmarkConfigBase


@dataclass
class Config(BenchmarkConfigBase):
    vllm_path = Path.home() / "Workspace/Projects/vllm"
    hf_path = Path(os.environ["HF_HOME"])
    benchmark_path = vllm_path / "benchmarks/benchmark_serving.py"

    backend: str = "openai"
    endpoint: str = "/v1/completions"
    dataset_name: str = "sharegpt"
    dataset_path: Path = hf_path / "datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    request_rate: Optional[int] = None
    seed: int = 42

    @property
    def result_dir(self) -> str:
        assert self.result_file is not None
        return str(self.result_file.parent.absolute())
    
    @property
    def result_filename(self) -> str:
        assert self.result_file is not None
        return self.result_file.name


def benchmark(
    server_config: BaseClientConfig,
    result_file: Optional[Path] = None,
    **kwargs,
) -> None:
    benchmark_config = Config(result_file=result_file, **kwargs)

    subprocess.run(
        [
            "python",
            str(benchmark_config.benchmark_path),
            f"--host={server_config.host}",
            f"--port={server_config.port}",
            f"--endpoint={benchmark_config.endpoint}",
            f"--backend={benchmark_config.backend}",
            f"--dataset-name={benchmark_config.dataset_name}",
            f"--dataset-path={str(benchmark_config.dataset_path)}",
            f"--model={server_config.model}",
            f"--seed={benchmark_config.seed}",
            "--save-result",
            f"--result-dir={benchmark_config.result_dir}",
            f"--result-filename={benchmark_config.result_filename}",
        ]
        + (
            []
            if benchmark_config.request_rate is None
            else [f"--request-rate={benchmark_config.request_rate}"]
        )
    )


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
