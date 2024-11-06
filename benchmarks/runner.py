from agents.config import BaseClientConfig

import json
import logging
import multiprocessing as mp
import shutil
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.synchronize import Semaphore
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from agents.utils.vllm.start_server import BaseServerConfig, start_server

import pandas as pd

from tasks.base import BenchmarkConfigBase
from tasks import (
    generative_agents,
    gptswarm_mmlu,
    guessing_game,
    guessing_game_e2e,
    guessing_game_e2e_cot,
    multiturn_long,
    multiturn_short,
    sharegpt,
)
from utils.logging import init_logger

BENCHMARK_FUNC_TYPE = Callable[[BaseClientConfig, Optional[BenchmarkConfigBase]], None]
BENCHMARK_CONFIG_TYPE = Callable[..., BenchmarkConfigBase]

logger = init_logger()


@dataclass
class ServerConfig(BaseServerConfig):
    DEFAULT_SWAP_SPACE = 4
    log_file: Optional[Path] = None

    preemption_mode: Optional[str] = "recompute"
    num_gpu_blocks_override: Optional[int] = 512
    num_cpu_blocks_override: Optional[int] = 512
    swap_space: Optional[int] = None
    max_model_len: Optional[int] = 8192
    block_size: int = 16

    enable_prefix_caching: bool = True

    enable_multi_tier_prefix_caching: bool = True
    enable_async_swapping: bool = True
    enable_prefix_aware_scheduling: bool = True
    enable_async_prefetching: bool = True
    scheduler_window_size: Optional[int] = 10

    def default_args(self) -> Namespace:
        return Namespace(
            model=self.model,
            host=self.host,
            port=self.port,
            device=self.device,
            uvicorn_log_level=self.uvicorn_log_level,
            preemption_mode=self.preemption_mode,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            num_cpu_blocks_override=self.num_cpu_blocks_override,
            swap_space=self.swap_space or self.DEFAULT_SWAP_SPACE,
            max_model_len=self.max_model_len,
            block_size=self.block_size,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_multi_tier_prefix_caching=self.enable_multi_tier_prefix_caching,
            enable_async_swapping=self.enable_async_swapping,
            enable_prefix_aware_scheduling=self.enable_prefix_aware_scheduling,
            enable_async_prefetching=self.enable_async_prefetching,
            scheduler_window_size=self.scheduler_window_size,
        )


SERVER_CONFIGS = {
    "no-apc": ServerConfig(
        enable_prefix_caching=False,
        enable_multi_tier_prefix_caching=False,
        enable_async_swapping=False,
        enable_prefix_aware_scheduling=False,
        enable_async_prefetching=False,
    ),
    "apc": ServerConfig(
        enable_prefix_caching=True,
        enable_multi_tier_prefix_caching=False,
        enable_async_swapping=False,
        enable_prefix_aware_scheduling=False,
        enable_async_prefetching=False,
    ),
    "mt-apc": ServerConfig(
        enable_prefix_caching=True,
        enable_multi_tier_prefix_caching=True,
        enable_async_swapping=True,
        enable_prefix_aware_scheduling=True,
        enable_async_prefetching=True,
    ),
    "mt-apc-only-mt": ServerConfig(
        enable_prefix_caching=True,
        enable_multi_tier_prefix_caching=True,
        enable_async_swapping=False,
        enable_prefix_aware_scheduling=False,
        enable_async_prefetching=False,
    ),
    "mt-apc-only-async": ServerConfig(
        enable_prefix_caching=True,
        enable_multi_tier_prefix_caching=True,
        enable_async_swapping=True,
        enable_prefix_aware_scheduling=False,
        enable_async_prefetching=False,
    ),
    "mt-apc-only-sched": ServerConfig(
        enable_prefix_caching=True,
        enable_multi_tier_prefix_caching=True,
        enable_async_swapping=False,
        enable_prefix_aware_scheduling=True,
        enable_async_prefetching=False,
    ),
    "mt-apc-no-prefetch": ServerConfig(
        enable_prefix_caching=True,
        enable_multi_tier_prefix_caching=True,
        enable_async_swapping=True,
        enable_prefix_aware_scheduling=True,
        enable_async_prefetching=False,
    ),
    "mt-apc-no-sched": ServerConfig(
        enable_prefix_caching=True,
        enable_multi_tier_prefix_caching=True,
        enable_async_swapping=True,
        enable_prefix_aware_scheduling=False,
        enable_async_prefetching=True,
    ),
}


def start_llm_server(args: Namespace, config: ServerConfig, sema: Semaphore) -> None:
    vllm_logger = logging.getLogger("vllm")
    handler = vllm_logger.handlers[0]
    vllm_logger.removeHandler(handler)
    if config.log_file is not None:
        vllm_logger.addHandler(logging.FileHandler(config.log_file))
    start_server(args, config, sema)


class BenchmarkRunner:
    benchmark_dict: Dict[str, Tuple[BENCHMARK_FUNC_TYPE, BENCHMARK_CONFIG_TYPE]] = {
        "generative_agents": (
            generative_agents.benchmark,
            generative_agents.Config,
        ),
        "multiturn_long": (
            multiturn_long.benchmark,
            multiturn_long.Config,
        ),
        "multiturn_short": (
            multiturn_short.benchmark,
            multiturn_short.Config,
        ),
        "sharegpt": (
            sharegpt.benchmark,
            sharegpt.Config,
        ),
        "guessing_game": (
            guessing_game.benchmark,
            guessing_game.Config,
        ),
        "gptswarm_mmlu": (
            gptswarm_mmlu.benchmark,
            gptswarm_mmlu.Config,
        ),
        "guessing_game_e2e": (
            guessing_game_e2e.benchmark,
            guessing_game_e2e.Config,
        ),
        "guessing_game_e2e_cot": (
            guessing_game_e2e_cot.benchmark,
            guessing_game_e2e_cot.Config,
        ),
    }

    def __init__(
        self,
        benchmark: str,
        server_configs: Mapping[str, ServerConfig],
        num_trials: int = 1,
        result_dir: Optional[Path] = None,
    ):
        self._benchmark = benchmark
        self._server_configs = server_configs
        self._num_trials = num_trials
        self._result_dir = result_dir

    def run_all(self):
        for server_name in self._server_configs:
            for i in range(self._num_trials):
                result_file = None
                if self._result_dir is not None:
                    result_file = (
                        self._result_dir
                        / f"{self._benchmark}_{server_name}_{i + 1}.json"
                    )
                logger.info(
                    "Benchmark: %s, Method: %s, Iteration: %d",
                    self._benchmark,
                    server_name,
                    i + 1,
                )
                self.run(server_name, result_file)

    def run(
        self,
        server_name: str,
        result_file: Optional[Path] = None,
    ):
        benchmark_func, benchmark_config_cls = self.benchmark_dict[self._benchmark]
        benchmark_config = benchmark_config_cls(result_file=result_file)
        server_config = benchmark_config.overwrite(self._server_configs[server_name])

        args = server_config.parse_args([])
        server_sema = mp.Semaphore(0)
        server_proc = Process(
            target=start_llm_server, args=(args, server_config, server_sema)
        )

        server_proc.start()
        server_sema.acquire()

        benchmark_func(server_config, benchmark_config)

        server_proc.terminate()
        server_proc.join()


def run_benchmark_suite(
    benchmarks: Optional[List[str]],
    server_names: Optional[List[str]],
    num_trials: int = 1,
    result_dir: Optional[Path] = None,
    clear_result_dir: bool = False,
):
    if benchmarks is None:
        benchmarks = list(BenchmarkRunner.benchmark_dict)

    if server_names is None:
        server_configs = SERVER_CONFIGS
    else:
        server_configs = {name: SERVER_CONFIGS[name] for name in server_names}

    if result_dir is not None:
        if result_dir.exists():
            logger.warning("Result directory found: %s", result_dir)
            if clear_result_dir:
                logger.warning("Clearing the result directory.")
                shutil.rmtree(result_dir)
        result_dir.mkdir(exist_ok=False)

    for benchmark in benchmarks:
        logger.info("Running Benchmark: %s", benchmark)
        runner = BenchmarkRunner(
            benchmark, server_configs, num_trials=num_trials, result_dir=result_dir
        )
        runner.run_all()


def read_result_files(result_file: Path) -> pd.DataFrame:
    data: Dict[str, List[Any]] = {}
    with open(result_file) as f:
        if result_file.suffix == ".jsonl":
            for line in f:
                line_data = json.loads(line)
                for k, v in line_data:
                    data[k] = data.get(k, []) + [v]
        else:
            data = json.load(f)
            data = {k: [v] for k, v in data.items()}
    new_data: Dict[str, List[Any]] = {}
    for k, v in data.items():
        if any(isinstance(v_, dict) for v_ in v):
            for v_ in v:
                for k2, v2 in v_.items():
                    new_data[f"{k}_{k2}"] = new_data.get(f"{k}_{k2}", []) + [v2]
        else:
            new_data[k] = v
    return pd.DataFrame(new_data)


def clean_result_files(result_dir: Path) -> None:
    INDEX_COLS = ["method", "iteration"]

    files = result_dir.glob("*.json*")

    (result_dir / "raw").mkdir(exist_ok=True)
    (result_dir / "cleaned").mkdir(exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    for fname in files:
        bench_name, method, iteration = fname.stem.rsplit("_", 2)
        df: pd.DataFrame = results.pop(bench_name, pd.DataFrame())
        new_entry = read_result_files(fname)
        new_entry.insert(0, INDEX_COLS[1], iteration)
        new_entry.insert(0, INDEX_COLS[0], method)
        results[bench_name] = pd.concat([df, new_entry])
        fname.rename(result_dir / "raw" / fname.name)

    for bench_name, df in results.items():
        df.set_index(["method", "iteration"], inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(result_dir / "cleaned" / f"{bench_name}.csv")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="List of benchmarks to run. Default: all benchmarks",
    )
    parser.add_argument(
        "--servers",
        type=str,
        nargs="+",
        help="List of server configurations to run. Default: all server configurations",
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials to run for each benchmark and server configuration",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Directory to store results. Default: no results saved",
    )
    parser.add_argument(
        "--clear-result-dir",
        action="store_true",
        help="Clear result directory if it exists",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Arguments: %s", args)
    run_benchmark_suite(
        args.benchmarks,
        args.servers,
        num_trials=args.num_trials,
        result_dir=args.result_dir,
        clear_result_dir=args.clear_result_dir,
    )
    if args.result_dir is not None:
        clean_result_files(args.result_dir)


if __name__ == "__main__":
    main()
