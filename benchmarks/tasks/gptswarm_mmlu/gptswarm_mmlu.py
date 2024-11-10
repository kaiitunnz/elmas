from agents.config import BaseClientConfig
from tasks.guessing_game.benchmark_llm import BenchmarkLLM

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, List, Optional

from swarm.environment.operations.final_decision import MergingStrategy
from swarm.graph.swarm import Swarm
from swarm.graph.graph import Graph
from swarm.llm import BENCHMARK_MODEL_PREFIX
from swarm.utils.log import logger

from tasks.base import VLLMConfigBase
from tasks.gptswarm_mmlu.mmlu_dataset import MMLUDataset
from utils import utils
from utils.benchmarker import Benchmarker


@dataclass
class Config(VLLMConfigBase):
    mode: str = "full_connected_swarm"
    num_truthful_agents: int = 400
    domain: str = "mmlu"
    eval_batch_size: int = 1
    num_questions: Optional[int] = 20
    dataset_dir: Path = Path(__file__).resolve().parent / "data"
    dataset_split: str = "val"

    num_gpu_blocks_override: Optional[int] = None
    num_cpu_blocks_override: Optional[int] = None
    swap_space: int = 16
    max_model_len: Optional[int] = None
    num_thoughts: Optional[int] = None

    scheduler_window_size: Optional[int] = 10

    def get_num_requests(self, dataset: MMLUDataset) -> int:
        num_questions = (
            min(len(dataset), self.num_questions)
            if self.num_questions is not None
            else len(dataset)
        )
        # 1 truthful agent + 1 adversarial agent per question
        return num_questions * self.num_truthful_agents * 2 * (self.num_thoughts or 1)

    def overwrite(self, server_config: BaseClientConfig) -> BaseClientConfig:
        return replace(
            server_config,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            num_cpu_blocks_override=self.num_cpu_blocks_override,
            swap_space=self.swap_space,
            max_model_len=self.max_model_len,
            scheduler_window_size=self.scheduler_window_size,
        )  # type: ignore


async def _run_swarm(
    swarm: Swarm,
    dataset: MMLUDataset,
    mode: str,
    eval_batch_size: int,
    num_questions: Optional[int] = None,
):
    realized_graph: Graph
    if mode == "full_connected_swarm":
        realized_graph = swarm.connection_dist.realize_full(swarm.composite_graph)
    else:
        raise NotImplementedError()

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if num_questions is not None:
                if i_record >= num_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return

    for record_batch in eval_loader(batch_size=eval_batch_size):
        future_answers = []
        for record in record_batch:
            input_dict = dataset.record_to_swarm_input(record)

            future_answer = swarm.arun(input_dict, realized_graph)
            future_answers.append(future_answer)

        _ = await asyncio.gather(*future_answers)


async def _benchmark(server_config: BaseClientConfig, benchmark_config: Config) -> None:
    benchmarker = Benchmarker(
        server_config, benchmark_config.disabled_pbar, seed=benchmark_config.seed
    )
    BenchmarkLLM.configure(benchmarker)

    n = benchmark_config.num_truthful_agents
    agent_name_list = ["COT" if benchmark_config.num_thoughts else "IO"] * n
    agent_name_list += ["AdversarialAgent"] * n
    swarm = Swarm(
        agent_names=agent_name_list,
        domain=benchmark_config.domain,
        model_name=BENCHMARK_MODEL_PREFIX + server_config.model,
        final_node_class="FinalDecision",
        final_node_kwargs=dict(strategy=MergingStrategy.MajorityVote),
        edge_optimize=False,
    )
    dataset = MMLUDataset(benchmark_config.dataset_dir, benchmark_config.dataset_split)
    benchmarker.set_num_requests(benchmark_config.get_num_requests(dataset))

    with benchmarker:
        await _run_swarm(
            swarm,
            dataset,
            benchmark_config.mode,
            benchmark_config.eval_batch_size,
            benchmark_config.num_questions,
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
    logger.remove()
    benchmark_config = benchmark_config or Config()
    utils.set_seed(benchmark_config.seed)
    asyncio.run(_benchmark(server_config, benchmark_config))


if __name__ == "__main__":
    config = BaseClientConfig()
    benchmark(config)
