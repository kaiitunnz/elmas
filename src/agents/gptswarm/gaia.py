from ..config import BaseProfilingConfig  # Must be imported before any other modules

import time
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple

from swarm.graph.swarm import Swarm
from swarm.llm import OPENAI_MODEL_PREFIX

from ..vllm_utils.profiling import VLLMProfiling


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("-n", type=int, default=3, help="Number of agents in the swarm")
    parser.add_argument(
        "--profiling", action="store_true", default=False, help="Enable profiling"
    )

    return parser.parse_args()


def run(swarm: Swarm, inputs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    start = time.perf_counter()
    outputs = swarm.run(inputs)
    end = time.perf_counter()
    return outputs, {"elapsed_time": end - start}


def run_profile(
    swarm: Swarm, config: BaseProfilingConfig, inputs: Dict[str, Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    profiling = VLLMProfiling(config)
    profiling.start()
    results = run(swarm, inputs)
    profiling.stop()
    return results


def main():
    args = parse_args()

    config = BaseProfilingConfig()
    swarm = Swarm(
        ["IO"] * args.n, "gaia", model_name=OPENAI_MODEL_PREFIX + config.model
    )
    task = "What is the capital of Jordan?"
    inputs = {"task": task}

    print(f"Running a swarm of {args.n} agents...")
    if args.profiling:
        answer, metrics = run_profile(swarm, config, inputs)
    else:
        answer, metrics = run(swarm, inputs)
    print("Answer:", answer)
    print(f"Elapsed time:{metrics['elapsed_time']:.2f} seconds")


if __name__ == "__main__":
    main()
