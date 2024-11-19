from ..config import BaseClientConfig  # Must be imported before any other modules

import json
from pathlib import Path

from swarm.environment.domain.crosswords.evaluator import CrosswordsEvaluator
from swarm.graph.swarm import Swarm
from swarm.llm import OPENAI_MODEL_PREFIX
from swarm.optimizer.edge_optimizer.optimization import optimize


GPTSWARM_ROOT_DIR = Path.cwd() / "GPTSwarm"
CROSSWORDS_FILE_PATH = GPTSWARM_ROOT_DIR / "datasets/crosswords/mini0505_0_100_5.json"


def main():
    experiment_id = f"experiment{0}"
    config = BaseClientConfig()

    with open(CROSSWORDS_FILE_PATH) as file:
        test_data = json.load(file)

    init_connection_probability = 0.1
    batch_size = 20
    use_learned_order = False
    include_inner_agent_connections = True
    connect_output_nodes_to_final_node = True
    window_size = 10
    evaluator = CrosswordsEvaluator(
        test_data,
        batch_size=batch_size,
        metric="words",
        window_size=window_size,
        init_socre=0.4,
        use_init_score=True,
    )
    swarm = Swarm(
        ["CrosswordsReflection", "CrosswordsToT", "CrosswordsBruteForceOpt"],
        "crosswords",
        OPENAI_MODEL_PREFIX + config.model,
        final_node_class="ReturnAll",
        final_node_kwargs={},
        edge_optimize=True,
        init_connection_probability=init_connection_probability,
        connect_output_nodes_to_final_node=connect_output_nodes_to_final_node,
        include_inner_agent_connections=include_inner_agent_connections,
    )
    optimize(
        swarm,
        evaluator,
        batch_size=batch_size,
        num_iter=11,
        display_freq=1,
        record=False,
        experiment_id=experiment_id,
        lr=0.4,
        use_learned_order=use_learned_order,
    )


if __name__ == "__main__":
    main()
