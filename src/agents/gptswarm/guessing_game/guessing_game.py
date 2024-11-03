from .config import ClientConfig

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
import argparse
import shortuuid
from swarm.graph.composite_graph import CompositeGraph
from swarm.graph.graph import Graph
from swarm.graph.swarm import Swarm
from swarm.llm import OPENAI_MODEL_PREFIX
from swarm.memory import GlobalMemory
from swarm.optimizer.edge_optimizer.parameterization import EdgeWiseDistribution
from swarm.utils.log import logger

from .operations.final_answer import FinalAnswer
from .prompt.prompt_set import DEFAULT_PROMPT_PATH
from .graph.coordinator import Coordinator
from .graph.participant import Participant


class GuessTwoThirdGame(Swarm):
    @dataclass
    class Output:
        round_winners: List[float]
        participant_answers: List[List[float]]

        @property
        def winner(self) -> float:
            return self.round_winners[-1]

    def __init__(
        self,
        model_name: str,
        num_participants: int,
        num_steps: int,
        prompt_path: Optional[str] = None,
        max_value: int = 100,
        ratio: str = "2/3",
        domain: str = "GuessTwoThirdGame",
        async_exec: bool = True,
    ):
        self.id = shortuuid.ShortUUID().random(length=4)
        self.agent_names = []
        self.domain = domain
        self.model_name = model_name
        self.open_graph_as_html = False
        self.memory = GlobalMemory.instance()
        self.final_node_class = "UNUSED"
        self.final_node_kwargs: Dict[str, Any] = {}
        self.edge_optimize = False
        self.node_optimize = False
        self.init_connection_probability = 0.5
        self.connect_output_nodes_to_final_node = False

        self.async_exec = async_exec

        self.num_participants = num_participants
        self.num_steps = num_steps
        self.prompt_path = prompt_path
        self.max_value = max_value
        self.ratio = ratio

        self.organize()

    def organize(self, include_inner_agent_connections: bool = True):
        assert isinstance(self.model_name, str)

        self.used_agents: List[Graph] = []
        decision_method = FinalAnswer(domain=self.domain, max_value=self.max_value)
        self.composite_graph = CompositeGraph(
            decision_method, domain=self.domain, model_name=self.model_name
        )

        self.coordinator = Coordinator(
            domain=self.domain,
            num_steps=self.num_steps,
            model_name=self.model_name,
            max_value=self.max_value,
            ratio=self.ratio,
            async_exec=self.async_exec,
        )
        self.composite_graph.add_graph(self.coordinator)
        self.used_agents.append(self.coordinator)

        self.participants: List[Graph] = []
        for i in range(self.num_participants):
            participant = Participant(
                domain=self.domain,
                num_steps=self.num_steps,
                model_name=self.model_name,
                prompt_path=self.prompt_path,
                participant_id=i,
                max_value=self.max_value,
                ratio=self.ratio,
                async_exec=self.async_exec,
            )
            self.composite_graph.add_graph(participant)
            self.used_agents.append(participant)
            self.participants.append(participant)

            self.coordinator.add_participant(participant)

        self.coordinator.output_nodes[0].add_successor(decision_method)

        self.connection_dist = EdgeWiseDistribution(
            [], self.init_connection_probability
        )

    def get_output(self) -> "GuessTwoThirdGame.Output":
        round_winners = [
            output["winner"] for output in self.memory.query_by_id(self.coordinator.id)
        ]
        participant_answers = [
            [
                output["value"]
                for output in self.memory.query_by_id(participant.id)
                if "value" in output
            ]
            for participant in self.participants
        ]
        return GuessTwoThirdGame.Output(
            round_winners=round_winners, participant_answers=participant_answers
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-participants", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt-path", type=str, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--ratio", type=str, default="2/3")
    parser.add_argument("--dump-file", type=str, default=None)
    return parser.parse_args()


def _check_dump_file(dump_file: str) -> None:
    dump_fpath = Path(dump_file)
    if not dump_fpath.parent.exists():
        raise FileNotFoundError(f"Parent directory of {dump_fpath} does not exist.")
    if dump_fpath.exists():
        raise FileExistsError(f"{dump_fpath} already exists.")


async def run(
    num_participants: int,
    num_steps: int,
    model_name: Optional[str] = None,
    prompt_path: Optional[str] = None,
    max_value: int = 100,
    ratio: str = "2/3",
    dump_file: Optional[str] = None,
    async_exec: bool = True,
) -> GuessTwoThirdGame.Output:
    if dump_file is not None:
        _check_dump_file(dump_file)

    config = ClientConfig()
    if model_name is None:
        model_name = OPENAI_MODEL_PREFIX + config.model
    swarm = GuessTwoThirdGame(
        model_name=model_name,
        num_participants=num_participants,
        num_steps=num_steps,
        prompt_path=prompt_path,
        max_value=max_value,
        ratio=ratio,
        async_exec=async_exec,
    )
    _ = await swarm.arun({"task": "unused"})

    if dump_file is not None:
        with open(dump_file, "w") as f:
            f.write(repr(swarm.memory))
        logger.info(f'Memory dumped to "{dump_file}".')

    output = swarm.get_output()
    return output


async def main(
    num_participants: int,
    num_steps: int,
    model_name: Optional[str] = None,
    prompt_path: Optional[str] = None,
    max_value: int = 100,
    ratio: str = "2/3",
    dump_file: Optional[str] = None,
    async_exec: bool = True,
):
    output = await run(
        num_participants=num_participants,
        num_steps=num_steps,
        model_name=model_name,
        prompt_path=prompt_path,
        max_value=max_value,
        ratio=ratio,
        dump_file=dump_file,
        async_exec=async_exec,
    )
    logger.info(f"Final output: {output.winner}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            num_participants=args.num_participants,
            num_steps=args.num_steps,
            model_name=args.model,
            prompt_path=args.prompt_path,
            max_value=args.max_value,
            ratio=args.ratio,
            dump_file=args.dump_file,
        )
    )
