from typing import List

from swarm.graph.graph import Graph

from ..operations.coordinator_step import CoordinatorStep
from .participant import Participant


class Coordinator(Graph):
    def __init__(
        self,
        domain: str,
        num_steps: int,
        model_name: str,
        max_value: int = 100,
        ratio: str = "2/3",
        meta_prompt: bool = False,
        async_exec: bool = True,
    ):
        self._num_steps = num_steps
        self._max_value = max_value
        self._ratio = ratio

        super().__init__(domain, model_name, meta_prompt, async_exec)

    def build_graph(self):
        for i in range(self._num_steps):
            coordinator_step = CoordinatorStep(
                domain=self.domain,
                max_value=self._max_value,
                ratio=self._ratio,
                graph_id=self.id,
                is_last_step=i == self._num_steps - 1,
            )
            self.add_node(coordinator_step)
        self.input_nodes = []
        self.output_nodes = [coordinator_step]

    def add_participant(self, participant: Participant) -> None:
        assert (
            len(participant.parsing_steps)
            == len(participant.participant_steps)
            == self._num_steps
        )

        coordinator_steps: List[CoordinatorStep] = list(self.nodes.values())
        for parsing_step, participant_step, coordinator_step in zip(
            participant.parsing_steps,
            participant.participant_steps[1:],
            coordinator_steps,
        ):
            coordinator_step.add_predecessor(parsing_step)
            coordinator_step.add_successor(participant_step)
        coordinator_steps[-1].add_predecessor(participant.parsing_steps[-1])
