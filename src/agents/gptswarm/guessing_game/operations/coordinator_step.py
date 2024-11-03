from typing import Any, Dict, List, Optional

from swarm.graph.node import Node
from swarm.llm.format import Message

from ..prompt.prompt_set import PromptSet

RATIO_MAP = {
    "2/3": 2 / 3,
}


class CoordinatorStep(Node):
    def __init__(
        self,
        domain: str,
        max_value: int,
        ratio: str,
        graph_id: str,
        operation_description: str = "Summarize the results of each round of the game",
        is_last_step: bool = False,
        id: Optional[str] = None,
    ):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self._max_value = max_value
        self._ratio = ratio
        self._graph_id = graph_id
        self._is_last_step = is_last_step
        self._prompt_set = PromptSet()

    async def _execute(self, input: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        inputs: List[Dict[str, Any]] = self.process_input(input)
        sum = 0
        count = 0
        for inp in inputs:
            output = inp["value"]
            if 0 <= output <= self._max_value:
                sum += output
                count += 1

        winner = sum / count * RATIO_MAP[self._ratio]
        output_msg = (
            winner
            if self._is_last_step
            else Message(
                role="user",
                content=self._prompt_set.get_coordinator_output_prompt(winner),
            )
        )
        self.memory.add(self._graph_id, {"operation": self.node_name, "winner": winner})

        output = {"operation": self.node_name, "output": output_msg}
        return output
