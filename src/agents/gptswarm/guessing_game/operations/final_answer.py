from typing import Any, Dict, List, Optional

from swarm.graph.node import Node


class FinalAnswer(Node):
    def __init__(
        self,
        domain: str,
        max_value: int,
        operation_description: str = "Output the final answer",
        id: Optional[str] = None,
    ):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self._max_value = max_value

    @property
    def node_name(self):
        return self.__class__.__name__

    async def _execute(self, input: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        inputs = self.process_input(input)
        assert len(inputs) == 1
        output = {"operation": self.node_name, "output": inputs[0]["output"]}
        return output
