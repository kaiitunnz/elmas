from ..config import ClientConfig

from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.graph.node import Node
from swarm.llm import LLMRegistry, OPENAI_MODEL_PREFIX
from swarm.llm.format import Message


class ParticipantStep(Node):
    _memory_initialized: bool = False

    def __init__(
        self,
        domain: str,
        step: int,
        model_name: str,
        graph_id: str,
        operation_description: str = "Generation step of a participant agent",
        id: Optional[str] = None,
    ):
        super().__init__(operation_description, id, False)
        self.domain = domain
        self.step = step
        self.model_name = model_name
        self.graph_id = graph_id
        self.llm = LLMRegistry.get(model_name)
        self.config = ClientConfig(model=model_name).load_json(
            str(Path(__file__).parent / "model_config.json")
        )

    @property
    def node_name(self):
        return self.__class__.__name__

    def _get_messages(self) -> List[Message]:
        previous_outputs = self.memory.query_by_id(self.graph_id)
        messages = [output["output"] for output in previous_outputs]
        return messages

    async def _execute(
        self, input: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        inputs = self.process_input(input)
        outputs = []
        for inp in inputs:
            if "output" in inp:
                self.memory.add(self.graph_id, inp)
            messages = self._get_messages()
            output = await self.llm.agen(messages, **self.config.to_kwargs())
            assert isinstance(output, str)
            output_msg = Message(role="assistant", content=output)
            output_dict = {
                "operation": self.node_name,
                "output": output_msg,
            }
            outputs.append(output_dict)
        return outputs
