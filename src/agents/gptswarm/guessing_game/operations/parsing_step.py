from ..config import ClientConfig

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.graph.node import Node
from swarm.llm import LLMRegistry
from swarm.llm.format import Message

from ..prompt.prompt_set import PromptSet


class ParsingStep(Node):
    def __init__(
        self,
        domain: str,
        model_name: str,
        graph_id: str,
        operation_description: str = "Parsing step of a participant agent",
        id: Optional[str] = None,
    ):
        super().__init__(operation_description, id, False)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.graph_id = graph_id
        self.prompt_set = PromptSet()
        self.usr_message = Message(
            role="user",
            content=self.prompt_set.get_parser_user_prompt(),
        )
        self.config = ClientConfig(model=model_name).load_json(
            str(Path(__file__).parent / "model_config.json")
        )

    def _get_message(self, msg: Message) -> List[Message]:
        messages = [
            Message(
                role="system",
                content=self.prompt_set.get_parser_system_prompt(msg.content),
            ),
            self.usr_message,
        ]
        return messages

    async def _execute(self, input, **kwargs) -> Dict[str, Any]:
        inputs = self.process_input(input)
        assert len(inputs) == 1

        input = inputs[0]
        input_msg = input["output"]
        messages = self._get_message(input_msg)
        output = await self.llm.agen(messages, **self.config.to_kwargs())
        assert isinstance(output, str)
        numbers = re.findall(r"(\d+(\.\d+)?)", output)
        output_number = -1 if len(numbers) == 0 else float(numbers[0][0])

        self.memory.add("parser", {"input": messages, "output": output})
        output_dict = {
            "operation": self.node_name,
            "output": input_msg,
            "value": output_number,
        }
        self.memory.add(self.graph_id, output_dict)
        return output_dict
