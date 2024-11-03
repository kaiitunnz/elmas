from typing import List, Optional

from swarm.graph.graph import Graph
from swarm.llm.format import Message

from ..operations.parsing_step import ParsingStep
from ..operations.participant_step import ParticipantStep
from ..prompt.prompt_set import PromptSet  # type: ignore


class Participant(Graph):
    def __init__(
        self,
        domain: str,
        num_steps: int,
        model_name: str,
        prompt_path: Optional[str] = None,
        participant_id: Optional[int] = None,
        max_value: int = 100,
        ratio: str = "2/3",
        meta_prompt: bool = False,
        async_exec: bool = True,
    ):
        self._num_steps = num_steps
        self._participant_id = participant_id
        self._max_value = max_value
        self._ratio = ratio
        self._prompt_path = prompt_path

        self._prompt_set = PromptSet(prompt_path)
        self._sys_message = Message(
            role="system",
            content=self._prompt_set.get_formatted_participant_system_prompt(
                max_value=max_value,
                ratio=ratio,
                persona=None if prompt_path is None else participant_id,
            ),
        )
        self._usr_message = Message(
            role="user", content=self._prompt_set.get_participant_user_prompt()
        )

        self.parsing_steps: List[ParsingStep] = []
        self.participant_steps: List[ParticipantStep] = []

        super().__init__(domain, model_name, meta_prompt, async_exec)
        self._init_memory()

    @property
    def graph_name(self):
        return self.__class__.__name__

    def _init_memory(self):
        for msg in [self._sys_message, self._usr_message]:
            self.memory.add(self.id, {"operation": self.graph_name, "output": msg})

    def build_graph(self):
        for i in range(self._num_steps):
            participant_step = ParticipantStep(
                domain=self.domain, step=i, model_name=self.model_name, graph_id=self.id
            )
            parsing_step = ParsingStep(
                domain=self.domain, model_name=self.model_name, graph_id=self.id
            )
            participant_step.add_successor(parsing_step)
            self.add_node(participant_step)
            self.add_node(parsing_step)
            self.participant_steps.append(participant_step)
            self.parsing_steps.append(parsing_step)

        self.input_nodes = self.participant_steps[:1]
        self.output_nodes = []
