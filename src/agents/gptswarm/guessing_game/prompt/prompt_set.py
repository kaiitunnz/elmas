"""Adapted from https://github.dev/modelscope/agentscope."""

from typing import Dict, List, Optional

from ....utils.utils import read_jsonl


class PromptSet:
    _system_prompts: List[str] = [
        "You are playing a multiplayer game.\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to {ratio} of the average of all reported numbers.\n\n",
        "You are playing a multiplayer game.\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to {ratio} of the average of all reported numbers.\n\n# Note:\n1. All players are rational.\n\n",
        "You are playing a multiplayer game.\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to {ratio} of the average of all reported numbers.\n\n# Note:\n1. All players are rational.\n2. All players will try to guess the others' strategies to adjust their own strategies.\n\n",
        'You are playing a multiplayer game.\n\n# Game Rule\n1. This game is a variation of the famous "guess 2/3 of the average" game\n2. Each player reports a real number between 0 and {max_value}, inclusive.\n3. The winner will be the player whose number is the closest to {ratio} of the average of all reported numbers.\n\n',
        'You are playing a multiplayer game.\n\n# Game Rule\n1. This game is a variation of the famous "guess 2/3 of the average" game\n2. Each player reports a real number between 0 and {max_value}, inclusive.\n3. The winner will be the player whose number is the closest to {ratio} of the average of all reported numbers.\n\n# Note:\n1. All players are rational.\n\n',
        "You are playing a multiplayer game.\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to 5 plus {ratio} of the average of all reported numbers .\n\n",
        "You are playing a multiplayer game.\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to 5 plus {ratio} of the average of all reported numbers .\n\n# Note:\n1. All players are rational.\n\n",
        'You are playing a multiplayer game.\n\n# Game Rule\n1. This game is a variation of the famous "guess 2/3 of the average" game\n2. Each player reports a real number between 0 and {max_value}, inclusive.\n3. The winner will be the player whose number is the closest to 5 plus {ratio} of the average of all reported numbers .\n\n',
        'You are playing a multiplayer game.\n\n# Game Rule\n1. This game is a variation of the famous "guess 2/3 of the average" game\n2. Each player reports a real number between 0 and {max_value}, inclusive.\n3. The winner will be the player whose number is the closest to 5 plus {ratio} of the average of all reported numbers .\n\n# Note:\n1. All players are rational.\n\n',
        "You are playing a role in a multiplayer game, make sure your behavior fits the following character background.\n\n# Character Background\n\n{background}\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to the {ratio} of the average of all reported numbers.\n\n# Note\n1. Please strictly follow your character background in the game.\n\n",
        "You are playing a role in a multiplayer game, make sure your behavior fits the following character background.\n\n# Character background\n\n{background}\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to the {ratio} of the average of all reported numbers.\n\n# Note:\n1. Please strictly follow your character background in the game.\n2. There are a total of 1000 players, with 200 individuals at each education level: Elementary School, High School, Bachelor's Degree, Master's Degree, and Ph.D.\n\n",
        "You are playing a role in a multiplayer game, make sure your behavior fits the following character background.\n\n# Character background\n\n{background}\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to the {ratio} of the average of all reported numbers.\n\n# Note:\n1. Please strictly follow your character background in the game.\n2. There are a total of 1200 players, with 200 individuals in each profession: Writers, Artists, Psychologists, Economists, and Professor of game theory\n\n",
        "You are playing a role in a multiplayer game, make sure your behavior fits the following character background.\n\n# Character background\n\n{background}\n\n# Game Rule\n1. Each player reports a real number between 0 and {max_value}, inclusive.\n2. The winner will be the player whose number is the closest to the {ratio} of the average of all reported numbers.\n\n# Note:\n1. Please strictly follow your character background in the game.\n2. There are a total of 1200 players, with different professions, including Writers, Artists, Psychologists, Economists, and Professors.\n3. Only one player is an expert in the field of game theory (it may be you, please judge for yourself based on your background information)\n\n",
    ]
    _user_prompts: List[str] = [
        "Directly report your number without additional information.",
        "Think step by step and then report your number.",
    ]
    _persona_prompts: Dict[str, List[str]] = {}
    _parser_prompts: List[str] = [
        "You need to extract the number that the speaker wants to answer from the following text.\n",
        "Now please directly give the extracted number in the following format:\nThe answer is [number].\n\nIf you can't extract the number, please reply directly:\nI CAN'T.\n",
    ]
    _coordinator_prompts: str = (
        "The winner number of this round is {winner:.2f}. Let's move on to the next round.\n{usr_prompt}"
    )

    def __init__(self, prompt_path: Optional[str] = None):
        self._prompt_path = prompt_path
        if (prompt_path is not None) and (
            prompt_path not in self.__class__._persona_prompts
        ):
            self.__class__._persona_prompts[prompt_path] = (
                self._normalize_persona_prompts(
                    [prompt["prompt"] for prompt in read_jsonl(prompt_path)]
                )
            )

    def _normalize_persona_prompts(self, persona_prompts: List[str]) -> List[str]:
        ret = []
        for prompt in persona_prompts:
            lines = [line.strip() for line in prompt.strip().split("\n")]
            lines = [(" " * 4 + line) for line in lines if line]
            ret.append("\n".join(lines))
        return ret

    def get_participant_system_prompt(self, idx: int = -1) -> str:
        return self._system_prompts[idx]

    def get_participant_user_prompt(self, idx: int = -1) -> str:
        return self._user_prompts[idx]

    def get_participant_persona_prompt(self, idx: int) -> str:
        if self._persona_prompts is None or self._prompt_path is None:
            raise ValueError("Persona prompts are not loaded.")
        persona_prompts = self._persona_prompts[self._prompt_path]
        return persona_prompts[idx % len(persona_prompts)]

    def get_formatted_participant_system_prompt(
        self,
        idx: int = -1,
        max_value: int = 100,
        ratio: str = "2/3",
        persona: Optional[int] = None,
    ) -> str:
        if persona is None:
            idx = 8
            return self._system_prompts[idx].format(max_value=max_value, ratio=ratio)
        return self._system_prompts[idx].format(
            max_value=max_value,
            ratio=ratio,
            background=self.get_participant_persona_prompt(persona),
        )

    def get_parser_system_prompt(self, text_to_parse: str) -> str:
        return self._parser_prompts[0] + text_to_parse

    def get_parser_user_prompt(self) -> str:
        return self._parser_prompts[1]

    def get_coordinator_output_prompt(
        self, winner: float, usr_prompt_idx: int = -1
    ) -> str:
        return self._coordinator_prompts.format(
            winner=winner, usr_prompt=self._user_prompts[usr_prompt_idx]
        )
