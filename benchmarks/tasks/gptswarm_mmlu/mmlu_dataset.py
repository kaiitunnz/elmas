"""
Adapted from https://github.com/metauto-ai/GPTSwarm.
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import pandas as pd

SwarmInput = Dict[str, Any]


class MMLUDataset:
    def __init__(
        self,
        data_dir: Path,
        split: Union[Literal["dev"], Literal["val"], Literal["test"]],
    ) -> None:
        self._split = split
        self._total_df: pd.DataFrame = self._load_data(data_dir / split)

    @staticmethod
    def _load_data(
        data_path: Path,
    ) -> pd.DataFrame:
        csv_paths = sorted(data_path.glob("*.csv"))

        names = ["question", "A", "B", "C", "D", "correct_answer"]

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None, names=names)
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        index = total_df.index.tolist()
        random.shuffle(index)
        total_df = total_df.reindex(index)

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> Union[pd.DataFrame, pd.Series]:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_swarm_input(record: pd.DataFrame) -> SwarmInput:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
        )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            answer = answer[0]  # Try to format the answer by taking the first letter
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record["correct_answer"]
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)"
            f" record={record}"
        )
        return correct_answer
