"""
Adapted from https://github.com/sgl-project/sglang.
"""

import random
import string
from typing import Any, Dict, List

from vllm.transformers_utils.tokenizer import AnyTokenizer


def gen_prompt(tokenizer: AnyTokenizer, token_num: int) -> str:
    cha_set = string.ascii_letters + string.digits
    ret = "".join(random.choices(cha_set, k=token_num))
    while len(tokenizer(ret).input_ids) < token_num:
        ret += random.choice(cha_set)
    return ret


def gen_arguments(
    turns: int,
    num_qa: int,
    min_len_q: int,
    max_len_q: int,
    min_len_a: int,
    max_len_a: int,
    tokenizer: AnyTokenizer,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    multi_qas: List[Dict[str, List[Dict[str, Any]]]] = [
        {"qas": []} for _ in range(num_qa)
    ]
    for i in range(num_qa):
        qas = multi_qas[i]["qas"]
        for _ in range(turns):
            prompt_len = random.randint(min_len_q, max_len_q)
            new_tokens = random.randint(min_len_a, max_len_a)
            qas.append(
                {
                    "prompt": gen_prompt(tokenizer, prompt_len),
                    "new_tokens": new_tokens,
                }
            )

    return multi_qas
