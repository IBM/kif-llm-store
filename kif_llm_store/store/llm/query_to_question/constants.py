# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum, auto

DEFAULT_SYSTEM_INSTRUCTION_PROMPT = '''\
You are a helpful assistant that translate question templates into natural \
language question.'''

DEFAULT_EXAMPLES_FOR_Q_2_Q = [
    {
        "query": '''Fill in the gap to complete the relation:
Brazil official language _''',
        "question": "What are the official languages of Brazil?",
    },
    {
        "query": '''Fill in the gap to complete the relation:
_ inventor Michael Faraday''',
        "question": "What were invented by Michael Faraday?",
    },
    {
        "query": '''Fill in the gap to complete the relation:
_ spouse Laura Bush''',
        "question": "Laura Bush is the spouse of whom?",
    },
    {
        "query": '''Fill in the gap to complete the relation:
X official language Portuguese
where
X instance of country''',
        "question": "Which countries have Portuguese as their official language?",
    },
    {
        "query": '''Fill in the gap to complete the relation:
Argentina shares border with X
where
X instance of country''',
        "question": "Which countries share a border with Argentina?",
    },
    {
        "query": '''Fill in the gap to complete the relation:
The Beatles has part(s) X
where
X instance of person''',
        "question": "Who are the member of The Beatles?",
    },
    {
        "query": '''Fill in the gap to complete the relation:
France population X
where
X is a number''',
        "question": "How big is the population of France?",
    },
]


class LLM_Providers(StrEnum):
    IBM = auto()
    OPEN_AI = auto()
    OLLAMA = auto()
