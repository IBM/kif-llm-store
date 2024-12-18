# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum, auto

DEFAULT_SYSTEM_INSTRUCTION_PROMPT = '''\
You are a helpful assistant that translate question templates into natural \
language question.'''

DEFAULT_EXAMPLE_TEMPLATE = '''\
Question template:
{query}

Natural Language Question:
{question}'''

DEFAULT_EXAMPLES_PROMPT = '''\
Question template:
Fill in the gap to complete the relation:
Brazil official language _

Natural Language Question:
What are the official languages of Brazil?


Question template:
Fill in the gap to complete the relation:
_ inventor Michael Faraday

Natural Language Question:
What were invented by Michael Faraday?


Question template:
Fill in the gap to complete the relation:
_ spouse Laura Bush

Natural Language Question:
Laura Bush is the spouse of whom?


Question template:
Fill in the gap to complete the relation:
X official language Portuguese
where
X instance of country

Natural Language Question:
Which countries have Portuguese as their official language?


Question template:
Fill in the gap to complete the relation:
Argentina shares border with X
where
X instance of country

Natural Language Question:
Which countries share a border with Argentina?


Question template:
Fill in the gap to complete the relation:
The Beatles has part(s) X
where
X instance of person

Natural Language Question:
Who are the member of The Beatles?


Question template:
Fill in the gap to complete the relation:
France population X
where
X is a number

Natural Language Question:
How big is the population of France?'''


class LLM_Providers(StrEnum):
    IBM = auto()
    OPEN_AI = auto()
    OLLAMA = auto()
