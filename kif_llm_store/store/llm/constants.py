# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import os
from enum import StrEnum, auto

from typing_extensions import TypeAlias

WID: TypeAlias = str
QID: TypeAlias = str
PID: TypeAlias = str
Label: TypeAlias = str


WIKIDATA_SPARQL_ENDPOINT_URL = os.getenv(
    'WIKIDATA_SPARQL_ENDPOINT_URL',
    'https://query.wikidata.org/sparql',
)

WIKIDATA_SEARCH_API_BASE_URL = os.getenv(
    'WIKIDATA_SEARCH_API_BASE_URL', 'https://www.wikidata.org/w/api.php'
)

DEFAULT_WIKIDATA_SEARCH_API_TEMPLATE = (
    f'{WIKIDATA_SEARCH_API_BASE_URL}'
    '?action=wbsearchentities'
    '&search={label}'
    '&language=en'
    '&format=json'
    '&limit={limit}'
    '&type={type}'
)

WIKIDATA_REST_API_BASE_URL = os.getenv(
    'WIKIDATA_REST_API_BASE_URL',
    'https://www.wikidata.org/w/rest.php',
)

DEFAULT_ONE_VARIABLE_PROMPT_OUTPUT = '''\
Return the results as a comma-separated list, or leave it empty if no \
information is available.'''

DEFAULT_GENERIC_PROMPT_OUTPUT = '''\
Return only an object where each variable in the task (starting with `?`) \
should be a key and the responses should be a list of values ​​for each of \
those keys.'''

DEFAULT_SYSTEM_PROMPT_INSTRUCTION = '''\
You are a helpful and honest assistant that resolves a human given TASK. \
Please, respond concisely and truthfully with no further explanation.'''

SYSTEM_PROMPT_INSTRUCTION_WITH_CONTEXT = '''\
You are a helpful and honest assistant that resolves a TASK using \
a given CONTEXT. Please, respond concisely and truthfully with no further \
explanation.'''

SYSTEM_PROMPT_INSTRUCTION_WITH_ENFORCED_CONTEXT = '''\
You are a helpful and honest assistant that resolves a TASK based on \
a given CONTEXT. Only perfect and explicit matches mentioned in CONTEXT are \
accepted. Please, respond concisely and truthfully with no further \
explanation.'''

SYSTEM_PROMPT_INSTRUCTION_FOR_QUERY_TO_QUESTION = '''\
You are a helpful assistant that translate question templates into natural \
language question. Please, respond concisely and truthfully with no further \
explanation.

Examples:
Question template:
Fill in the gap to complete the relation:
Brazil official language _

Natural Language Question:
What is Brazil' official language?

Question template:
Fill in the gap to complete the relation:
Argentina shares border with X
where
X instance of country

Natural Language Question:
Which country shares a border with Argentina?


Question template:
Fill in the gap to complete the relation:
The Beatles has part(s) X
where
X instance of person

Natural Language Question:
Who are the member of The Beatles?

Question template:
Fill in the gap to complete the relation:
Freddie Mercury place of death X
where
X instance of city or
X instance of country

Natural Language Question:
In which city or country did Freddie Mercury die?'''


ONE_VARIABLE_PROMPT_TASK = '''\
Fill in the gap to complete the relation:
'''


class EntityResolutionMethod(StrEnum):
    KEYWORD = auto()
    NAIVE = auto()
    LLM = auto()
    SIM = auto()


class LLM_Providers(StrEnum):
    IBM = auto()
    OPEN_AI = auto()
    HUGGING_FACE_HUB = 'hf'
    OLLAMA = auto()


class KIF_FilterTypes(StrEnum):
    EMPTY = auto()
    ONE_VARIABLE = auto()
    GENERIC = auto()
