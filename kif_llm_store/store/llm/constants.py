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
You are a helpful and honest assistant that resolves a given TASK. \
Please, respond concisely, with no further explanation, and truthfully.'''

SYSTEM_PROMPT_INSTRUCTION_WITH_CONTEXT = '''\
You are a helpful and honest assistant that resolves a TASK using \
a given CONTEXT. Please, respond concisely, with no further explanation, and \
truthfully.'''

SYSTEM_PROMPT_INSTRUCTION_WITH_ENFORCED_CONTEXT = '''\
You are a helpful and honest assistant that resolves a TASK based on \
given CONTEXT. Only perfect and explicit matches mentioned in CONTEXT are \
accepted. Please, respond concisely, with no further explanation, and \
truthfully.'''


ONE_VARIABLE_PROMPT_TASK = '''\
Replace the variable with all possible values that can complete the relation:
'''


class Disambiguation_Method(StrEnum):
    KEYWORD = auto()
    BASELINE = auto()
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
