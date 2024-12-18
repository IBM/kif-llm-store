# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import os
from enum import StrEnum, auto

from typing_extensions import TypeAlias

WID: TypeAlias = str
QID: TypeAlias = str
PID: TypeAlias = str
PDF: TypeAlias = str
SITES: TypeAlias = str
Label: TypeAlias = str


WIKIDATA_SPARQL_ENDPOINT_URL = os.getenv(
    'WIKIDATA_SPARQL_ENDPOINT_URL',
    'https://query.wikidata.org/sparql',
)

DBPEDIA_SEARCH_API_BASE_URL = os.getenv(
    'DBPEDIA_SEARCH_API_BASE_URL', 'https://lookup.dbpedia.org/api/search'
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

DEFAULT_AVOID_EXPLANATION_INSTRUCTION = '''\
Please, respond only the noun phrase, truthfully, and avoid any additional \
explanation.
'''

DEFAULT_SYSTEM_PROMPT_INSTRUCTION = '''\
You are a helpful and honest assistant that resolves a human given TASK.'''

SYSTEM_PROMPT_INSTRUCTION_WITH_CONTEXT = '''\
You are a helpful and honest assistant that resolves a TASK using \
a given CONTEXT.'''

SYSTEM_PROMPT_INSTRUCTION_WITH_ENFORCED_CONTEXT = '''\
You are a helpful and honest assistant that resolves a TASK based on \
a given CONTEXT. Only perfect and explicit matches mentioned in CONTEXT are \
accepted.'''


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
