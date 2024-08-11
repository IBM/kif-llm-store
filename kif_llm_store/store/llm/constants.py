# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum

SIBLINGS_QUERY_SURFACE_FORM = """SELECT distinct ?sibling ?siblingLabel WHERE
{
  {} wdt:P31 ?o .
  ?sibling wdt:P31 ?o
  FILTER (?sibling != {})
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE], en".
  }
}"""

LABEL_QUERY_SURFACE_FORM = """SELECT ?eLabel
WHERE {
  VALUES ?e {{}}
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "{}" .
  }
}
"""

WIKIDATA_SPARQL_ENDPOINT_URL = os.getenv(
    'WIKIDATA_SPARQL_ENDPOINT_URL',
    'https://query.wikidata.org/sparql',
)

WIKIDATA_SEARCH_API_BASE_URL = os.getenv(
    'WIKIDATA_SEARCH_API_BASE_URL', 'https://www.wikidata.org/w/api.php'
)

WIKIDATA_REST_API_BASE_URL = os.getenv(
    'WIKIDATA_REST_API_BASE_URL',
    'https://www.wikidata.org/w/rest.php',
)


user_prompt = '\n\nTASK:\n"{task}"'

default_output = (
    '\n\nThe output should be only a '
    'list containing the answers, such as ["answer_1", '
    '"answer_2", ..., "answer_n"]. Do not provide '
    'any further explanation and avoid false answers. '
    'Return an empty list, such as [], if no information '
    'is available.'
)
DEFAULT_PROMPT_TEMPLATE = {
    'system': (
        'You are a helpful and honest assistant that resolves a TASK. '
        'Please, respond concisely, with no further explanation, and '
        'truthfully.'
    ),
    'user': user_prompt + default_output,
}

DEFAULT_SUPPORT_CONTEXT = {
    'system': (
        'You are a helpful and honest assistant that resolves a TASK. '
        'Use the CONTEXT to support the answer. Please, respond concisely, '
        'with no further explanation, and truthfully.'
    ),
    'user': user_prompt + '\n\nCONTEXT:\n{context}' + default_output,
}
DEFAULT_ENFORCED_CONTEXT = {
    'system': (
        'You are a helpful and honest assistant that resolves a TASK '
        'based on the CONTEXT. Only perfect and explicit matches '
        'mentioned in CONTEXT are accepted. Please, respond concisely, '
        'with no further explanation, and truthfully.'
    ),
    'user': user_prompt + '\n\nCONTEXT:\n{context}' + default_output,
}


class Prompt_Type(Enum):
    TRIPLE = 'triple'
    QUESTION = 'question'


class Disambiguation_Method(Enum):
    KEYWORD = 'keyword'
    BASELINE = 'baseline'
    LLM = 'llm'
    SIM_IN_CONTEXT = 'SIM_IN_CONTEXT'


class LLM_Models(Enum):
    BAM = 'bam'
    GPT = 'gpt'
    HUGGING_FACE_HUB = 'hf'
    GENERIC = 'generic'


class ChatRole(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class LLM_Model_Type(Enum):
    INSTRUCT = 'instruct'
    CHAT = 'chat'
    GENERAL = 'general'
