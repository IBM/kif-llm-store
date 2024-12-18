# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging

from typing import Any, Optional, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableLambda

from .constants import (
    DEFAULT_EXAMPLES_PROMPT,
    DEFAULT_SYSTEM_INSTRUCTION_PROMPT,
    DEFAULT_EXAMPLE_TEMPLATE,
    LLM_Providers,
)

LOG = logging.getLogger(__name__)


class Example:
    query: str
    question: str

    def __init__(self, query: str, question: str):
        self.query = query
        self.question = question

    def to_prompt(self) -> str:
        return DEFAULT_EXAMPLE_TEMPLATE.format(
            query=self.query, question=self.question
        )


class QueryToQuestion:
    _system_prompt_template: str
    _examples: list[Example]
    _examples_prompt: str
    _model: BaseChatModel

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        llm_provider: Optional[LLM_Providers] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_params: Optional[dict[str, Any]] = None,
        system_prompt_template: Optional[str] = None,
        examples: Optional[
            Union[list[Example], tuple[str, str], list[tuple[str, str]]]
        ] = None,
    ):
        self._init_model(
            model, llm_provider, model_id, base_url, api_key, model_params
        )

        self._init_examples_prompt(examples)

        self._init_prompt_template(system_prompt_template)

    def _init_model(
        self,
        model: Optional[BaseChatModel] = None,
        llm_provider: Optional[LLM_Providers] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_params: Optional[dict[str, Any]] = None,
    ):
        if model:
            self._model = model
        else:
            assert llm_provider, "No LLM provider was set."
            assert llm_provider in LLM_Providers, "Invalid LLM provider."
            assert model_id, "No model identifier was set."

            llm_params: dict[str, Any] = {}

            if base_url:
                llm_params['base_url'] = base_url
            if api_key:
                llm_params['api_key'] = api_key

            try:
                if llm_provider == LLM_Providers.OPEN_AI:
                    from langchain_openai import ChatOpenAI

                    self._model = ChatOpenAI(
                        model=model_id, **{**llm_params, **model_params}
                    )
                elif llm_provider == LLM_Providers.OLLAMA:
                    from langchain_ollama import ChatOllama

                    self._model = ChatOllama(
                        model=model_id, **{**llm_params, **model_params}
                    )
                elif llm_provider == LLM_Providers.IBM:
                    from langchain_ibm import ChatWatsonx

                    self._model = ChatWatsonx(
                        model_id=model_id,
                        apikey=api_key,
                        url=base_url,
                        params=model_params,
                    )
                else:
                    raise ValueError(f"Unsupported provider: {llm_provider}")
            except Exception as e:
                raise e

    def _init_prompt_template(
        self,
        system_prompt_template: Optional[str] = None,
    ):

        if not system_prompt_template:
            system_prompt_template = DEFAULT_SYSTEM_INSTRUCTION_PROMPT

        if self._examples_prompt:
            system_prompt_template += f'\nExamples:\n{self._examples_prompt}'

        self._system_prompt_template = system_prompt_template

    def _init_examples_prompt(
        self,
        examples: Optional[
            Union[list[Example], tuple[str, str], list[tuple[str, str]]]
        ] = None,
    ):
        if not examples:
            self._examples_prompt = DEFAULT_EXAMPLES_PROMPT

        if isinstance(examples, tuple):
            query, question = examples
            example = Example(query, question)

            self._examples_prompt = example.to_prompt()
        elif isinstance(examples, list):

            if isinstance(examples[0], Example):
                self._examples_prompt = f'{examples[0].to_prompt()}'
                for example in examples[1:]:
                    self._examples_prompt += f'\n{example.to_prompt()}'
            elif isinstance(examples[0], tuple):
                query, question = examples[0]
                example = Example(query, question)
                self._examples_prompt = f'{example.to_prompt()}'
                for example in examples[1:]:
                    query, question = example
                    new_example = Example(query, question)
                    self._examples_prompt += f'\n{new_example.to_prompt()}'

    def run(self, query: str) -> ChatPromptTemplate:
        system_prompt_template = SystemMessage(
            content=self._system_prompt_template
        )

        human = '''Question template:
{query}

Natural Language Question:'''

        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt_template,
                ('human', human),
            ]
        )

        debug_chain = RunnableLambda(lambda entry: (LOG.info(entry), entry)[1])

        chain: RunnableSequence = (
            prompt | debug_chain | self._model | StrOutputParser()
        )

        try:
            return chain.invoke({'query': query})
        except Exception as e:
            raise e
