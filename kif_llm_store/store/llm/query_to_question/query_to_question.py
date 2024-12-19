# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging

from typing import Any, Optional, TypedDict, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableLambda

from .constants import (
    DEFAULT_EXAMPLES_FOR_Q_2_Q,
    DEFAULT_SYSTEM_INSTRUCTION_PROMPT,
    LLM_Providers,
)

LOG = logging.getLogger(__name__)


class Example(TypedDict):
    query: str
    question: str


class QueryToQuestion:
    _system_prompt_template: str
    _examples: list[Example]
    _example_prompt: ChatPromptTemplate
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
        examples: Optional[Union[list[Example], Example]] = None,
    ):
        self._init_model(
            model, llm_provider, model_id, base_url, api_key, model_params
        )

        self._init_example_prompt(examples)

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

        if self._example_prompt:
            system_prompt_template += f'\nExamples:\n{self._example_prompt}'

        self._system_prompt_template = system_prompt_template

    def _init_example_prompt(
        self,
        examples: Optional[Union[list[Example], Example]] = None,
    ):
        example_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", "{query}"),
                ("ai", "{question}"),
            ]
        )
        if not examples:
            examples = DEFAULT_EXAMPLES_FOR_Q_2_Q

        if not isinstance(examples, list):
            examples = [examples]

        self._example_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt_template,
            examples=examples,
        )

    def run(self, query: str) -> str:
        system_prompt_template = SystemMessage(
            content=self._system_prompt_template
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt_template,
                self._example_prompt,
                ('human', '{query}'),
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
