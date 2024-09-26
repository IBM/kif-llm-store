# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
import asyncio
import nest_asyncio
from collections.abc import Set
from typing import List, Dict

from kif_lib import (
    AnnotationRecord,
    AnnotationRecordSet,
    Descriptor,
    Filter,
    Item,
    ItemDescriptor,
    Statement,
    Store,
    ValueSnak,
)

from kif_lib.typing import (
    Any,
    AsyncIterator,
    ClassVar,
    Iterable,
    Iterator,
    Optional,
    Union,
    override,
)

from .language_models import BaseChatModel
from .output_parsers import (
    CommaSeparatedListOutputParserCleaned,
    BaseOutputParser,
)
from .prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence

from ..llm.compiler.llm.filter_compiler import (
    LLM_FilterCompiler,
    Variable,
    LogicalComponent,
)

from ..llm.constants import (
    DEFAULT_SYSTEM_PROMPT_INSTRUCTION,
    SYSTEM_PROMPT_INSTRUCTION_WITH_CONTEXT,
    SYSTEM_PROMPT_INSTRUCTION_WITH_ENFORCED_CONTEXT,
    WIKIDATA_SPARQL_ENDPOINT_URL,
    Disambiguation_Method,
    KIF_FilterTypes,
    LLM_Providers,
)


nest_asyncio.apply()

LOG = logging.getLogger(__name__)


class LLM_Store(
    Store,
    store_name='llm',
    store_description=(
        'KIF Store powered by Large Language Models.'
        'Disclaimer: LLMs can make mistakes. Check important info.'
    ),
):
    """LLM Store

    .. Setup::

        Install ``kif-llm-store`` and set the models you want to use from LLM Store.

        .. code-block:: bash

            pip install -e .
            pip install -U kif-llm-store

    Properties::
      store_name: Store plugin to instantiate.
      model: LangChain LLM model instance.
      llm_provider: LLM provider to be used, such as `watsonx` for IBM WatsonX
      base_url: Endpoint to access the LLM provider
      api_key: API Key to access the LLM provider
      prompt_template: Template to use while filtering
      model_id: LLM model identifier
      textual_context: Text to In-Context Prompting
      enforce_context: Whether to enforce LLM to search the answer
        in context or use the context to support the answer
      model_args: Arguments to the LLM model, e.g. {'max_new_tokens': 2048}
    """  # noqa E501

    __slots__ = (
        '_model',
        '_task_prompt_template',
        '_parser',
        '_textual_context',
        '_disambiguation_method',
        '_disambiguation_model',
        '_wikidata_store',
        '_examples',
        '_output_format_prompt',
        '_enforce_context',
        '_create_entity',
        '_compiler',
    )

    _model: BaseChatModel
    _task_prompt_template: Optional[str]
    _parser: Optional[BaseOutputParser]
    _textual_context: Optional[str]
    _disambiguation_method: Optional[Disambiguation_Method]
    _disambiguation_model: Optional[BaseChatModel]
    _output_format_prompt: Optional[str]
    _compiler: LLM_FilterCompiler

    def __init__(
        self,
        store_name: str,
        model: Optional[BaseChatModel] = None,
        llm_provider: Optional[LLM_Providers] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        task_prompt_template: Optional[str] = None,
        parser: Optional[BaseOutputParser] = None,
        textual_context: Optional[str] = None,
        disambiguation_method: Optional[Disambiguation_Method] = None,
        disambiguation_model: Optional[BaseChatModel] = None,
        wikidata_store: Optional[Store] = None,
        examples: Optional[List[str]] = None,
        output_format_prompt: Optional[str] = None,
        enforce_context=True,
        create_entity=False,
        model_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert store_name == self.store_name

        if model:
            self._model = model
        else:
            self._model = LLM_Store._init_model(
                llm_provider=llm_provider,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                model_params=model_params,
                **kwargs,
            )

        default_parser_fn = CommaSeparatedListOutputParserCleaned()

        self._parser = parser or default_parser_fn
        self._disambiguation_model = disambiguation_model or self._model

        self._output_format_prompt = (
            output_format_prompt or default_parser_fn.get_format_instructions()
        )

        self._disambiguation_method = (
            disambiguation_method or Disambiguation_Method.BASELINE
        )

        self._task_prompt_template = task_prompt_template

        self._examples = examples

        self._textual_context = textual_context

        self._enforce_context = enforce_context

        self._create_entity = create_entity

        self._wikidata_store = (
            wikidata_store
            if wikidata_store
            else Store("sparql", WIKIDATA_SPARQL_ENDPOINT_URL)
        )

    @classmethod
    def from_model_providers_args(
        cls,
        llm_provider: LLM_Providers,
        model_id: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        disambiguation_method: Optional[Disambiguation_Method] = None,
        *args,
        **model_kwargs: Any,
    ):
        model = cls._init_model(
            llm_provider=llm_provider,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            **model_kwargs,
        )

        return cls(
            store_name="llm",
            model=model,
            disambiguation_method=disambiguation_method,
            *args,
            **model_kwargs,
        )

    @classmethod
    def _init_model(
        cls,
        llm_provider: LLM_Providers,
        model_id: str,
        base_url: str,
        api_key: str,
        model_params: Optional[Dict[str, Any]] = {},
        **kwargs,
    ) -> BaseChatModel:

        assert llm_provider, "No LLM provider was set."
        assert llm_provider in LLM_Providers, "Invalid LLM provider."
        assert model_id, "No model identifier was set."

        llm_params: Dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
        }

        try:
            if llm_provider == LLM_Providers.OPEN_AI:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=model_id, **{**llm_params, **model_params, **kwargs}
                )
            elif llm_provider == LLM_Providers.OLLAMA:
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=model_id, **{**llm_params, **model_params, **kwargs}
                )
            elif llm_provider == LLM_Providers.IBM:
                from langchain_ibm import ChatWatsonx

                return ChatWatsonx(
                    model_id=model_id,
                    apikey=api_key,
                    url=base_url,
                    params=model_params,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unsupported provider: {llm_provider}")
        except Exception as e:
            raise e

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @model.setter
    def model(self, value: BaseChatModel):
        self._model = value

    @property
    def task_prompt_template(
        self,
    ) -> Optional[str]:
        return self._task_prompt_template

    @task_prompt_template.setter
    def task_prompt_template(self, value: str):
        self._task_prompt_template = value

    @property
    def output_format_prompt(self) -> Optional[str]:
        return self._output_format_prompt

    @output_format_prompt.setter
    def output_format_prompt(self, value: str) -> Optional[str]:
        self._output_format_prompt = value

    @property
    def parser(self) -> Optional[BaseOutputParser]:
        return self._parser

    @parser.setter
    def parser(self, value: BaseOutputParser):
        self._parser = value

    @property
    def disambiguation_method(self) -> Disambiguation_Method:
        return self._disambiguation_method

    @disambiguation_method.setter
    def disambiguation_method(self, value: Disambiguation_Method):
        self._disambiguation_method = value

    @property
    def enforce_context(self) -> bool:
        return self._enforce_context

    @enforce_context.setter
    def enforce_context(self, value: bool):
        self._enforce_context = value

    @property
    def create_entity(self) -> bool:
        return self._create_entity

    @create_entity.setter
    def create_entity(self, value: bool):
        self._create_entity = value

    @property
    def textual_context(self) -> Optional[str]:
        return self._textual_context

    @textual_context.setter
    def textual_context(self, value: str):
        self._textual_context = value

    @override
    def _filter(
        self,
        filter: Filter,
        limit: int,
        distinct=False,
    ) -> Iterator[Statement]:
        assert limit > 0, (
            "Invalid limit value. Please, provide a limit " "bigger than 0."
        )

        async def sync_wrapper():
            stmts = []
            async for stmt in await self.afilter(filter, limit, distinct):
                stmts.append(stmt)
            return stmts

        return iter(asyncio.run(sync_wrapper()))

    async def afilter(
        self, filter: Filter, limit: int, distinct=False
    ) -> AsyncIterator[Statement]:
        assert (
            limit > 0
        ), 'Invalid limit value. Please, provide a limit bigger than 0.'

        self._compiler = self._compile_filter(filter)
        query = self._compiler.query_template
        if self._compiler.get_filter_type() == KIF_FilterTypes.ONE_VARIABLE:
            query = self._compiler.query_template.replace('var1', 'X')

        chain = self._create_pipline_chain(limit=limit, distinct=distinct)

        return await chain.ainvoke(
            {'query': query, 'textual_context': self.textual_context}
        )

    #: Flags to be passed to filter compiler.
    _compile_filter_flags: ClassVar[LLM_FilterCompiler.Flags] = (
        LLM_FilterCompiler.default_flags
    )

    def _compile_filter(self, filter: Filter) -> LLM_FilterCompiler:
        compiler = LLM_FilterCompiler(filter, self._compile_filter_flags)

        if self.has_flags(self.DEBUG):
            compiler.set_flags(compiler.DEBUG)
        else:
            compiler.unset_flags(compiler.DEBUG)
        if self.has_flags(self.BEST_RANK):
            compiler.set_flags(compiler.BEST_RANK)
        else:
            compiler.unset_flags(compiler.BEST_RANK)

        compiler.compile()
        return compiler

    def _create_pipline_chain(
        self, limit: int, distinct=True
    ) -> RunnableSequence:

        def distinct_fn(labels: List[str]):
            if distinct:
                labels = list(set(labels))
            return labels[:limit]

        prompt = self._build_prompt_template()

        do_distinct = RunnableLambda(distinct_fn)

        disambiguate = RunnableLambda(
            lambda labels: self._disambiguate(labels)
        )

        to_statements = RunnableLambda(
            lambda binds: self._to_statements(binds)
        )

        debug_chain = RunnableLambda(lambda entry: (print(entry), entry)[1])

        chain: RunnableSequence = (
            prompt
            | debug_chain
            | self.model
            | debug_chain
            | self.parser
            | debug_chain
            | do_distinct
            | debug_chain
            | disambiguate
            | debug_chain
            | to_statements
            | debug_chain
        )

        return chain

    async def _disambiguate(
        self, labels: List[str]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Asynchronously disambiguates labels.
        """
        binds = self._compiler.get_binds()

        from ..llm.entity_resolution import (
            Disambiguator,
            BaselineDisambiguator,
            LLM_Disambiguator,
        )

        disambiguator: Disambiguator = Disambiguator(
            BaselineDisambiguator.disambiguator_name
        )
        if self.disambiguation_method == Disambiguation_Method.LLM:
            if (
                self._compiler.get_filter_type()
                == KIF_FilterTypes.ONE_VARIABLE
            ):
                sentence = self._compiler.get_task_sentence_template().replace(
                    'var1', '{term}'
                )
                disambiguator = Disambiguator(
                    disambiguator_name=LLM_Disambiguator.disambiguator_name,
                    model=self._disambiguation_model,
                    sentence_term_template=sentence,
                )
        if isinstance(binds['property'], Variable):
            async for label, entity in disambiguator.adisambiguate_property(
                labels
            ):  # noqa: E501
                binds['property'].set_value(entity)
                yield binds
        else:
            async for label, entity in disambiguator.adisambiguate_item(
                labels
            ):  # noqa: E501
                if isinstance(binds['subject'], Variable):
                    binds['subject'].set_value(entity)
                if isinstance(binds['value'], Variable):
                    binds['value'].set_value(entity)
                yield binds

    async def _to_statements(
        self, binds: Dict[str, Any]
    ) -> AsyncIterator[Statement]:

        # from ..llm.utils import is_url

        async for bind in binds:

            def get_entity(key: str):
                if isinstance(bind[key], Variable):
                    entity = bind[key].get_value()
                    if entity:
                        return bind[key].get_value()
                    # if self._create_entity:
                    # TODO: create item
                    return None
                return bind[key]

            subject = get_entity('subject')
            if not subject:
                continue
            property = get_entity('property')
            if not property:
                continue
            value = get_entity('value')
            if not value:
                continue

            value_snaks = []
            if isinstance(value, LogicalComponent):
                for component in value.components:
                    value_snaks.append(
                        ValueSnak(
                            property=property,
                            value=component,
                        )
                    )
            else:
                value_snaks.append(
                    ValueSnak(
                        property=property,
                        value=value,
                    )
                )

            if isinstance(subject, LogicalComponent):
                for component in subject.components:
                    for vs in value_snaks:
                        stmt = Statement(component, vs)
                        yield stmt

            else:
                for vs in value_snaks:
                    stmt = Statement(subject, vs)
                    yield stmt

    def _cache_get_annotations(
        self, stmt: Statement
    ) -> Optional[Set[AnnotationRecord]]:
        return self._cache.get(stmt, "annotations")

    @override
    def _get_annotations(
        self, stmts: Iterable[Statement]
    ) -> Iterator[tuple[Statement, Optional[AnnotationRecordSet]]]:
        return self._wikidata_store.get_annotations(stmts)

    @override
    def _get_item_descriptor(
        self,
        items: Union[Item, Iterable[Item]],
        language: Optional[str] = None,
        mask: Optional[Descriptor.TAttributeMask] = None,
    ) -> Iterator[tuple[Item, Optional[ItemDescriptor]]]:
        return self._wikidata_store.get_item_descriptor(items, language, mask)

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """
        generate prompts based on input entities and templates

        :param prompt_template: template string
        :return: prompt
        """
        from langchain_core.messages import SystemMessage

        system = DEFAULT_SYSTEM_PROMPT_INSTRUCTION

        human = ''
        if self.textual_context:
            system = SYSTEM_PROMPT_INSTRUCTION_WITH_CONTEXT
            human += 'CONTEXT:\n"{textual_context}"\n\n'
            if self.enforce_context:
                system = SYSTEM_PROMPT_INSTRUCTION_WITH_ENFORCED_CONTEXT

        system += f' {self.output_format_prompt}'
        human += 'TASK:\n{query}'

        return ChatPromptTemplate.from_messages(
            [SystemMessage(content=system), ('human', human)]
        )
