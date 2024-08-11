# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0


import logging
from collections.abc import Set

from kif_lib import (
    IRI,
    AnnotationRecord,
    AnnotationRecordSet,
    Descriptor,
    Entity,
    FilterPattern,
    Item,
    ItemDescriptor,
    KIF_Object,
    SnakSet,
    Statement,
    Store,
    String,
    ValueSnak,
)
from kif_lib.typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Union,
    override,
)
from kif_lib.vocabulary import wd

from ..llm.constants import (
    SIBLINGS_QUERY_SURFACE_FORM,
    WIKIDATA_SPARQL_ENDPOINT_URL,
    ChatRole,
    Disambiguation_Method,
    LLM_Model_Type,
)
from ..llm.disambiguation import (
    baseline_disambiguation,
    keyword_based_disambiguation,
    llm_based_disambiguation,
    similarity_based_disambiguation,
)
from ..llm.llm_parsers import LLM_Parsers
from ..llm.utils import is_url
from .abc import CHAT_CONTENT, LLM

LOG = logging.getLogger(__name__)


class LLM_Store(
    Store,
    store_name='llm',
    store_description='''KIF Store on top of Large Language
    Models.
    Disclaimer: LLMs can make mistakes. Check important info.''',
):
    """LLM store.

    Parameters:
       store_name: Store plugin to instantiate.
       llm_name: Name of the LLM to be used
       llm_endpoint: Name of the LLM to be used
       llm_api_key: API Key to access the LLM model
       prompt_template: Template to use while filtering
       llm_model_id: LLM model identifier
       context: Text to In-Context Prompting
       context_disambiguation: Text to In-Context Prompting in
        disambiguation step
       enforce_context: Whether to enforce LLM to search the answer
        in context or use the context to support the answer
       model_args: Arguments to the LLM model, e.g. {'max_new_tokens': 2048}
    """

    __slots__ = (
        '_llm_name',
        '_llm_endpoint',
        '_llm_api_key',
        '_llm_model_id',
        '_llm_model',
        '_llm_model_type',
        '_disambiguate',
        '_disambiguation_method',
        '_prompt_template',
        '_create_item',
        '_wikidata_store',
        '_context',
        '_context_disambiguation',
        '_enforce_context',
        '_parser',
        '_model_args',
    )

    _llm_name: str
    _llm_endpoint: Optional[str]
    _llm_api_key: Optional[str]
    _llm_model_id: str
    _llm_model: LLM
    _llm_model_type: LLM_Model_Type

    _disambiguate: bool
    _disambiguation_method: Disambiguation_Method

    _create_item: bool

    _wikidata_store: Store

    _enforce_context: bool

    _parse: Optional[Callable[[str], list[str]]]
    _prompt_template: Optional[Union[str, dict[ChatRole, CHAT_CONTENT]]]

    _context: Optional[str]
    _context_disambiguation: Optional[str]

    _model_args: dict[str, Any]

    def __init__(
        self,
        store_name: str,
        llm_name: str,
        llm_endpoint: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model_id: Optional[str] = None,
        wikidata_store: Optional[Store] = None,
        llm_model_type: Optional[Union[str, LLM_Model_Type]] = None,
        prompt_template: Optional[
            Union[str, dict[ChatRole, CHAT_CONTENT]]
        ] = None,
        disambiguate=True,
        disambiguation_method: Optional[
            Union[Disambiguation_Method, str]
        ] = None,
        parser: Optional[Callable[[str], list[str]]] = None,
        examples: Optional[list[str]] = None,
        context: Optional[str] = None,
        context_disambiguation: Optional[str] = None,
        enforce_context=True,
        create_item=True,
        model_args: dict[str, Any] = {},
        **kwargs: Any,
    ):
        assert store_name == self.store_name

        self._llm_name = llm_name
        self._llm_endpoint = llm_endpoint
        self._llm_api_key = llm_api_key

        self.llm_model_type = llm_model_type or LLM_Model_Type.GENERAL

        self.disambiguation_method = (
            disambiguation_method or Disambiguation_Method.LLM
        )

        self._prompt_template = prompt_template

        self._disambiguate = disambiguate

        self._context = context
        self._enforce_context = enforce_context

        self._context_disambiguation = context_disambiguation

        self._parser = parser or LLM_Parsers.to_list

        self._create_item = create_item

        super().__init__(**kwargs)

        llm_params: dict[str, Any] = {
            'llm_name': llm_name,
            'endpoint': llm_endpoint,
            'api_key': llm_api_key,
        }

        self._wikidata_store = (
            wikidata_store
            if wikidata_store
            else Store('sparql', WIKIDATA_SPARQL_ENDPOINT_URL)
        )

        self._model_args = model_args
        if llm_model_id:
            self._llm_model_id = llm_model_id
            llm_params['model_id'] = llm_model_id
        self._llm_model = LLM(**{**llm_params, **model_args})

    @property
    def name(self) -> str:
        return self._llm_name

    @property
    def endpoint(self) -> Optional[str]:
        return self._llm_endpoint

    @property
    def api_key(self):
        return self._llm_api_key

    @property
    def model_id(self) -> str:
        return self._llm_model_id

    @property
    def prompt_template(
        self,
    ) -> Optional[Union[str, dict[ChatRole, CHAT_CONTENT]]]:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value: Union[str, dict[ChatRole, CHAT_CONTENT]]):
        self._prompt_template = value

    @property
    def llm_model_type(self) -> LLM_Model_Type:
        return self._llm_model_type

    @llm_model_type.setter
    def llm_model_type(self, value: Union[LLM_Model_Type, str]):
        mt: LLM_Model_Type
        if isinstance(value, str):
            for setting in LLM_Model_Type:
                if value == setting.value:
                    mt = setting
        else:
            mt = value
        self._llm_model_type = mt

    @property
    def parser(self) -> Optional[Callable[[str], list[str]]]:
        return self._parser

    @parser.setter
    def parser(self, value: Callable[[str], list[str]]):
        self._parser = value

    @property
    def disambiguate(self) -> bool:
        return self._disambiguate

    @disambiguate.setter
    def disambiguate(self, value: bool):
        self._disambiguate = value

    @property
    def enforce_context(self) -> bool:
        return self._enforce_context

    @enforce_context.setter
    def enforce_context(self, value: bool):
        self._enforce_context = value

    @property
    def create_item(self) -> bool:
        return self._create_item

    @create_item.setter
    def create_item(self, value: bool):
        self._create_item = value

    @property
    def disambiguation_method(self) -> Disambiguation_Method:
        return self._disambiguation_method

    @disambiguation_method.setter
    def disambiguation_method(self, value: Union[Disambiguation_Method, str]):
        dm: Disambiguation_Method
        if isinstance(value, str):
            for setting in Disambiguation_Method:
                if value == setting.value:
                    dm = setting
        else:
            dm = value
        self._disambiguation_method = dm

    @property
    def context(self) -> Optional[str]:
        return self._context

    @context.setter
    def context(self, value: str):
        self._context = value

    @property
    def llm_model(self) -> LLM:
        return self._llm_model

    @llm_model.setter
    def llm_model(self, value: LLM):
        self._llm_model = value

    @property
    def context_disambiguation(self) -> Optional[str]:
        return self._context_disambiguation

    @context_disambiguation.setter
    def context_disambiguation(self, value: str):
        self._context_disambiguation = value

    @property
    def model_args(self) -> dict[str, Any]:
        return self._model_args

    @model_args.setter
    def model_args(self, value: dict[str, Any]):
        self._model_args = value

    def execute_probing(
        self,
        pattern: FilterPattern,
        model: LLM,
        model_type: LLM_Model_Type,
        limit: int,
        distinct=True,
        parser: Optional[Callable[[str], list[str]]] = None,
        context: Optional[str] = None,
        task_template: Optional[str] = None,
        prompt_template: Optional[
            Union[str, dict[ChatRole, CHAT_CONTENT]]
        ] = None,
        **kwargs: Any,
    ) -> list[str]:

        if not parser:
            parser = LLM_Parsers.to_list

        prompt: Union[str, dict[ChatRole, CHAT_CONTENT]]
        if not prompt_template:
            prompt_template = self._default_prompt_template(
                model_type=model_type,
                context=context,
                enforce_context=kwargs.get('enforce_context', True),
            )
            if not task_template:
                # task_template = 'Find X to complete the relation:\n'
                task_template = 'Fill in the gap to complete the relation:\n'
                task_template += '{subject} {predicate} {object}'

        prompt = self._complete_prompt_template(
            prompt_template=prompt_template,
            pattern=pattern,
            context=context,
            task_template=task_template,
        )

        try:
            response_from_llm, _ = model.execute_prompt(prompt)
        except Exception as e:
            raise e

        try:
            labels = parser(response_from_llm)
            if distinct:
                labels = list(set(labels))
            return labels[:limit]
        except Exception as e:
            raise e

    def execute_disambiguation(
        self,
        pattern: FilterPattern,
        labels: list[str],
        prompt_template: Optional[str] = None,
        context: Optional[str] = None,
        create_item=False,
        disambiguation_method=Disambiguation_Method.LLM,
    ) -> dict[str, Union[str, Item]]:
        return self._disambiguation_pipeline(
            labels=labels,
            pattern=pattern,
            method=disambiguation_method,
            prompt_template=prompt_template,
            create_item=create_item,
            context=context,
        )

    def _run_pipeline(
        self,
        pattern: FilterPattern,
        model: LLM,
        model_type: LLM_Model_Type,
        limit: int,
        disambiguation_method: Disambiguation_Method,
        create_item=False,
        disambiguate=True,
        distinct=False,
        enforce_context=False,
        context: Optional[str] = None,
        context_disambiguation: Optional[str] = None,
        prompt_template: Optional[
            Union[str, dict[ChatRole, CHAT_CONTENT]]
        ] = None,
        parser: Optional[Callable[[str], list[str]]] = LLM_Parsers.to_list,
    ) -> Iterator[Statement]:

        try:
            labels = self.execute_probing(
                pattern=pattern,
                model=model,
                model_type=model_type,
                enforce_context=enforce_context,
                context=context,
                prompt_template=prompt_template,
                limit=limit,
                parser=parser,
                distinct=distinct,
            )
        except Exception as exc:
            logging.error('Exception during probing: %s', exc)
            raise exc

        if disambiguate:
            try:
                # TODO: return a Iterator so each disambiguated option
                # may be parsed to a stmt in advance

                if labels and labels.__len__() > 0:
                    disam = self.execute_disambiguation(
                        labels=labels,
                        pattern=pattern,
                        prompt_template=prompt_template,
                        context=context_disambiguation,
                        disambiguation_method=disambiguation_method,
                        create_item=create_item,
                    )
                    return self._to_kif_stmts(
                        pattern=pattern,
                        entities=disam,
                    )
            except Exception as e:
                raise e

        return self._to_kif_stmts(pattern=pattern, entities=labels)

    def _to_kif_stmts(
        self,
        pattern: FilterPattern,
        entities: Union[list[str], dict[str, Union[str, Item]]],
    ) -> Iterator[Statement]:
        assert entities is not None
        subject = pattern.subject.entity
        property = pattern.property.property

        if not subject or not property:
            return

        if isinstance(entities, list):
            for value in entities:
                vs = ValueSnak(property=property, value=String(value))
                stmt = Statement(subject, vs)
                yield stmt
        elif isinstance(entities, dict):
            for item in entities.values():
                if isinstance(item, str) and is_url(item):
                    vs = ValueSnak(property=property, value=IRI(item))
                    stmt = Statement(subject, vs)
                    yield stmt
                elif isinstance(item, Item):
                    vs = ValueSnak(property=property, value=item)
                    stmt = Statement(subject, vs)
                    yield stmt

    def get_entities(self, obj: KIF_Object) -> set[Entity]:
        if not isinstance(obj, KIF_Object):
            return set()
        elif obj.is_entity():
            return set([obj])
        else:
            return set.union(*map(self.get_entities, obj.args))

    def _query_wikidata(self, query: str) -> Iterator[dict[str, Any]]:
        # TODO implement when convinient
        yield {}

    def _get_siblings_entities(
        self, subject, limit=5
    ) -> Iterator[dict[str, Any]]:
        assert subject is not None
        assert limit > 0

        sparql_query = SIBLINGS_QUERY_SURFACE_FORM.format(subject, subject)

        return self._query_wikidata(sparql_query)

    def _wikipedia_paragraphs_loader(self, entity: str, sections: list) -> str:
        import wikipedia
        from wikipedia.exceptions import DisambiguationError, PageError

        """load the Wikipedia paragraphs of an entity and section

        :param entity: entity string
        :param sections: list of section titles
        :return: Wikipedia content string
        """
        wikipedia.set_lang('en')
        try:
            page_content = wikipedia.page(entity, auto_suggest=False).content
        except (DisambiguationError, PageError):
            return ""

        wikipedia_paragraphs = {}
        section_content = page_content.split("\n\n\n")
        for section in sections:
            if section == 'introduction':
                wikipedia_paragraphs['introduction'] = section_content[0]
            else:
                # TODO: parse other sections
                pass
        return str(wikipedia_paragraphs)

    def _get_wikidata_examples(
        self, pattern: FilterPattern, limit=5
    ) -> dict[str, Any]:
        assert limit > 0
        new_pattern: FilterPattern = pattern

        """
        TODO: Currently it only works when the pattern is <S> <P> ?o
        We need to generalize this to other patterns like <S> ?p ?o
        """
        structured_examples: dict[str, Any] = {}
        if pattern.subject and pattern.property:
            for result in self._get_siblings_entities(
                subject=pattern.subject, limit=limit
            ):
                sibling = result['sibling']['value']
                sibling_label = result["siblingLabel"]["value"]

                subject = Item(sibling)
                new_pattern = FilterPattern(
                    subject=subject, property=pattern.property
                )

                stmts = self._wikidata_store.filter(
                    pattern=new_pattern, limit=20
                )
                property = wd.get_entity_label(pattern.property.property)

                for stmt in stmts:
                    if structured_examples.keys().__len__() <= limit:
                        example_line: dict[str, Any] = {}
                        example_line['s'] = sibling_label

                        if property:
                            example_line['p'] = property
                        else:
                            property_descriptor = dict(
                                self._wikidata_store.get_descriptor(
                                    stmt.snak.args[0],
                                    mask=Descriptor.LABEL,
                                )
                            )
                            for _, v in property_descriptor.items():
                                if v.label:
                                    example_line['p'] = v.label.value

                        subj_pred_key = example_line['s'] + example_line['p']

                        object_descriptor = dict(
                            self._wikidata_store.get_item_descriptor(
                                stmt.snak.args[1],
                                mask=Descriptor.LABEL,
                            )
                        )

                        for _, v in object_descriptor.items():
                            if v.label:
                                if subj_pred_key in structured_examples:
                                    structured_examples[subj_pred_key][
                                        'o'
                                    ].append(v.label.value)
                                else:
                                    example_line['o'] = [v.label.value]
                                    structured_examples[subj_pred_key] = (
                                        example_line
                                    )
                    else:
                        break

        return structured_examples

    def _get_examples(
        self,
        pattern: FilterPattern,
        prompt_template: str,
        limit: Optional[int] = 5,
        datasource: Optional[Any] = None,
    ) -> str:

        examples = ''
        if not datasource:
            structured_examples = self._get_wikidata_examples(
                pattern=pattern, limit=limit
            )

            for example in structured_examples.values():
                examples += f'{example["s"]}, {example["p"]}: {example["o"]}\n'
        return examples

    @override
    def _filter(
        self,
        pattern: FilterPattern,
        limit: int,
        distinct=False,
    ) -> Iterator[Statement]:
        assert (
            limit > 0
        ), 'Invalid limit value. Please, provide a limit bigger than 0.'
        return self._run_pipeline(
            pattern=pattern,
            model=self.llm_model,
            disambiguate=self.disambiguate,
            disambiguation_method=self.disambiguation_method,
            prompt_template=self.prompt_template,
            model_type=self.llm_model_type,
            limit=limit,
            distinct=distinct,
            create_item=self.create_item,
            parser=self.parser,
            context=self.context,
            context_disambiguation=self.context_disambiguation,
            enforce_context=self.enforce_context,
        )

    def _cache_get_annotations(
        self, stmt: Statement
    ) -> Optional[Set[AnnotationRecord]]:
        return self._cache.get(stmt, "annotations")

    def _get_annotations(
        self, stmts: Iterable[Statement]
    ) -> Iterator[tuple[Statement, Optional[AnnotationRecordSet]]]:
        return self._wikidata_store.get_annotations(stmts)

    def get_item_descriptor(
        self,
        items: Union[Item, Iterable[Item]],
        language: Optional[str] = None,
        mask: Optional[Descriptor.TAttributeMask] = None,
    ) -> Iterator[tuple[Item, Optional[ItemDescriptor]]]:
        return self._wikidata_store.get_item_descriptor(items, language, mask)

    def _disambiguation_pipeline(
        self,
        pattern: FilterPattern,
        labels: list[str],
        create_item=False,
        method: Optional[Disambiguation_Method] = None,
        keywords: dict[str, list[str]] = {},
        context: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> dict[str, Union[str, Item]]:
        if not method:
            method = Disambiguation_Method.LLM
            if context:
                method = Disambiguation_Method.SIM_IN_CONTEXT

        assert labels is not None and labels.__len__() > 0
        assert (
            method in Disambiguation_Method
        ), f'No disambiguation method named {method}'

        if method == Disambiguation_Method.KEYWORD:
            return keyword_based_disambiguation(
                labels=labels,
                keywords=keywords,
                create_item=create_item,
            )
        elif method == Disambiguation_Method.SIM_IN_CONTEXT and context:
            return similarity_based_disambiguation(
                labels=labels, context=context
            )
        elif method == Disambiguation_Method.BASELINE:
            return baseline_disambiguation(
                labels=labels, create_wikidata_entity=create_item
            )

        return llm_based_disambiguation(
            labels=labels,
            pattern=pattern,
            llm_model=self._llm_model,
            prompt_template=None,
            context=context,
            create_wikidata_entity=create_item,
            complete_template=self._complete_prompt_template,
        )

    def _default_prompt_template(
        self,
        context: Optional[str] = None,
        enforce_context=True,
        model_type: LLM_Model_Type = LLM_Model_Type.GENERAL,
    ) -> dict[ChatRole, CHAT_CONTENT]:
        """
        generate prompts based on input entities and templates

        :param prompt_template: template string
        :return: prompt
        """
        assert model_type in LLM_Model_Type

        prompt = {
            ChatRole.SYSTEM: '',
            ChatRole.USER: '',
            ChatRole.ASSISTANT: '',
        }

        # TODO: create prompts optimized to each type of model
        # for while we are considering instruct models as general
        if (
            model_type == LLM_Model_Type.GENERAL
            or model_type == LLM_Model_Type.INSTRUCT
        ):
            if context:
                if enforce_context:
                    prompt[ChatRole.SYSTEM] += (
                        'You are a helpful and honest assistant that '
                        'resolves a TASK based on the CONTEXT. '
                        'Only perfect and explicit matches mentioned '
                        'in CONTEXT are accepted. Please, respond '
                        'concisely, with no further explanation, '
                        'and truthfully.'
                    )
                else:
                    prompt[ChatRole.SYSTEM] += (
                        'You are a helpful and honest assistant that '
                        'resolves a TASK. Use the CONTEXT '
                        'to support the answer. Please, respond '
                        'concisely, with no further explanation, '
                        'and truthfully.'
                    )
            else:
                prompt[ChatRole.SYSTEM] += (
                    'You are a helpful and honest assistant that '
                    'resolves a TASK. Please, respond concisely, with '
                    'no further explanation, and truthfully.'
                )

            prompt[ChatRole.USER] += '\n\nTASK:\n"{task_template}"'

            if context:
                prompt[ChatRole.USER] += '\n\nCONTEXT:\n"{context}"'

            prompt[ChatRole.USER] += (
                '\n\nThe output should be only a '
                'list containing the answers, such as ["answer_1", '
                '"answer_2", ..., "answer_n"]. Do not provide '
                'any further explanation and avoid false answers. '
                'Return an empty list, such as [], if no information '
                'is available.'
            )

        return prompt

    def _complete_prompt_template(
        self,
        pattern: FilterPattern,
        prompt_template: Union[str, dict[ChatRole, CHAT_CONTENT]],
        context: Optional[str] = None,
        task_template: Optional[str] = None,
    ) -> Union[str, dict[ChatRole, CHAT_CONTENT]]:
        # mount the triple prompt: {subject} {predicate} {object} : _ _ _
        assert prompt_template is not None

        subject = (
            wd.get_entity_label(pattern.subject.entity)
            if pattern.subject
            else '_'
        )

        predicate = (
            wd.get_entity_label(pattern.property.property)
            if pattern.property
            else '_'
        )

        object: str = '_'
        if pattern.value:

            if pattern.value.snak_set and isinstance(
                pattern.value.snak_set, SnakSet
            ):
                object = 'X'
                object += ' where X'
                for objects in pattern.value.snak_set[0]:
                    object += f' {wd.get_entity_label(objects)}'
            else:
                object = wd.get_entity_label(pattern.value.value)

        context = context or ''
        task_template = task_template or ''

        def replace_templates(template: str) -> str:
            response = template.replace('{task_template}', task_template)
            response = response.replace('{subject}', subject)
            response = response.replace('{predicate}', predicate)
            response = response.replace('{object}', object)
            response = response.replace('{context}', context)
            return response

        if isinstance(prompt_template, str):
            return replace_templates(prompt_template)

        prompt_template[ChatRole.USER.value] = replace_templates(
            prompt_template[ChatRole.USER.value]
        )
        return prompt_template
