# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from typing import Dict, Tuple

from kif_lib import Item, Property
from kif_lib.typing import (
    Any,
    Callable,
    Iterator,
    Optional,
)
from kif_lib.vocabulary import wd
from langchain_core import language_models as LC_Models
from langchain_core import output_parsers as LC_Parsers
from langchain_core import prompts as LC_Prompts

from ..constants import (
    DEFAULT_WIKIDATA_SEARCH_API_TEMPLATE,
    PID,
    QID,
    WID,
    Label,
)
from .abc import Disambiguator
from .util import fetch_wikidata_entities

LOG = logging.getLogger(__name__)


class LLM_Disambiguator(Disambiguator, disambiguator_name='llm'):

    _model: LC_Models.BaseChatModel
    _sentence_term_template: str
    _textual_context: Optional[str]

    def __init__(
        self,
        model: LC_Models.BaseChatModel,
        sentence_term_template: str,
        textual_context: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._model = model
        self._sentence_term_template = sentence_term_template
        self._textual_context = textual_context

    @property
    def textual_context(self) -> str:
        return self._textual_context

    @property
    def model(self) -> LC_Models.BaseChatModel:
        return self._model

    @property
    def sentence_term_template(self) -> str:
        return self._sentence_term_template

    async def _disambiguate_item(
        self,
        label: str,
        url_template: Optional[str] = None,
        url_template_mapping: Optional[Dict[str, Any]] = None,
        parser_fn: Optional[Callable[..., Tuple[Label, QID]]] = None,
        limit=10,
    ) -> Iterator[Tuple[Label, Optional[Item]]]:
        """
        Disambiguates a label by retrieving its Wikidata Item ID.

        Args:
            label (str): The label to disambiguate.
            url_template (str): Optional template for the URL to query.
            parser_fn (function): Optional function to parse the response.
            limit (int): Maximum number of results to retrieve per entity type.
        Returns:
            Optional[Item]: A Wikidata item if found, otherwise None.
        """

        assert label, 'Label can not be undefined.'

        try:
            label_from_wikidata, q_id = (
                await self._llm_entity_disambiguation(
                    label,
                    'item',
                    url_template,
                    url_template_mapping,
                    parser_fn,
                    limit,
                )
            )

            if q_id:
                if label_from_wikidata:
                    return label_from_wikidata, wd.Q(
                        name=q_id, label=label_from_wikidata
                    )
                else:
                    return label, wd.Q(name=q_id, label=label)
            else:
                return label, None

        except Exception as e:
            raise e

    async def _disambiguate_property(
        self,
        label: str,
        url_template: Optional[str] = None,
        url_template_mapping: Optional[Dict[str, Any]] = None,
        parser_fn: Optional[Callable[..., Tuple[Label, PID]]] = None,
        limit=10,
    ) -> Iterator[Tuple[Label, Optional[Property]]]:
        """
        Disambiguates a label by retrieving its Wikidata Property ID.

        Args:
            label (str): The label to disambiguate.
            url_template (str): Optional template for the URL to query.
            parser_fn (function): Optional function to parse the response.
            limit (int): Maximum number of results to retrieve per property
              type.
        Returns:
            Optional[Property]: A Wikidata property if found, otherwise None.
        """

        assert label, 'Label can not be undefined.'

        try:
            label_from_wikidata, p_id = (
                await self._llm_entity_disambiguation(
                    label,
                    'property',
                    url_template,
                    url_template_mapping,
                    parser_fn,
                    limit,
                )
            )

            if p_id:
                if label_from_wikidata:
                    return label_from_wikidata, wd.P(
                        name=p_id, label=label_from_wikidata
                    )
                else:
                    return label, wd.P(name=p_id, label=label)
            else:
                return label, None
        except Exception as e:
            raise e

    async def _llm_entity_disambiguation(
        self,
        label: str,
        entity_type: str,
        url_template: Optional[str] = None,
        url_template_mapping: Optional[Dict[str, Any]] = None,
        parser_fn: Optional[Callable[..., Tuple[Label, WID]]] = None,
        limit=10,
    ) -> Tuple[Label, Optional[WID]]:
        """
        Disambiguates a label by retrieving its Wikidata Entity ID.

        Args:
            label (str): The label to disambiguate.
            entities_types (list[str]): List of entity types to include in the
              search.
            url_template (str): Optional template for the URL to query.
            parser_fn (function): Optional function to parse the response.
            limit (int): Maximum number of results to retrieve per entity type.
        Returns:
            Tuple[Label, Optional[WID]]:
        """
        assert label, 'Label can not be None'

        url = DEFAULT_WIKIDATA_SEARCH_API_TEMPLATE.format_map(
            defaultdict(str, label=label, limit=limit, type=entity_type)
        )
        if url_template:
            try:
                url = url_template.format_map(
                    defaultdict(str, url_template_mapping)
                )
            except Exception as e:
                LOG.warning(
                    f'Invalid URL template {url_template}: {e}. Using the ',
                    'default template.',
                )

        w_id: Optional[str] = None
        try:
            wikidata_results = await fetch_wikidata_entities(
                label, url, parser_fn
            )
            label_from_wikidata = label
            if wikidata_results and len(wikidata_results) > 0:
                candidate_prompt = ''
                for candidate_entity in wikidata_results:
                    candidate_prompt += f'Candidate: {candidate_entity['id']}\n'
                    if 'label' in candidate_entity:
                        candidate_prompt += f'Label: {candidate_entity['label']}\n'
                    if 'description' in candidate_entity:
                        candidate_prompt += f'Description: {candidate_entity['description']}\n\n'

                system_template, user_template = self._default_prompt(entity_type)

                promp_template = LC_Prompts.ChatPromptTemplate.from_messages(
                    [('system', system_template), ('human', user_template)]
                )
                from langchain_core.runnables import RunnableLambda

                def parse_entity(w_id: str):
                    import re
                    fl = 'Q'
                    if entity_type == 'property':
                        fl = 'P'
                    match = re.search(fr'{fl}\d+', w_id)

                    if match:
                        return match.group()
                    return None

                parse_to_entity = RunnableLambda(lambda w_id: parse_entity(w_id))

                chain = (
                  promp_template
                  | self.model
                  | LC_Parsers.StrOutputParser()
                  | parse_to_entity
                )

                sentence = self.sentence_term_template.format(term=label)
                w_id = await chain.ainvoke({
                  'context': self.textual_context,
                  'sentence': sentence,
                  'term': label,
                  'candidates': candidate_prompt
                })

                if w_id:
                    entity_info = [d for d in wikidata_results if w_id in d]
                    if entity_info:
                        label_from_wikidata = entity_info['label']

            if not w_id:
                LOG.info(
                    f'No Wikidata entity was found to the label `{label}`.'
                )
            return label_from_wikidata, w_id
        except Exception as e:
            raise e

    def _default_prompt(self, entity_type: str) -> Tuple[str, str]:
        example = "Q123456"
        if entity_type == 'property':
            example = "P123456"
        user = f'''\
Candidates:
{{candidates}}'''

        x = 'description and label'
        if self.textual_context:
            x = 'description, label, and CONTEXT'
            user += '''

CONTEXT:
{context}'''

        system = f'''\
You are a helpful and honest assistant that given a list of candidates, \
select the one whose {x} best fit the TERM in the sentence \
"{{sentence}}". The correct candidate should be one that accurately completes \
the sentence based on the provided {x}. Return only the \
candidate ID, such as "{example}", if the candidate is a valid match for both \
the label/description and the context of the sentence. Reason about the \
answer, checking whether any candidate should actually be returned. If no \
candidate fits the context, return an empty string. Please, respond \
concisely, with no further explanation, and truthfully.\

TERM: {{term}}'''
        return (system, user)
