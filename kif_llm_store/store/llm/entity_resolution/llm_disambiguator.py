# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

from kif_lib import Item, Property
from kif_lib.typing import (
    Any,
    Iterator,
    Optional,
)

from langchain_core import language_models as LC_Models
from langchain_core import output_parsers as LC_Parsers
from langchain_core import prompts as LC_Prompts

from .abc import Disambiguator


LOG = logging.getLogger(__name__)


class LLM_Disambiguator(Disambiguator, disambiguator_name='llm'):

    _model: LC_Models.BaseChatModel
    _sentence_term_template: str
    _textual_context: Optional[str]

    def __init__(
        self,
        disambiguator_name: str,
        model: LC_Models.BaseChatModel,
        sentence_term_template: str,
        textual_context: Optional[str] = None,
        *args,
        **kwargs,
    ):
        assert disambiguator_name == self.disambiguator_name
        super().__init__(*args, **kwargs)
        self._model = model
        self._textual_context = textual_context
        self._sentence_term_template = sentence_term_template

    @property
    def textual_context(self) -> str:
        return self._textual_context

    @property
    def model(self) -> LC_Models.BaseChatModel:
        return self._model

    @property
    def sentence_term_template(self) -> str:
        return self._sentence_term_template

    @sentence_term_template.setter
    def sentence_term_template(self, value):
        self._sentence_term_template = value

    async def _label_to_item(
        self,
        label: str,
        limit: Optional[int] = 10,
    ) -> Iterator[Tuple[str, Optional[Item]]]:
        """
        Disambiguates a label by retrieving its Item ID.

        Args:
            label (str): The label to disambiguate.
            limit (int): Maximum number of results to retrieve per entity type.
        Returns:
            Optional[Item]: A item if found, otherwise None.
        """

        assert label, 'Label can not be undefined.'

        try:
            results = await self._target_store.get_items_from_label(
                label,
                limit,
            )

            if not results:
                LOG.info(f'No item was found for the label `{label}`.')
                return label, None

            label_from_target_store, item_id = (
                await self._llm_entity_disambiguation(label, results, 'item')
            )

            if item_id:
                if label_from_target_store:
                    return label_from_target_store, item_id
                return label, item_id
            return label, None
        except Exception as e:
            raise e

    async def _label_to_property(
        self,
        label: str,
        limit: Optional[int] = 10,
    ) -> Iterator[Tuple[str, Optional[Property]]]:
        """
        Disambiguates a label by retrieving its Property ID.

        Args:
            label (str): The label to disambiguate.
            limit (int): Maximum number of results to retrieve per property
              type.
        Returns:
            Optional[Property]: A property if found, otherwise None.
        """

        assert label, 'Label can not be undefined.'

        try:
            results = await self._target_store.get_properties_from_label(
                label,
                limit,
            )

            if not results:
                LOG.info(f'No property was found for the label `{label}`.')
                return label, None

            label_from_target_store, property_id = (
                await self._llm_entity_disambiguation(
                    label, results, 'property', limit
                )
            )

            if property_id:
                if label_from_target_store:
                    return label_from_target_store, property_id
                return label, property_id
            return label, None

        except Exception as e:
            raise e

    async def _llm_entity_disambiguation(
        self,
        label: str,
        candidates: list[Any],
        entity_type: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Disambiguates a label by retrieving its Entity ID.

        Args:
            label (str): The label to disambiguate.
            entity_type (str): whether it is a `property` or an `item`.
            limit (int): Maximum number of results to retrieve per entity type.
        Returns:
            Tuple[Label, Optional[WID]]:
        """

        entity_id: Optional[str] = None
        try:
            label_from_target_store = label
            if candidates and len(candidates) > 0:
                c_prompt = ''
                for candidate in candidates:
                    c_prompt += f'Candidate: {candidate['id']}\n'
                    if 'label' in candidate:
                        c_prompt += f'Label: {candidate['label']}\n'
                    if 'description' in candidate:
                        c_prompt += f'Description: {candidate['description']}\n\n'  # noqa E501

                s_template, u_template = self._default_prompt(entity_type)

                promp_template = LC_Prompts.ChatPromptTemplate.from_messages(
                    [('system', s_template), ('human', u_template)]
                )
                from langchain_core.runnables import RunnableLambda

                # TODO: fix to work with any knowledge source than Wikidata
                def parse_entity(w_id: str) -> Optional[str]:
                    import re

                    fl = 'Q'
                    if entity_type == 'property':
                        fl = 'P'
                    match = re.search(fr'{fl}\d+', w_id)

                    if match:
                        return match.group()
                    return None

                to_entity = RunnableLambda(lambda w_id: parse_entity(w_id))

                debug = RunnableLambda(
                    lambda entry: (LOG.info(entry), entry)[1]
                )

                chain = (
                    promp_template
                    | debug
                    | self.model
                    | debug
                    | LC_Parsers.StrOutputParser()
                    | debug
                    | to_entity
                    | debug
                )

                sentence = self.sentence_term_template.format(term=label)
                entity_id = await chain.ainvoke(
                    {
                        'context': self.textual_context,
                        'sentence': sentence,
                        'term': label,
                        'candidates': c_prompt,
                    }
                )

                if entity_id:
                    entity_info = [
                        c for c in candidates if entity_id == c['id']
                    ]
                    if entity_info:
                        label_from_target_store = entity_info[0]['label']
                        entity_id = entity_info[0]['iri']

            if not entity_id:
                LOG.info(f'No entity was found to the label `{label}`.')
            return label_from_target_store, entity_id
        except Exception as e:
            raise e

    def _default_prompt(self, entity_type: str) -> Tuple[str, str]:
        example = "Q123456"
        if entity_type == 'property':
            example = "P123456"
        user = '''\
CANDIDATES:
{candidates}'''

        x = 'description and label'
        if self.textual_context:
            x = 'description, label, and CONTEXT'
            user += '''

CONTEXT:
{context}'''

        system = f'''\
You are a helpful and honest assistant that given a list of CANDIDATES, \
select the one whose {x} best fit the TERM in the sentence \
"{{sentence}}". The correct candidate should be one that accurately completes \
the sentence based on the provided {x}. Return only the \
candidate ID, such as "{example}", if the candidate is a valid match for both \
the label/description and the context of the sentence. Reason about the \
answer, checking whether any candidate should actually be returned. If no \
candidate fits the context, return an empty string. Please, respond \
concisely, truthfully and with no further explanation.\

TERM: {{term}}'''
        return (system, user)
