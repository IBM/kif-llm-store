# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

from kif_lib import Item
from kif_lib.typing import (
    Any,
    Iterator,
    Optional,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .abc import Disambiguator


# from .abc import Disambiguator


LOG = logging.getLogger(__name__)


class LLM_Disambiguator(Disambiguator, disambiguator_name='llm'):

    _model: BaseChatModel
    _sentence_term_template: str
    _textual_context: Optional[str]

    def __init__(
        self,
        disambiguator_name: str,
        model: BaseChatModel,
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
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def sentence_term_template(self) -> str:
        return self._sentence_term_template

    @sentence_term_template.setter
    def sentence_term_template(self, value):
        self._sentence_term_template = value

    async def _disambiguate(
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
            results = await self._source.lookup_item_search(
                label,
                limit,
            )

            if not results:
                LOG.info(f'No item was found for the label `{label}`.')
                return label, None

            label_from_source, item_id = await self._llm_entity_disambiguation(
                label, results, 'item'
            )

            if item_id:
                if label_from_source:
                    return label_from_source, item_id
                return label, item_id
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
            label_from_source = label
            if candidates and len(candidates) > 0:
                c_prompt = ''
                for candidate in candidates:
                    c_prompt += f'Candidate: {candidate["id"]}\n'
                    if 'label' in candidate:
                        c_prompt += f'Label: {candidate["label"]}\n'
                    if 'description' in candidate:
                        c_prompt += f'Description: {candidate["description"]}\n\n'  # noqa E501

                s_template, u_template = self._default_prompt()

                promp_template = ChatPromptTemplate.from_messages(
                    [('system', s_template), ('human', u_template)]
                )
                from langchain_core.runnables import RunnableLambda

                # TODO: fix to work with any knowledge source than Wikidata

                to_entity = RunnableLambda(
                    lambda id: self._source.parse_entity(id)
                )

                debug = RunnableLambda(
                    lambda entry: (LOG.info(entry), entry)[1]
                )

                chain = (
                    promp_template
                    | debug
                    | self.model
                    | debug
                    | StrOutputParser()
                    | debug
                    | to_entity
                    | debug
                )

                # sentence = self.sentence_term_template.format(term=label)
                sentence = self.sentence_term_template
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
                        label_from_source = entity_info[0]['label']
                        entity_id = entity_info[0]['iri']

            if not entity_id:
                LOG.info(f'No entity was found to the label `{label}`.')
            return label_from_source, entity_id
        except Exception as e:
            raise e

    def _default_prompt(self) -> Tuple[str, str]:
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
candidate ID, if the candidate is a valid match for both \
the label/description and the context of the sentence. Reason about the \
answer, checking whether any candidate should actually be returned. If no \
candidate fits the context, return an empty string. Please, respond \
concisely, truthfully and with no further explanation.\

TERM: {{term}}'''
        return (system, user)
