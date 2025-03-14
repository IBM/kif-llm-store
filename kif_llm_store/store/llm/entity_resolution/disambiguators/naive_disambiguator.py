# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

from kif_lib import Item
from kif_lib.typing import Optional

from .abc import Disambiguator

LOG = logging.getLogger(__name__)


class NaiveDisambiguator(Disambiguator, disambiguator_name='naive'):

    def __init__(
        self,
        disambiguator_name: str,
        *args,
        **kwargs,
    ):
        assert disambiguator_name == self.disambiguator_name
        super().__init__(*args, **kwargs)

    async def _disambiguate(
        self,
        label: str,
        limit: Optional[int] = 10,
    ) -> Tuple[str, Optional[Item]]:
        """

        Disambiguates a label by retrieving the Top-1 ranked
            Item ID from the target store.

        Args:
            label (str): The label to disambiguate.
            limit (int): Maximum number of results to retrieve per entity type.
        Returns:
            Optional[Item]: A item if found, otherwise None.
        """

        try:
            results = await self._source.lookup_item_search(label, limit)
            if not results:
                LOG.info(f'No item was found for the label `{label}`.')
                return label, None

            label_from_source = label
            iri = results[0].get('iri')

            if not iri:
                LOG.info(f'No item was found for the label `{label}`.')
                return label, None

            label_from_source = results[0].get('label')
            return label_from_source, iri

        except Exception as e:
            LOG.error(
                f'No item was found for the label `{label}`: {e.__str__()}.'
            )
            raise e
