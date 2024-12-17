# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

from kif_lib import Item, Property
from kif_lib.typing import Optional

from .constants import (
    Label,
)

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

    async def _label_to_item(
        self,
        label: str,
        limit: Optional[int] = 10,
    ) -> Tuple[Label, Optional[Item]]:
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
            results = await self._target_store.get_items_from_label(
                label, limit
            )
            if not results:
                LOG.info(f'No item was found for the label `{label}`.')
                return label, None

            label_from_target_store = label
            iri = results[0].get('iri')

            if not iri:
                LOG.info(f'No item was found for the label `{label}`.')
                return label, None

            label_from_target_store = results[0].get('label')
            return label_from_target_store, iri

        except Exception as e:
            LOG.error(
                f'No item was found for the label `{label}`: {e.__str__()}.'
            )
            raise e

    async def _label_to_property(
        self,
        label: str,
        limit: Optional[int] = 10,
    ) -> Tuple[Label, Optional[Property]]:
        """
        Disambiguates a label by retrieving the Top-1 ranked
            Property ID.

        Args:
            label (str): The label to disambiguate.
            limit (int): Maximum number of results to retrieve per property
              type.
        Returns:
            Optional[Property]: A property if found, otherwise None.
        """

        try:
            result = await self._target_store.get_properties_from_label(
                label, limit
            )

            if not result:
                LOG.info(f'No property was found for the label `{label}`.')
                return label, None

            label_from_target_store = label
            property = result[0].get('iri')

            if not property:
                LOG.info(f'No property was found for the label `{label}`.')
                return label, None

            label_from_target_store = result[0].get('label')
            return label_from_target_store, property

        except Exception as e:
            LOG.error(
                f'No property was found for the label `{label}`: {e.__str__()}.'  # noqa: E501
            )
            raise e
