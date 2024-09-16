# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from typing import Dict, Tuple

from kif_lib import Item, Property
from kif_lib.typing import Any, Callable, Iterator, Optional
from kif_lib.vocabulary import wd

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


class BaselineDisambiguator(Disambiguator, disambiguator_name='baseline'):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                await self._baseline_entity_disambiguation(
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
                await self._baseline_entity_disambiguation(
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

    async def _baseline_entity_disambiguation(
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
        search_results = await fetch_wikidata_entities(label, url, parser_fn)
        label_from_wikidata = label
        if search_results:
            w_id = search_results[0].get('id')
            label_from_wikidata = search_results[0].get('label', label)

        if not w_id:
            LOG.info(f'No Wikidata entity was found to the label `{label}`.')
        return label_from_wikidata, w_id
