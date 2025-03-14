# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
import nest_asyncio

from .abc import EntitySource

from collections import defaultdict

from typing import (
    Any,
    Optional,
)

from ...constants import (
    WIKIDATA_SEARCH_API_BASE_URL,
)

from kif_lib.namespace.wikidata import Wikidata

LOG = logging.getLogger(__name__)

nest_asyncio.apply()


class WikidataEntitySource(
    EntitySource,
    source_name='wikidata',
):

    #: The default Wikidata API endpoint IRI.
    _default_api_uri: str = WIKIDATA_SEARCH_API_BASE_URL

    _api_uri: Optional[str]

    _default_prefix_item_iri = str(Wikidata.WD)
    _default_prefix_property_iri = str(Wikidata.WDT)

    def __init__(
        self, source_name: str, uri: Optional[str] = None, **kwargs: Any
    ) -> None:
        assert source_name == self.source_name

        self.default_prefix_item_iri = self._default_prefix_item_iri
        self.default_prefix_property_iri = self._default_prefix_property_iri

        self._api_uri = uri or self._default_api_uri
        super().__init__(source_name, **kwargs)

    def _lookup_item_search(
        self, label: str, limit=10
    ) -> dict[str, Optional[Any]]:
        return self.__get_entity_from_label(label, 'item', limit)

    def _lookup_property_search(
        self, label, limit
    ) -> dict[str, Optional[Any]]:
        return self.__get_entity_from_label(label, 'property', limit)

    async def __get_entity_from_label(
        self, label: str, type: str, limit: int
    ) -> dict[str, Optional[Any]]:
        import httpx

        assert type in ['item', 'property']

        template = (
            f'{self._api_uri}'
            '?action=wbsearchentities'
            '&search={label}'
            '&language=en'
            '&format=json'
            '&limit={limit}'
            '&type={type}'
        )

        url = template.format_map(
            defaultdict(str, label=label, limit=limit, type=type)
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                data = response.json()
                search_result = data.get('search', [])
                return [
                    {
                        'id': result['id'],
                        'iri': result.get(
                            'concepturi',
                            f"http://www.wikidata.org/entity/{result['id']}",
                        ),
                        'label': result.get('label', ''),
                        'description': result.get('description', ''),
                    }
                    for result in search_result
                ]
        except httpx.RequestError as e:
            LOG.error('Request error for label "%s": %s', label, e)
            raise e
        except httpx.HTTPStatusError as e:
            LOG.error(
                (
                    f'HTTP status error for label `{label}`: '
                    f'{e.response.status_code} - {e.response.text}'
                )
            )
            raise e
        except Exception as e:
            LOG.error(f'Unexpected error for label `{label}`: {e}')
            raise e

    def _parse_entity(self, id: str, entity_type='item') -> Optional[str]:
        import re

        fl = 'Q'
        if entity_type == 'property':
            fl = 'P'
        match = re.search(fr'{fl}\d+', id)

        if match:
            return match.group()
        return None
