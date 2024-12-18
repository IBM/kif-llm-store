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

from ..constants import (
    DBPEDIA_SEARCH_API_BASE_URL,
)

from kif_lib.namespace.dbpedia import DBpedia

LOG = logging.getLogger(__name__)

nest_asyncio.apply()


class DBpediaEntitySource(
    EntitySource,
    source_name='dbpedia',
):

    #: The default DBpedia API endpoint IRI.
    _default_api_uri: str = DBPEDIA_SEARCH_API_BASE_URL

    _api_uri: Optional[str]

    _default_prefix_item_iri = str(DBpedia.RESOURCE)
    _default_prefix_property_iri = str(DBpedia.PROPERTY)

    def __init__(
        self, source_name: str, uri: Optional[str] = None, **kwargs: Any
    ) -> None:
        assert source_name == self.source_name

        self._api_uri = uri or self._default_api_uri
        super().__init__(source_name, **kwargs)

    async def _get_items_from_label(
        self, label: str, limit=10
    ) -> dict[str, Optional[Any]]:
        return await self.__get_entity_from_label(label, 'item', limit)

    async def _get_properties_from_label(
        self, label, limit
    ) -> dict[str, Optional[Any]]:
        return await self.__get_entity_from_label(label, 'property', limit)

    async def __get_entity_from_label(
        self, label: str, type: str, limit: int
    ) -> dict[str, Optional[Any]]:
        import httpx
        import re

        assert type in ['item', 'property']

        template = (
            f'{self._api_uri}'
            '?query={label}'
            '&format=JSON'
            '&maxResults={limit}'
        )

        url = template.format_map(defaultdict(str, label=label, limit=limit))

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                data = response.json()
                search_result = data.get('docs', [])
                a = [
                    {
                        'id': result.get('id', [''])[0],
                        'iri': result.get('resource', [''])[0],
                        'label': re.sub(
                            r"<.*?>", "", result.get('label', [''])[0]
                        ),
                        'description': re.sub(
                            r"<.*?>", "", result.get('comment', [''])[0]
                        ),
                    }
                    for result in search_result
                ]

                return a
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
