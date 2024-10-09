# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0


from collections import defaultdict
import logging
from typing import Any, Dict, List, Tuple

import httpx
from kif_lib.typing import Callable, Optional

from ..constants import DEFAULT_WIKIDATA_SEARCH_API_TEMPLATE, WID, Label

LOG = logging.getLogger(__name__)


async def fetch_wikidata_entities(
    label: str,
    url_template: str,
    url_template_mapping: Optional[Dict[str, Any]] = None,
    limit=10,
    entity_type='item',
    parser_fn: Optional[Callable[..., Tuple[Label, WID]]] = None,
    **kwargs,
) -> List:

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

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            if parser_fn:
                return parser_fn(data, **kwargs)
            else:
                search_results = data.get('search', [])
                return search_results
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
