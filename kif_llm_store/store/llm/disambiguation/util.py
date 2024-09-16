# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import List, Tuple

import httpx
from kif_lib.typing import Callable, Optional

from ..constants import WID, Label

LOG = logging.getLogger(__name__)


async def fetch_wikidata_entities(
    label: str,
    url: str,
    parser_fn: Optional[Callable[..., Tuple[Label, WID]]] = None,
    **kwargs,
) -> List:
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
