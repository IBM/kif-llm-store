# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import httpx
from typing_extensions import Iterator, override

from ..context_generator import ContextGenerator


class WikiTree_PlaceOfDeathPlugin(
    ContextGenerator.Plugin,
    plugin_name='wikitree-place-of-death',
    plugin_patterns=[r'.*https?://(www\.)?wikitree\.com/wiki/'],
    plugin_priority=101,
):
    """Plugin to get the person place of death from WikiTree."""

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')

        death_place = soup.find('span', itemprop='deathPlace')
        if death_place:
            death_place_name = death_place.find(
                'span', itemprop='name'
            ).get_text(strip=True)
            yield f'The city where {{subject}} is in the following sentence: {death_place_name}.'
