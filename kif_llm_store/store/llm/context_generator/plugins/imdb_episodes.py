# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0


import httpx
from typing_extensions import Iterator, override

from ..context_generator import ContextGenerator


class IMDB_EpisodesPlugin(
    ContextGenerator.Plugin,
    plugin_name='imdb-episodes',
    plugin_patterns=[r'.*https?://www\.imdb\.com/&id='],
    plugin_priority=101,
):
    """Plugin to get the number of episodes from IMDB."""

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import re

        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')
        total_episodes_element = soup.find(
            'span', {'class': 'ipc-title__subtext'}
        )
        if total_episodes_element:
            total_episodes_text = total_episodes_element.get_text()
            m = re.search(r'\d+', total_episodes_text)
            if m:
                yield f'The number of episodes of {{subject}} is {m.group(0)}.'
