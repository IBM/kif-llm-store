# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import re

import httpx
from typing_extensions import Iterator, override

from ..context_generator import ContextGenerator


class Google_EpisodesPlugin(
    ContextGenerator.Plugin,
    plugin_name='google-episodes',
    plugin_patterns=[r'.*https?://www\.google\.com/search\?q='],
    plugin_priority=101,
):
    """Plugin to get the number of episodes from Google."""

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')

        total_episodes_element = soup.find("span", class_="FCUp0c rQMQod")
        if total_episodes_element:
            total_episodes_text = total_episodes_element.get_text()
            m = re.search(r'\d+', total_episodes_text)
            if m:
                yield f'The number of episodes of {{subject}} is {m.group(0)}.'
            else:
                match = re.search(
                    r'\b(\d+)\s+episodes\b', soup.get_text(), re.IGNORECASE
                )
                if match:
                    yield f'The number of episodes of {{subject}} is {match.group(1)}.'
        else:
            match = re.search(
                r'\b(\d+)\s+episodes\b', soup.get_text(), re.IGNORECASE
            )
            if match:
                yield f'The number of episodes of {{subject}} is {match.group(1)}.'
