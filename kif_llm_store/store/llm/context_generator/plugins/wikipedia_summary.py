# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0


import httpx
from typing_extensions import Iterator, override

from ..context_generator import ContextGenerator


class WikipediaSummaryPlugin(
    ContextGenerator.Plugin,
    plugin_name='wikipedia-summary',
    plugin_patterns=[r'https?://en\.wikipedia\.org/wiki/'],
    plugin_priority=101,
):
    """Gets the summary of Wikipedia page."""

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')
        p = soup.find('p')
        yield p.get_text().strip()
