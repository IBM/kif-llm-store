# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import re

import httpx
from typing_extensions import ClassVar, Iterator, Pattern, override

from .ner_extract import NER_ExtractPlugin


class NER_ExtractWikipediaPlugin(
    NER_ExtractPlugin,
    plugin_name='ner-extract-wikipedia',
    plugin_patterns=[r'https?://([a-z][a-z])\.wikipedia\.org/'],
    plugin_priority=101,
):
    """Get entity names from Wikipedia pages."""

    _section_re: ClassVar[Pattern[str]] = re.compile(r'^h[1-6]')

    @override
    def _process_response(self, response: httpx.Response) -> Iterator[str]:
        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')
        for x in soup(('script', 'style')):
            x.extract()  # delete
        for x in soup.find_all('div', {'class': 'reflist'}):
            x.extract()  # delete
        content = soup.find('div', {'id': 'mw-content-text'})
        yield '\n'.join(map(lambda x: x.get_text(), content.find_all('ul')))
        yield '\n'.join(map(lambda x: x.get_text(), content.find_all('table')))
