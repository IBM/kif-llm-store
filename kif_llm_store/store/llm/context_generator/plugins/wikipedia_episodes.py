# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import httpx
from typing_extensions import Iterator, override

from ..context_generator import ContextGenerator


class Wikipedia_EpisodesPlugin(
    ContextGenerator.Plugin,
    plugin_name='wikipedia-episodes',
    plugin_patterns=[r'.*https?://(en|es|pt|zh|fr)\.wikipedia\.org/wiki/'],
    plugin_priority=101,
):
    """Plugin to get the number of episodes from IMDB."""

    wikipedia_mapper = {
        'en': 'No. of episodes',
        'es': 'N.º de episodios',
        'pt': 'Episódios',
        'zh': '集数',
        'fr': "Nb. d'épisodes",
    }

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import re

        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')
        infobox = soup.find(['table', 'div'], {'class': 'infobox'})
        protocol_end = self.url.find("://") + 3
        lang_code_end = self.url.find(".", protocol_end)
        language_code = self.url[protocol_end:lang_code_end]

        if infobox:
            for row in infobox.find_all('tr'):
                header = row.find('th')
                if header and (
                    self.wikipedia_mapper.get(language_code)
                    in header.get_text()
                ):
                    episodes_str = row.find('td').get_text(
                        separator=' ', strip=True
                    )

                    total_episodes = re.search(r'\d+(?:,\d+)*', episodes_str)

                    if total_episodes is not None:
                        yield f'The number of episodes of {{subject}} is {total_episodes.group(0).replace(",", "").replace(".", "")}.'
