# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import httpx
from typing_extensions import Iterator, override

from ..context_generator import ContextGenerator


class Wikipedia_PlaceOfDeathPlugin(
    ContextGenerator.Plugin,
    plugin_name='wikipedia-place-of-death',
    plugin_patterns=[
        r'.*https?://(en|pl|de|ru|es|uk|az|arz|nl)\.wikipedia\.org/wiki/'
    ],
    plugin_priority=101,
):
    """Plugin to get the place of death from Wikipedia."""

    wikipedia_mapper = {
        'pl': ['śmierci', 'spoczynku'],
        'ru': ['Место смерти', 'Смерть'],
        'de': ['gestorben', 'sterbeort', 'Sterbeort'],
        'es': 'Fallecimiento',
        'uk': 'Помер',
        'nl': 'Overleden',
        'az': 'Vəfat yeri',
        'arz': 'الوفاة',
    }

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import re

        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')
        infobox = soup.find('table', {'class': 'infobox'})

        protocol_end = self.url.find("://") + 3
        lang_code_end = self.url.find(".", protocol_end)
        language_code = self.url[protocol_end:lang_code_end]

        if infobox:
            for row in infobox.find_all('tr'):
                header = row.find('th')
                if not header:
                    header = row.find('td')
                if header and (
                    'Died' in header.get_text() or 'Death' in header.get_text()
                ):
                    death_date = row.find('td').get_text(
                        separator=' ', strip=True
                    )
                    death_date = str(death_date).replace("\xa0", " ")

                    pattern = re.compile(
                        r"([A-Za-z\s-]+),\s*([A-Za-z\s-]+)(?=\s*,|$)|([A-Za-z\s-]+)\s+([A-Za-z\s-]+)$"
                    )

                    match = pattern.findall(death_date)
                    death_place = match[-1] if match else ""
                    if death_place and len(death_place) > 0:
                        death_place = ', '.join(
                            [
                                part.strip()
                                for part in death_place
                                if part.strip()
                            ]
                        )
                        yield f'The city where {{subject}} is in the following sentence: {death_place}.'
                        break
                else:
                    if header:
                        death_list_str = self.wikipedia_mapper.get(
                            language_code
                        )
                        if death_list_str:
                            if not isinstance(death_list_str, list):
                                death_list_str = [death_list_str]

                            for death_str in death_list_str:
                                if death_str in header.get_text():
                                    death_date = row.find('td').get_text(
                                        separator=' ', strip=True
                                    )
                                    death_date = str(death_date).replace(
                                        "\xa0", " "
                                    )
                                    if header.name == 'td':
                                        death_date = row.find_all('td')
                                        if len(death_date) > 1:
                                            death_date = death_date[
                                                1
                                            ].get_text(
                                                separator=' ', strip=True
                                            )

                                    if (
                                        death_date
                                        and len(death_date) > 0
                                        and (
                                            'Место смерти' == death_str
                                            or 'Vəfat yeri' == death_str
                                        )
                                    ):
                                        yield f'The city where {{subject}} is in the following sentence (translate from {language_code} to en): {death_date}.'
                                        break

                                    elif (
                                        death_date
                                        and len(death_date) > 0
                                        and 'Помер' == death_str
                                    ):
                                        pattern = re.compile(
                                            r"\(\d+ років\)\s*(.*)"
                                        )
                                        match = pattern.search(death_date)

                                        if match:
                                            death_place = (
                                                match.group(1).strip()
                                                if match
                                                else ""
                                            )
                                            if (
                                                death_place
                                                and len(death_place) > 0
                                            ):
                                                yield f'The city where {{subject}} is in the following sentence (translate from {language_code} to en): {death_place}.'
                                                break
                                        else:
                                            pattern = re.compile(
                                                r"\(\d+ рік\)\s*(.*)"
                                            )
                                            match = pattern.search(death_date)

                                            if match:
                                                death_place = (
                                                    match.group(1).strip()
                                                    if match
                                                    else ""
                                                )
                                                if (
                                                    death_place
                                                    and len(death_place) > 0
                                                ):
                                                    yield f'The city where {{subject}} is in the following sentence (translate from {language_code} to en): {death_place}.'
                                                    break
                                    elif (
                                        death_date
                                        and len(death_date) > 0
                                        and 'Смерть' == death_str
                                    ):
                                        pattern = re.compile(
                                            r"\(\d+ год\)\s*(.*)"
                                        )
                                        match = pattern.search(death_date)
                                        if not match:
                                            pattern = re.compile(
                                                r"\(\d+ лет\)\s*(.*)"
                                            )
                                            match = pattern.search(death_date)

                                        if match:
                                            death_place = (
                                                match.group(1).strip()
                                                if match
                                                else ""
                                            )
                                            if (
                                                death_place
                                                and len(death_place) > 0
                                            ):
                                                yield f'The city where {{subject}} is in the following sentence (translate from {language_code} to en): {death_place}.'
                                                break

                                    else:
                                        death_place = death_date.strip()

                                        if (
                                            death_place
                                            and len(death_place) > 0
                                        ):
                                            yield f'The city where {{subject}} is in the following sentence: {death_date}.'
                                            break
