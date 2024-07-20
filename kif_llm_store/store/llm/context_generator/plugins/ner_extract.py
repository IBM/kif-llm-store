# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import pathlib

import httpx
from typing_extensions import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Iterator,
    Optional,
    cast,
    override,
)

from ..context_generator import ContextGenerator

STANFORD_NER_ZIP_URL: Final[str] = os.getenv(
    'CONTEXT_MANAGER_STANFORD_NER_ZIP_URL',
    'https://nlp.stanford.edu/software/stanford-ner-4.2.0.zip',
)

STANFORD_NER_TAGGER_ARGS: Final[tuple[str]] = (
    'classifiers/english.all.3class.distsim.crf.ser.gz',
    'stanford-ner.jar',
)


class NER_ExtractPlugin(
    ContextGenerator.FallbackPlugin,
    plugin_name='ner-extract',
    plugin_patterns=['.'],
    plugin_priority=101,
):
    """Gets entity names from web pages."""

    _standford_ner_tagger: ClassVar[Optional[object]] = None
    _entity_kind: ClassVar[Optional[str]] = None

    async def _load_standford_ner_tagger(
        self, client: httpx.AsyncClient
    ) -> object:
        nltk_data_dir = self.context_generator.options.nltk_data_dir
        nltk = self.context_generator._nltk_load(nltk_data_dir)
        assert nltk_data_dir.exists()
        stanford_ner_dir = cast(pathlib.Path, nltk_data_dir / 'stanford-ner')
        if not stanford_ner_dir.exists():
            import zipfile

            stanford_ner_zip = nltk_data_dir / 'stanford-ner.zip'
            if not stanford_ner_zip.exists():
                res = await client.get(
                    STANFORD_NER_ZIP_URL,
                    follow_redirects=self.options.follow_redirects,
                )
                with open(stanford_ner_zip, 'wb') as fp:
                    fp.write(res.content)
            assert stanford_ner_zip.exists()
            with zipfile.ZipFile(stanford_ner_zip, 'r') as zip:
                zip.extractall(stanford_ner_dir)
        assert stanford_ner_dir.exists()
        args = list(
            map(
                lambda p: stanford_ner_dir / 'stanford-ner-2020-11-17' / p,
                STANFORD_NER_TAGGER_ARGS,
            )
        )
        for arg in args:
            assert arg.exists(), f'{arg} does not exist!'
        return nltk.tag.StanfordNERTagger(*map(str, args), encoding='utf-8')

    @override
    async def _do_run(
        self, client: httpx.AsyncClient
    ) -> Iterator['ContextGenerator.Result']:
        if self._standford_ner_tagger is None:
            self.__class__._standford_ner_tagger = (
                await self._load_standford_ner_tagger(client)
            )
        return await super()._do_run(client)

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        if TYPE_CHECKING:
            from nltk.tag import StanfordNERTagger

        recipients = self._search_award_recipients(response)
        if recipients:

            def it_names():
                for recipient in recipients:
                    yield recipient

            return iter(('; '.join(sorted(set(it_names()))),))

        it = self._process_response(response)
        st = cast(Optional['StanfordNERTagger'], self._standford_ner_tagger)
        assert st is not None

        def it_names():
            for text in it:
                tagged: Iterator[tuple[str, str]] = iter(st.tag(text.split()))
                for name, kind in self._join_contiguous(tagged):
                    if self._entity_kind is None or self._entity_kind == kind:
                        yield name

        return iter(('; '.join(sorted(set(it_names()))),))

    def _search_award_recipients(self, response):
        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')
        tables = soup.find_all("table", {'class': "wikitable"})

        recipients = []
        if tables:
            for table in tables:
                headers = table.find_all("th")
                recipient_column_index = -1

                # Encontrar a posição da coluna "Recipients"
                for i, th in enumerate(headers):
                    table_column = th.get_text(strip=True).strip().lower()
                    if (
                        table_column == "Recipient".lower()
                        or table_column == "Player".lower()
                        or table_column == "Winner".lower()
                        or table_column == "Author".lower()
                    ):
                        recipient_column_index = i
                        break

                if recipient_column_index != -1:
                    rows = table.find_all("tr")

                    for row in rows[1:]:
                        cells = row.find_all(["td", "th"])
                        if len(cells) > recipient_column_index:
                            recipients.append(
                                cells[recipient_column_index].get_text(
                                    strip=True
                                )
                            )
        return list(set(recipients))

    def _process_response(self, response: httpx.Response) -> Iterator[str]:
        return super()._process(response)

    def _join_contiguous(
        self, it: Iterator[tuple[str, str]]
    ) -> Iterator[tuple[str, str]]:
        name: str
        tag: str
        while True:
            try:
                it = self._skip_Os(it)
                name, tag = next(it)
            except StopIteration:
                break
            while True:
                try:
                    name1, tag1 = next(it)
                except StopIteration:
                    yield name, tag
                if tag1 == tag:
                    name += ' ' + name1
                else:
                    it = itertools.chain(((name1, tag1),), it)
                    yield name, tag
                    break

    def _skip_Os(
        self, it: Iterator[tuple[str, str]]
    ) -> Iterator[tuple[str, str]]:
        while True:
            try:
                name, tag = next(it)
            except StopIteration:
                break
            if tag != 'O':
                it = itertools.chain(((name, tag),), it)
                break
        return it


class NER_ExtractPersonPlugin(
    NER_ExtractPlugin,
    plugin_name='ner-extract-person',
    plugin_patterns=['.'],
    plugin_priority=101,
):
    """Get person names from Web pages."""

    _entity_kind: ClassVar[str] = 'PERSON'


class NER_ExtractOrganizationPlugin(
    NER_ExtractPlugin,
    plugin_name='ner-extract-organization',
    plugin_patterns=['.'],
    plugin_priority=101,
):
    """Gets organization names from Web pages."""

    _entity_kind: ClassVar[str] = 'ORGANIZATION'


class NER_ExtractLocationPlugin(
    NER_ExtractPlugin,
    plugin_name='ner-extract-location',
    plugin_patterns=['.'],
    plugin_priority=101,
):
    """Gets location names from Web pages."""

    _entity_kind: ClassVar[str] = 'LOCATION'
