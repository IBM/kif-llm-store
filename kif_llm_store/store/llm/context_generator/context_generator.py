# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
import copy
import itertools
import json
import logging
import os
import pathlib
import re
import types
from collections.abc import Awaitable, Mapping, Sequence, Set

import httpx
from typing_extensions import (
    Any,
    Callable,
    ClassVar,
    Final,
    Iterable,
    Iterator,
    Optional,
    Pattern,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
    cast,
    override,
)

from .options import Options as Options_

T = TypeVar('T')
JSON: TypeAlias = Mapping[str, Any]
URL: TypeAlias = str
WID: TypeAlias = str


class ContextGenerator:
    """Context generator."""

    #: Module-level logger.
    _logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    #: The NLTK module.
    _nltk: ClassVar[Optional[types.ModuleType]] = None

    #: Alias for :class:`Options`.
    Options: ClassVar[type['Options_']] = Options_

    class Result(TypedDict):
        """Context generator result."""

        #: The result text.
        text: str

        #: The target URL.
        url: URL

        #: The name of the plugin that generated this result.
        plugin: str

        #: The similarity rank of this result.
        similarity: float

    class Plugin(abc.ABC):
        """Abstract base class for context generator plugins."""

        #: Plugin registry.
        registry: ClassVar[dict[str, type['ContextGenerator.Plugin']]] = {}

        #: The name of this plugin
        name: ClassVar[str]

        #: The URI patterns of this context generator.
        patterns: ClassVar[Sequence[Pattern[str]]]

        #: Priority.
        priority: int

        @classmethod
        def _register(
            cls,
            plugin: type['ContextGenerator.Plugin'],
            plugin_name: str,
            plugin_patterns: Sequence[str],
            plugin_priority: int,
        ):
            plugin.name = plugin_name
            plugin.patterns = list(
                map(lambda pat: re.compile(pat), plugin_patterns)
            )
            plugin.priority = plugin_priority
            cls.registry[plugin.name] = plugin

        @classmethod
        def _match(
            cls,
            url: URL,
            plugins: Optional[
                Sequence[type['ContextGenerator.Plugin']]
            ] = None,
        ) -> Optional[type['ContextGenerator.Plugin']]:
            if not plugins:
                plugins = sorted(
                    cls.registry.values(), key=lambda p: p.priority
                )
            for plugin_class in plugins:
                for pattern in plugin_class.patterns:
                    if pattern.match(url):
                        return plugin_class
            return None

        @classmethod
        def __init_subclass__(
            cls,
            plugin_name: str,
            plugin_patterns: Sequence[str],
            plugin_priority: int,
        ):
            cls._register(cls, plugin_name, plugin_patterns, plugin_priority)

        __slots__ = (
            'context_generator',
            'client',
            'url',
            'options',
        )

        #: The parent context generator.
        context_generator: 'ContextGenerator'

        #: The target URL.
        url: URL

        #: Plugin options.
        options: Options_

        def __init__(
            self,
            context_generator: 'ContextGenerator',
            url: URL,
            options: Options_,
        ):
            self.context_generator = context_generator
            self.url = url
            self.options = copy.deepcopy(options)

        @property
        def _logger(self) -> logging.Logger:
            return self.context_generator._logger

        @property
        def _url_hash(self) -> str:
            import hashlib

            return hashlib.sha256(self.url.encode('utf-8')).hexdigest()

        @property
        def _cached_results_path(self) -> pathlib.Path:
            return (
                self.options.results_cache_dir
                / f'{self.name}-{self._url_hash}'
            )

        async def _load_cached_results(
            self,
        ) -> Optional[Sequence['ContextGenerator.Result']]:
            if self._cached_results_path.exists():
                import aiofiles

                async with aiofiles.open(self._cached_results_path) as fp:
                    results = []
                    async for line in fp:
                        line = line.strip()
                        if line[0] == '#':
                            continue
                        results.append(json.loads(line))
                    return results
            return None

        async def _overwrite_cached_results(
            self, results: Iterator['ContextGenerator.Result']
        ) -> Iterator['ContextGenerator.Result']:
            import aiofiles

            if not self.options.results_cache_dir.exists():
                self.options.results_cache_dir.mkdir(
                    parents=True, exist_ok=True
                )
            saved, results = itertools.tee(results)
            count = 0
            async with aiofiles.open(self._cached_results_path, 'w') as fp:
                await fp.write(f'# {self.url}\n')
                for res in results:
                    await fp.write(json.dumps(res))
                    await fp.write('\n')
                    count += 1
            self._logger.info('%s: cached %d results', self.name, count)
            return saved

        async def _run(
            self, client: httpx.AsyncClient
        ) -> Iterator['ContextGenerator.Result']:
            results: Optional[Iterator['ContextGenerator.Result']] = None
            if (
                not self.options.overwrite_cached_results
                and self.options.use_cached_results
            ):
                seq = await self._load_cached_results()
                if seq is not None:
                    results = iter(seq)
            if results is None:
                results = await self._do_run(client)
            assert results is not None
            if self.options.overwrite_cached_results:
                results = await self._overwrite_cached_results(results)
            return results

        async def _do_run(
            self, client: httpx.AsyncClient
        ) -> Iterator['ContextGenerator.Result']:
            res = await client.get(
                self.url, follow_redirects=self.options.follow_redirects
            )
            res.raise_for_status()
            it = self._process(res)
            if self.options.split_paragraphs:
                it = self.context_generator._nltk_split_paragraphs(it)
            return map(
                lambda text: {
                    'text': text,
                    'url': self.url,
                    'plugin': self.name,
                    'similarity': 1.0,
                },
                it,
            )

        @abc.abstractmethod
        def _process(self, response: httpx.Response) -> Iterator[str]:
            raise NotImplementedError

    class NoOpPlugin(
        Plugin, plugin_name='no-op', plugin_patterns=[], plugin_priority=0
    ):
        """The built-in no-op plugin."""

        @override
        async def _run(
            self, client: httpx.AsyncClient
        ) -> Iterator['ContextGenerator.Result']:
            self._logger.info('%s: %s', self.name, self.url)
            await asyncio.sleep(0)
            return iter(())

        @override
        def _process(self, response: httpx.Response) -> Iterator[str]:
            raise RuntimeError('should not get here')

    class FallbackPlugin(
        Plugin,
        plugin_name='fallback',
        plugin_patterns=('.'),  # matches anything
        plugin_priority=99,
    ):
        """The builtin fallback plugin."""

        @override
        def _process(self, response: httpx.Response) -> Iterator[str]:
            import bs4

            soup = bs4.BeautifulSoup(response.text, features='html.parser')
            for script in soup(('script', 'style')):
                script.decompose()  # delete

            for x in soup.find_all('div', {'class': 'reflist'}):
                x.decompose()  # delete
            for x in soup.find_all('div', {'class': 'catlinks'}):
                x.decompose()  # delete

            if soup.body is not None:
                for par in soup.body.get_text(' ', strip=True).split('\n\n'):
                    par = par.strip()
                    if par:
                        yield par

    __slots__ = (
        'options',
        '_wapi_furl_cache',
    )

    #: Context generator options.
    options: Options_

    #: Cached Wikidata formatter URLs (loaded from local cache).
    _wapi_furl_cache: dict[WID, URL]

    def __init__(self, options: Optional[Options_] = None):
        self.options = options or Options_()
        if self.__class__._nltk is None:
            self.__class__._nltk = self._nltk_load(self.options.nltk_data_dir)
        self._wapi_furl_cache = self._wapi_furl_cache_load()

    def match(self, url: URL, plugins: Sequence[str] = ()) -> Plugin:
        """Instantiates a context-generator plugin for scraping `url`.

        Parameters:
           url: URL.
           plugins: Names of the plugins to be considered.

        Returns:
           Plugin.
        """

        def it():
            for name in plugins:
                if name not in self.Plugin.registry:
                    raise ValueError(
                        f'no such context-generator plugin: {name}'
                    )
                yield self.Plugin.registry[name]

        plugin_class = self.Plugin._match(url, list(it())) or self.NoOpPlugin
        return plugin_class(self, url, self.options)

    def generate(
        self, urls: Iterable[URL], plugins: Sequence[str] = ()
    ) -> Sequence[Result]:
        """Generate text context by scraping `urls`.

        Parameters:
           urls: URLs to be scraped.
           plugins: Names of the plugins to be used.

        Returns:
           A collection of :class:`ContextGenerator.Results`.
        """
        return asyncio.run(self.generate_async(urls, plugins))

    async def generate_async(
        self, urls: Iterable[URL], plugins: Sequence[str] = ()
    ) -> Sequence[Result]:
        """Async version of :meth:`ContextGenerator.generate`."""
        urls = self._filter_out_skipped(urls)
        seq = await self._httpx_gather(
            map(lambda x: self.match(x, plugins)._run, urls)
        )
        seq_without_exceptions = cast(
            Sequence[Iterator['ContextGenerator.Result']],
            filter(lambda x: not isinstance(x, BaseException), seq),
        )
        results: Sequence['ContextGenerator.Result'] = list(
            itertools.chain(*seq_without_exceptions)
        )
        if results and self.options.ranking_key:
            results = self._rank(self.options.ranking_key, results)
        return results

    def _filter_out_skipped(self, urls: Iterable[URL]) -> Iterator[URL]:
        for url in urls:
            skipped = any(
                map(
                    lambda pat: pat.match(url),
                    itertools.chain(
                        self.options.url_patterns_to_skip,
                        self.options.extra_url_patterns_to_skip,
                    ),
                )
            )
            if skipped:
                self._logger.info('SKIPPED: %s', url)
                continue
            else:
                yield url

    async def _httpx_gather(
        self,
        tasks: Iterable[Callable[[httpx.AsyncClient], Awaitable[T]]],
        headers: Optional[Mapping[str, str]] = None,
    ) -> Sequence[Union[T, BaseException]]:
        headers = headers or self.options.http_headers
        async with httpx.AsyncClient(headers=headers) as client:
            return await asyncio.gather(
                *map(lambda f: f(client), tasks), return_exceptions=True
            )

    ########
    # NLTK #
    ########

    @classmethod
    def _nltk_load(cls, nltk_data_dir: pathlib.Path):
        os.environ['NLTK_DATA'] = str(nltk_data_dir)
        if True:
            import nltk  # type: ignore

            nltk_data_dir.mkdir(parents=True, exist_ok=True)
            if not (nltk_data_dir / 'tokenizers' / 'punkt').exists():
                nltk.download('punkt', nltk_data_dir)
                nltk.download('punkt_tab', nltk_data_dir)
            return nltk

    def _nltk_split_paragraphs(self, it: Iterator[str]) -> Iterator[str]:
        assert self._nltk is not None
        return itertools.chain(*map(self._nltk.sent_tokenize, it))

    ########
    # Rank #
    ########

    _ranking_model: ClassVar[Optional[Any]] = None

    def _rank(self, key: str, results: Sequence[Result]) -> Sequence[Result]:
        from sentence_transformers import SentenceTransformer

        if self._ranking_model is None:
            self.__class__._ranking_model = SentenceTransformer(
                self.options.sentence_transformer_model,
                cache_folder=str(self.options.sentence_transformer_cache_dir),
            )
        assert self._ranking_model is not None
        model = cast(SentenceTransformer, self._ranking_model)
        text2result = {t['text']: t for t in results}
        unique_texts = list(text2result.keys())
        enc_key = model.encode(key)
        enc_texts = model.encode(unique_texts)
        sim = model.similarity(enc_key, enc_texts)  # type: ignore

        text2similarity = {
            text: sim[0][i].item() for i, text in enumerate(unique_texts)
        }
        for result in results:
            result['similarity'] = text2similarity[result['text']]

        return sorted(results, key=lambda t: t['similarity'], reverse=True)

    ################
    # Wikidata API #
    ################

    def wapi_expand(self, input: Iterable[Union[WID, URL]]) -> Sequence[URL]:
        """Expands the Wikidata URLs or WIDs in `input` into external URLs.

        Parameters:
           input: URL or Wikidata id.

        Returns:
           A sequence expanded URLs.
        """
        return asyncio.run(self.wapi_expand_async(input))

    async def wapi_expand_async(
        self, input: Iterable[Union[WID, URL]]
    ) -> Sequence[URL]:
        """Async version of :meth:`ContextGenerator.wapi_expand`."""

        def it_normalized():
            for x in input:
                try:
                    wid = self._wapi_normalize_wid(x)
                    yield (wid, wid)
                except ValueError:
                    yield (x, None)

        normalized_input = list(it_normalized())
        wids = cast(
            Iterable[WID],
            list(
                filter(
                    lambda x: x is not None,
                    map(lambda p: p[1], normalized_input),
                )
            ),
        )
        tr = await self.wapi_fetch_external_urls_async(wids)

        def it_translated():
            for x, wid in normalized_input:
                if wid is None:
                    yield x
                elif wid in tr:
                    for url in tr[wid]:
                        yield url

        return list(self._filter_out_skipped(it_translated()))

    def wapi_fetch_external_urls(
        self, wids: Iterable[WID]
    ) -> Mapping[WID, Set[URL]]:
        """Fetches the external URLs of `wids`.

        Parameters:
           wids: Wikidata ids.

        Returns:
           A dictionary mapping WIDs to a set of external URLs.
        """
        return asyncio.run(self.wapi_fetch_external_urls_async(wids))

    async def wapi_fetch_external_urls_async(
        self, wids: Iterable[WID]
    ) -> Mapping[WID, Set[URL]]:
        """Async version of
        :meth:`ContextGenerator.wapi_fetch_external_urls`."""

        def process(t: JSON) -> tuple[WID, Set[tuple[WID, str]]]:
            external_ids = self._wapi_fetch_external_urls_filter_xids(t)
            return t['id'], set(external_ids)

        external_ids = dict(await self._wapi_fetch(wids, process))
        pids = set(
            itertools.chain(
                *map(lambda xs: map(lambda p: p[0], xs), external_ids.values())
            )
        )
        tr = await self._wapi_fetch_formatter_urls(pids)
        return dict(
            map(
                lambda x: (
                    x[0],
                    set(
                        map(
                            lambda y: tr.get(y[0], '$1').replace('$1', y[1]),
                            x[1],
                        )
                    ),
                ),
                external_ids.items(),
            )
        )

    def _wapi_fetch_external_urls_filter_xids_sitelinks(self) -> Set[str]:
        tags = set(
            itertools.chain(
                self.options.language_tags, self.options.extra_language_tags
            )
        )

        def it():
            for tag in tags:
                yield tag + 'wikiquote'
                yield tag + 'wiki'

        return set(it())

    def _wapi_fetch_external_urls_filter_xids(
        self, t: JSON
    ) -> Iterator[tuple[WID, str]]:
        for name, entry in t['sitelinks'].items():
            if name in self._wapi_fetch_external_urls_filter_xids_sitelinks():
                yield name, entry['url']
        for entry in itertools.chain(*t['statements'].values()):
            if entry['property']['data_type'] == 'external-id':
                try:
                    yield entry['property']['id'], entry['value']['content']
                except KeyError:
                    pass

    async def _wapi_fetch_formatter_urls(
        self, pids: Iterable[WID]
    ) -> dict[WID, URL]:
        pids = set(pids)
        seen_pids = filter(lambda p: p in self._wapi_furl_cache, pids)
        unseen_pids = filter(lambda p: p not in self._wapi_furl_cache, pids)

        def process(t: JSON) -> tuple[WID, URL]:
            pid = t['id']
            if 'statements' in t:
                if 'P1630' in t['statements']:
                    if t['statements']['P1630']:
                        return (
                            pid,
                            t['statements']['P1630'][0]['value']['content'],
                        )
            return pid, '$1'

        unseen_pairs = list(await self._wapi_fetch(unseen_pids, process))
        self._wapi_furl_cache_add(unseen_pairs)
        return dict(
            itertools.chain(
                map(lambda pid: (pid, self._wapi_furl_cache[pid]), seen_pids),
                unseen_pairs,
            )
        )

    def _wapi_furl_cache_load(self) -> dict[WID, URL]:
        if self.options.wapi_furl_cache.exists():
            with open(self.options.wapi_furl_cache) as fp:

                def it():
                    for line in fp:
                        wid, url = line.split('\t')
                        yield wid.strip(), url.strip()

                return dict(it())
        else:
            return {}

    def _wapi_furl_cache_add(self, entries: Iterable[tuple[WID, URL]]):
        if not self.options.wapi_furl_cache.exists():
            self.options.wapi_furl_cache.parent.mkdir(
                parents=True, exist_ok=True
            )
        with open(self.options.wapi_furl_cache, mode='a') as fp:
            for pid, furl in entries:
                fp.write(f'{pid}\t{furl}\n')
            fp.flush()

    async def _wapi_fetch(
        self, wids: Iterable[WID], process: Callable[[JSON], T]
    ) -> Iterator[T]:
        res = await self._httpx_gather(
            map(
                lambda wid: lambda client: self._wapi_fetch_helper(
                    client, wid, process
                ),
                wids,
            )
        )
        return cast(
            Iterator[T],
            filter(lambda x: not isinstance(x, BaseException), res),
        )

    async def _wapi_fetch_helper(
        self, client: httpx.AsyncClient, wid: WID, process: Callable[[JSON], T]
    ) -> T:
        res = await client.get(
            self._wapi_build_call(wid),
            follow_redirects=self.options.follow_redirects,
        )
        res.raise_for_status()
        return process(res.json())

    _wapi_build_call_base: Final[str] = (
        'https://www.wikidata.org/w/rest.php/wikibase/v0/entities/'
    )

    _wapi_build_call_letter_map: Final[Mapping[str, str]] = {
        'Q': 'items/',
        'P': 'properties/',
    }

    @classmethod
    def _wapi_build_call(cls, wid: WID) -> URL:
        wid = cls._wapi_normalize_wid(wid)
        return (
            cls._wapi_build_call_base
            + cls._wapi_build_call_letter_map[wid[0]]
            + wid
        )

    _wapi_normalize_wid_regex: Final[Pattern[str]] = re.compile(
        r'^(http://www\.wikidata\.org/entity/)?([PQ])?([0-9]+)$'
    )

    @classmethod
    def _wapi_normalize_wid(cls, wid: str) -> WID:
        m = cls._wapi_normalize_wid_regex.match(wid)
        if m is not None:
            _, letter, number = m.groups()
            return (letter or 'Q') + number
        else:
            raise ValueError
