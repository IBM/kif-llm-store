# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import re
from pathlib import Path

from typing_extensions import (
    Any,
    cast,
    Collection,
    Optional,
    Pattern,
    Sequence,
    TypeAlias,
    Union,
)

from . import ignored

THeaders: TypeAlias = dict[str, str]
TPath: TypeAlias = Union[Path, str]
TPatterns: TypeAlias = Sequence[str]


class Options:
    """Context generator options."""

    @classmethod
    def _getenv(cls, name: str, default: Any) -> Any:
        return os.getenv(f'CONTEXT_GENERATOR_{name}', default)

    @classmethod
    def _getenvpath(cls, name: str, default: TPath) -> Path:
        return Path(cls._getenv(name, default))

    __slots__ = (
        '_cache_dir',
        '_nltk_data_dir',
        '_results_cache_dir',
        '_wapi_furl_cache',
        '_sentence_transformer_cache_dir',
        '_sentence_transformer_model',
        '_http_headers',
        '_language_tags',
        '_extra_language_tags',
        '_url_patterns_to_skip',
        '_extra_url_patterns_to_skip',
        '_ranking_key',
        '_follow_redirects',
        '_split_paragraphs',
        '_overwrite_cached_results',
        '_use_cached_results',
    )

    _cache_dir: Path
    _nltk_data_dir: Path
    _results_cache_dir: Path
    _wapi_furl_cache: Path
    _sentence_transformer_cache_dir: Path
    _sentence_transformer_model: str
    _http_headers: THeaders
    _language_tags: Collection[str]
    _extra_language_tags: Collection[str]
    _url_patterns_to_skip: Sequence[Pattern[str]]
    _extra_url_patterns_to_skip: Sequence[Pattern[str]]
    _ranking_key: str
    _follow_redirects: bool
    _split_paragraphs: bool
    _overwrite_cached_results: bool
    _use_cached_results: bool

    def __init__(
            self,
            cache_dir: Optional[TPath] = None,
            nltk_data_dir: Optional[TPath] = None,
            results_cache_dir: Optional[TPath] = None,
            wapi_furl_cache: Optional[TPath] = None,
            sentence_transformer_cache_dir: Optional[TPath] = None,
            sentence_transformer_model: Optional[str] = None,
            http_headers: Optional[THeaders] = None,
            language_tags: Optional[Collection[str]] = None,
            extra_language_tags: Optional[Collection[str]] = None,
            url_patterns_to_skip: Optional[TPatterns] = None,
            extra_url_patterns_to_skip: Optional[TPatterns] = None,
            ranking_key: Optional[str] = None,
            follow_redirects: Optional[bool] = None,
            split_paragraphs: Optional[bool] = None,
            overwrite_cached_results: Optional[bool] = None,
            use_cached_results: Optional[bool] = None,
    ):
        self.cache_dir = cast(Path, cache_dir)\
            or self._getenvpath('CACHE_DIR', './_ctxgen')
        self.nltk_data_dir = cast(Path, nltk_data_dir)\
            or self._getenvpath('NLTK_DATA_DIR', self.cache_dir / 'nltk')
        self.results_cache_dir = cast(Path, results_cache_dir)\
            or self._getenvpath(
                'RESULTS_CACHE_DIR', self.cache_dir / 'results')
        self.wapi_furl_cache = cast(Path, wapi_furl_cache)\
            or self._getenvpath('WAPI_FURL_CACHE', self.cache_dir / 'furl.tsv')
        self.sentence_transformer_cache_dir =\
            cast(Path, sentence_transformer_cache_dir)\
            or self._getenvpath(
                'SENTENCE_TRANSFORMER_CACHE_DIR',
                self.cache_dir / 'sentence_transformer')
        self.sentence_transformer_model = (
            sentence_transformer_model
            if sentence_transformer_model is not None
            else self._getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
        )
        self.http_headers = http_headers or {}
        self.language_tags =\
            language_tags or ignored.ENGLISH_LANGUAGE_TAGS
        self.extra_language_tags = extra_language_tags or ()
        self.url_patterns_to_skip =\
            url_patterns_to_skip\
            or ignored.get_ignored_url_patterns(
                self.language_tags + self.extra_language_tags)  # type: ignore
        self.extra_url_patterns_to_skip =\
            extra_url_patterns_to_skip or ()  # type: ignore
        self.ranking_key = (
            ranking_key
            if ranking_key is not None
            else self._getenv('RANKING_KEY', '')
        )
        self.follow_redirects = (
            follow_redirects
            if follow_redirects is not None
            else self._getenv('FOLLOW_REDIRECTS', True)
        )
        self.split_paragraphs = (
            split_paragraphs
            if split_paragraphs is not None
            else self._getenv('FOLLOW_REDIRECTS', False))
        self.overwrite_cached_results = (
            overwrite_cached_results
            if overwrite_cached_results is not None
            else self._getenv('OVERWRITE_CACHED_RESULTS', False))
        self.use_cached_results = (
            use_cached_results
            if use_cached_results is not None
            else self._getenv('use_CACHED_RESULTS', True))

    def __str__(self):
        def it():
            cls = self.__class__
            for name in sorted(dir(cls)):
                if name[0] != '_':
                    value = getattr(cls, name)
                    if isinstance(value, property):
                        yield f'{name}: {getattr(self, name)}'
        return '\n'.join(it())

    @property
    def cache_dir(self) -> Path:
        """The path to context-generator's cache directory."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, path: TPath):
        self._cache_dir = Path(path)

    @property
    def nltk_data_dir(self) -> Path:
        """The path to NLTK data directory."""
        return self._nltk_data_dir

    @nltk_data_dir.setter
    def nltk_data_dir(self, path: TPath):
        self._nltk_data_dir = Path(path)

    @property
    def results_cache_dir(self) -> Path:
        """The path to results cache directory."""
        return self._results_cache_dir

    @results_cache_dir.setter
    def results_cache_dir(self, path: TPath):
        self._results_cache_dir = Path(path)

    @property
    def wapi_furl_cache(self) -> Path:
        """The path to the external URL cache file."""
        return self._wapi_furl_cache

    @wapi_furl_cache.setter
    def wapi_furl_cache(self, path: TPath):
        self._wapi_furl_cache = Path(path)

    @property
    def sentence_transformer_cache_dir(self) -> Path:
        """The path to sentence transformer cache directory."""
        return self._sentence_transformer_cache_dir

    @sentence_transformer_cache_dir.setter
    def sentence_transformer_cache_dir(self, path: TPath):
        self._sentence_transformer_cache_dir = Path(path)

    @property
    def sentence_transformer_model(self) -> str:
        """The model used computing text similarity."""
        return self._sentence_transformer_model

    @sentence_transformer_model.setter
    def sentence_transformer_model(self, name: str):
        self._sentence_transformer_model = name

    @property
    def http_headers(self) -> THeaders:
        """The base HTTP headers to use."""
        return self._http_headers

    @http_headers.setter
    def http_headers(self, headers: THeaders):
        headers = copy.copy(headers)
        if 'User-Agent' not in headers:
            headers['User-Agent'] = self._getenv('USER_AGENT', 'Mozilla/5.0')
        self._http_headers = headers

    @property
    def language_tags(self) -> Collection[str]:
        """The language tags to consider."""
        return self._language_tags

    @language_tags.setter
    def language_tags(self, tags: Collection[str]):
        self._language_tags = list(sorted(set(
            map(lambda s: s.lower(), tags))))

    @property
    def extra_language_tags(self) -> Collection[str]:
        """Extra language tags to consider."""
        return self._extra_language_tags

    @extra_language_tags.setter
    def extra_language_tags(self, tags: Collection[str]):
        self._extra_language_tags = list(sorted(set(
            map(lambda s: s.lower(), tags))))

    @property
    def url_patterns_to_skip(self) -> Sequence[Pattern[str]]:
        """The URL patterns to skip."""
        return self._url_patterns_to_skip

    @url_patterns_to_skip.setter
    def url_patterns_to_skip(self, patterns: TPatterns):
        self._url_patterns_to_skip = tuple(map(re.compile, patterns))

    @property
    def extra_url_patterns_to_skip(self) -> Sequence[Pattern[str]]:
        """Extra URL patterns to skip."""
        return self._extra_url_patterns_to_skip

    @extra_url_patterns_to_skip.setter
    def extra_url_patterns_to_skip(self, patterns: TPatterns):
        self._extra_url_patterns_to_skip = tuple(map(re.compile, patterns))

    @property
    def ranking_key(self) -> str:
        """The key used for ranking the results."""
        return self._ranking_key

    @ranking_key.setter
    def ranking_key(self, key: str):
        self._ranking_key = key

    @property
    def follow_redirects(self) -> bool:
        """Whether to follow redirects."""
        return self._follow_redirects

    @follow_redirects.setter
    def follow_redirects(self, flag: bool):
        self._follow_redirects = flag

    @property
    def split_paragraphs(self) -> bool:
        """Whether to split paragraphs into sentences."""
        return self._split_paragraphs

    @split_paragraphs.setter
    def split_paragraphs(self, flag: bool):
        self._split_paragraphs = flag

    @property
    def overwrite_cached_results(self) -> bool:
        """Whether overwrite the cached results."""
        return self._overwrite_cached_results

    @overwrite_cached_results.setter
    def overwrite_cached_results(self, flag: bool):
        self._overwrite_cached_results = flag

    @property
    def use_cached_results(self) -> bool:
        """Whether to use the cached results."""
        return self._use_cached_results

    @use_cached_results.setter
    def use_cached_results(self, flag: bool):
        self._use_cached_results = flag
