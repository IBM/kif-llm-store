# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import re
import sys
from collections.abc import Sequence

from typing_extensions import NoReturn, Pattern

from .context_generator import ContextGenerator


def error(*args) -> NoReturn:
    print('error:', *args, file=sys.stderr)
    sys.exit(1)


def parse_headers(
        headers: Sequence[str],
        _re: Pattern[str] = re.compile(r'^([^:]*):(.*)$')
) -> dict[str, str]:
    def it():
        for header in headers:
            m = _re.match(header)
            if m is None:
                error('bad HTTP header:', header)
            yield m.group(1).strip(), m.group(2).strip()
    return dict(it())


def main() -> NoReturn:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='context generator')
    parser.add_argument(
        'target', nargs='*', action='extend',
        help='URL or Wikidata id')
    parser.add_argument(
        '-b', '--bottom', metavar='N', type=int,
        help='show the bottom N results')
    parser.add_argument(
        '--cache-dir', metavar='DIR', type=str,
        help='context generator cache DIR')
    parser.add_argument(
        '--do-not-follow-redirects', action='store_true',
        help='do not follow HTTP redirects')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='expand URLs and do a dry-run')
    parser.add_argument(
        '--expand-urls', action='store_true',
        help='expand URLs and exit')
    parser.add_argument(
        '-H', '--header', metavar='HEADER', action='append', default=[],
        help='add HEADER to HTTP requests')
    parser.add_argument(
        '-k', '--key', metavar='TEXT', type=str,
        help='rank the results by similarity with KEY')
    parser.add_argument(
        '-l', '--language', metavar='TAG', action='append', default=[],
        help='do not skip URLs matching non-English language TAG')
    parser.add_argument(
        '--nltk-data-dir', metavar='DIR', type=str,
        help='NLTK data DIR')
    parser.add_argument(
        '--overwrite-cached-results', action='store_true',
        help='overwrite cached results')
    parser.add_argument(
        '-P', '--plugin', metavar='PLUGIN', action='append', default=[],
        help='try to use PLUGIN')
    parser.add_argument(
        '--results-cache-dir', metavar='DIR', type=str,
        help='results cache DIR')
    parser.add_argument(
        '--sentence-transformer-cache-dir', metavar='DIR', type=str,
        help='Sentence Transformer cache DIR')
    parser.add_argument(
        '--sentence-transformer-model', metavar='NAME', type=str,
        help='use Sentence Transformer model NAME')
    parser.add_argument(
        '--show-options', action='store_true',
        help='show options')
    parser.add_argument(
        '-S', '--skip', metavar='REGEX', action='append', default=[],
        help='skip URLs matching REGEX')
    parser.add_argument(
        '--split-paragraphs', action='store_true',
        help='split paragraphs into sentences (using NLTK)')
    parser.add_argument(
        '-t', '--top', metavar='N', type=int,
        help='show the top N results')
    parser.add_argument(
        '-C', '--use-cached-results', action='store_true',
        help='use cached results (if any)')
    parser.add_argument(
        '--wapi-furl-cache', metavar='FILE', type=str,
        help='Wikidata formatter URL cache FILE')
    args = parser.parse_args()
    ctxgen_options = ContextGenerator.Options(
        cache_dir=args.cache_dir,
        extra_language_tags=args.language,
        extra_url_patterns_to_skip=args.skip,
        follow_redirects=not args.do_not_follow_redirects,
        http_headers=parse_headers(args.header),
        nltk_data_dir=args.nltk_data_dir,
        overwrite_cached_results=args.overwrite_cached_results,
        ranking_key=args.key,
        results_cache_dir=args.results_cache_dir,
        sentence_transformer_cache_dir=args.sentence_transformer_cache_dir,
        sentence_transformer_model=args.sentence_transformer_model,
        split_paragraphs=args.split_paragraphs,
        use_cached_results=args.use_cached_results,
        wapi_furl_cache=args.wapi_furl_cache,
    )
    if args.show_options:
        print(ctxgen_options)
        sys.exit(0)
    ctxgen = ContextGenerator(ctxgen_options)
    targets = ctxgen.wapi_expand(args.target)
    if args.expand_urls:
        for i, url in enumerate(targets, 1):
            print(i, url, sep='\t')
        sys.exit(0)
    plugins = args.plugin if not args.dry_run else ('no-op',)
    results = ctxgen.generate(targets, plugins)
    start, end = 0, len(results)
    if args.top is not None:
        end = min(abs(args.top), end)
    elif args.bottom is not None:
        start = end - min(args.bottom, end)
    for i in range(start, end):
        t = results[i]
        print('--', f'#{i+1}', f"[{t['plugin']}]",
              f"({t['similarity']:.4f})", t['url'], '--')
        print(t['text'])
        print()
    sys.exit(0)


if __name__ == '__main__':
    main()
