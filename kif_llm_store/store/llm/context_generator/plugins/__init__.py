# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from .exchange import (
    ExchangePlugin,
    Google_ExchangePlugin,
    Wikipedia_ExchangePlugin,
    Yahoo_ExchangePlugin,
)
from .google_episodes import Google_EpisodesPlugin
from .imdb_episodes import IMDB_EpisodesPlugin
from .ner_extract import (
    NER_ExtractLocationPlugin,
    NER_ExtractOrganizationPlugin,
    NER_ExtractPersonPlugin,
    NER_ExtractPlugin,
)
from .ner_extract_wikipedia import NER_ExtractWikipediaPlugin
from .wikipedia_episodes import Wikipedia_EpisodesPlugin
from .wikipedia_place_of_death import Wikipedia_PlaceOfDeathPlugin
from .wikipedia_summary import WikipediaSummaryPlugin
from .wikitree_place_of_death import WikiTree_PlaceOfDeathPlugin

__all__ = (
    'Google_EpisodesPlugin',
    'Google_ExchangePlugin',
    'IMDB_EpisodesPlugin',
    'NER_ExtractLocationPlugin',
    'NER_ExtractOrganizationPlugin',
    'NER_ExtractPersonPlugin',
    'NER_ExtractPlugin',
    'NER_ExtractWikipediaPlugin',
    'Wikipedia_EpisodesPlugin',
    'Wikipedia_ExchangePlugin',
    'Wikipedia_PlaceOfDeathPlugin',
    'WikipediaSummaryPlugin',
    'ExchangePlugin',
    'WikiTree_PlaceOfDeathPlugin',
    'Yahoo_ExchangePlugin',
)
