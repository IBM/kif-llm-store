# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0


import argparse
import json
import logging
import os
import pathlib
import re
import time
from itertools import chain
from typing import Optional

import dotenv
import pandas as pd
from kif_lib import FilterPattern, Property, Store, String
from kif_lib.vocabulary import wd

from kif_llm_store import LLM_Store
from kif_llm_store.store.llm.constants import (
    DEFAULT_ENFORCED_CONTEXT,
    DEFAULT_PROMPT_TEMPLATE,
)
from kif_llm_store.store.llm.context_generator import ContextGenerator

dotenv.load_dotenv()


def countries_border_dict() -> dict[str, list[str]]:
    import requests

    rest_countries = requests.get("https://restcountries.com/v3.1/all")
    countries = rest_countries.json()

    country_name_by_code = {
        country["cca3"]: country["name"]["common"] for country in countries
    }
    countries_border = {}
    for country in countries:
        country_name = country["name"]["common"]
        borders = country.get("borders", [])
        border_names = [country_name_by_code[border] for border in borders]

        countries_border[country_name] = border_names
    return countries_border


# we can access the bordering countries from the api on the function below:
# countries_border = countries_border_dict()

# However, as this is an information that rarely changes we can store it once and use as a dict:
countries_border = {}
with open(
    f'{os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/lm-kbc-2024/countries_land_countries.json")}',
    'r',
) as file:
    countries_border = json.load(file)


def convert_keys_to_lowercase(d):
    if isinstance(d, dict):
        return {k.lower(): convert_keys_to_lowercase(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_lowercase(i) for i in d]
    else:
        return d


countries_border = convert_keys_to_lowercase(countries_border)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_json_l_file(json_l_file):
    with open(json_l_file) as f:
        data = [json.loads(line) for line in f]
    return data


def get_context(
    subject_item,
    predicate_item,
    plugins,
    extra_urls=[],
    keywords=[],
    order_by_plugin=True,
    filter_wiki_urls=False,
    split_paragraphs=True,
    extra_language_tags=[],
    concat_results=False,
) -> Optional[str]:
    context = None
    query = (
        f'{wd.get_entity_label(subject_item)} '
        f'{wd.get_entity_label(predicate_item)}'
    )
    if keywords:
        query += (
            ' ' + ' '.join(keywords) if keywords and len(keywords) > 0 else ''
        )

    ctx_gen: ContextGenerator = ContextGenerator(
        ContextGenerator.Options(
            ranking_key=query,
            split_paragraphs=split_paragraphs,
            overwrite_cached_results=True,
            use_cached_results=True,
            extra_language_tags=extra_language_tags,
            # extra_url_patterns_to_skip=[r'.*https?://(www).quora\.com/.*', r'']
        )
    )
    urls = ctx_gen.wapi_expand([subject_item.value])

    if filter_wiki_urls:
        wiki_pattern = re.compile(
            r"https?://(?:[a-z]{2}\.)?((?:[a-z0-9-]+\.){1,2})wikipedia\.org/.*"
        )
        wiki_urls = [url for url in urls if wiki_pattern.match(url)]
        if len(wiki_urls) > 0:
            urls = wiki_urls

    results: list = ctx_gen.generate(
        urls=urls + extra_urls,
        plugins=plugins,
    )

    if order_by_plugin:

        def sort_key(entry):
            plugin_priority = (
                plugins.index(entry['plugin'])
                if entry['plugin'] in plugins
                else -1
            )

            if entry['url'].startswith('https://en.wikipedia.org'):
                url_priority = 0
            elif 'wikipedia.org' in entry['url']:
                url_priority = 1
            else:
                url_priority = 2

            return (plugin_priority, url_priority)

        results = sorted(results, key=sort_key)
    if results and len(results) > 0:
        context = ''
        for result in results:
            log.write(f'Plugin: {result["plugin"]}\n')
            log.write(f'Similarity: {result["similarity"]}\n')
            log.flush()
            if concat_results:
                context += '; ' + result['text'].replace(
                    '{subject}', wd.get_entity_label(subject_item)
                )
            else:
                context = result['text'].replace(
                    '{subject}', wd.get_entity_label(subject_item)
                )

                break

    return context


def filter(
    llm_name,
    model_id,
    json_data,
    df_prompt_template,
    relation_filter=[],
    use_context=True,
):
    data = json_data
    limit = None
    if not relation_filter:
        relation_filter = []

    subject_item = wd.Q(
        data['SubjectEntityID'].replace('Q', ''), data['SubjectEntity']
    )

    relation = data['Relation'].lower()

    predicate_item: Property = None

    llm: LLM_Store = Store(
        'llm',
        llm_name=llm_name,
        llm_endpoint=os.environ['LLM_API_ENDPOINT'],
        llm_api_key=os.environ['LLM_API_KEY'],
        llm_model_id=model_id,
        create_item=False,
        distinct=True,
    )
    task: str = None
    context = None
    if df_prompt_template:
        task = df_prompt_template[data['Relation']]
    try:
        if relation == 'AwardWonBy'.lower() and (
            (
                (data['Relation'] in relation_filter)
                or (relation_filter is None or relation_filter.__len__() == 0)
            )
        ):
            log.write(f'Relation: {data["Relation"]}\n')
            log.write(
                f'Subject: {data["SubjectEntityID"]} {data["SubjectEntity"]}'
                + '\n'
            )
            llm: LLM_Store = Store(
                'llm',
                llm_name=llm_name,
                llm_endpoint=os.environ['LLM_API_ENDPOINT'],
                llm_api_key=os.environ['LLM_API_KEY'],
                llm_model_id=model_id,
                create_item=False,
                distinct=True,
                disambiguation_method='baseline',
                model_args={'max_new_tokens': 2048},
            )
            if use_context:
                context = get_context(
                    subject_item=subject_item,
                    predicate_item=predicate_item,
                    plugins=[
                        'ner-extract',
                    ],
                    filter_wiki_urls=True,
                    concat_results=True,
                )

            default_output = (
                '\n\nThe output should be only a '
                'list containing the answers, such as ["answer_1", '
                '"answer_2", ..., "answer_n"]. Do not provide '
                'any further explanation and avoid false answers. '
                'Return an empty list, such as [], if no information '
                'is available.'
            )

            predicate_item = wd.P(1346, 'winner')

            if not context or len(context) == 0:
                if task:
                    prompt_template = DEFAULT_PROMPT_TEMPLATE.copy()

                    prompt_template['user'] = (
                        ('\n\nTASK:\n"{task}"' + f'\n\n {default_output}')
                        .replace('{task}', task)
                        .replace('{subject_entity}', '{subject}')
                        .replace('{mask_token}', '_')
                    )

                    llm.prompt_template = prompt_template
                    log.write(f'Prompt template: {prompt_template}\n')
                    log.flush()

                return llm.filter(
                    subject=subject_item,
                    property=predicate_item,
                    distinct=True,
                )

            winners = iter([])

            a_context = context.split('; ')
            log.write(f'Returned names: {a_context}')

            for letter in range(ord('A'), ord('Z') + 1):
                try:
                    names = ', '.join(
                        [
                            item.strip()
                            for item in a_context
                            if item.startswith(chr(letter))
                        ]
                    )

                    if not names or len(names) == 0:
                        continue

                    llm.context = f'The winners of the {data["SubjectEntity"]} award are among the following people (consider only full names): {names}'
                    log.write(f'Context: {llm.context}\n')
                    if task:
                        prompt_template = DEFAULT_ENFORCED_CONTEXT.copy()
                        context_template = '\n\nCONTEXT:\n{context}'

                        prompt_template['user'] = (
                            (
                                '\n\nTASK:\n"{task}"'
                                + context_template
                                + f'\n\n {default_output}'
                            )
                            .replace('{task}', task)
                            .replace('{subject_entity}', '{subject}')
                            .replace('{mask_token}', '_')
                        )

                        llm.prompt_template = prompt_template
                        log.write(f'Prompt template: {prompt_template}\n')

                    log.write('\n')
                    log.flush()
                    winners = chain(
                        winners,
                        llm.filter(
                            subject=subject_item,
                            property=predicate_item,
                            distinct=True,
                        ),
                    )

                except Exception as e:
                    print(f'Error in {letter}: {e}')
            return winners

        elif relation == 'CompanyTradesAtStockExchange'.lower() and (
            (
                (data['Relation'] in relation_filter)
                or (relation_filter is None or relation_filter.__len__() == 0)
            )
        ):
            log.write(f'Relation: {data["Relation"]}\n')
            log.write(
                f'Subject: {data["SubjectEntityID"]} {data["SubjectEntity"]}\n'
            )
            predicate_item = wd.P(414, 'stock exchange')

            if use_context:
                context = get_context(
                    subject_item=subject_item,
                    predicate_item=predicate_item,
                    extra_urls=[
                        f'https://finance.yahoo.com/lookup/all?s={data["SubjectEntity"].replace(" ", "%20")}'
                    ],
                    plugins=[
                        'company-exchange',
                    ],
                    extra_language_tags=[
                        "it",
                        "fr",
                        "ru",
                        "th",
                        "nl",
                        "fa",
                        "he",
                        "pt",
                        "ja",
                        "zh",
                    ],
                )
                if context:
                    llm.context = context
                    log.write(f'Context: {context}\n')
                else:
                    context = f'The company {data["SubjectEntity"]} does not trade shares on any exchange.'
                    llm.context = context
            llm.disambiguation_method = 'baseline'
            if task:
                prompt_template = (
                    DEFAULT_ENFORCED_CONTEXT.copy()
                    if context
                    else DEFAULT_PROMPT_TEMPLATE.copy()
                )
                prompt_template['user'] = (
                    prompt_template['user']
                    .replace('{task}', task)
                    .replace('{subject_entity}', '{subject}')
                    .replace('{mask_token}', '_')
                )
                llm.prompt_template = prompt_template
                log.write(f'Prompt template: {prompt_template}\n')

            log.flush()

            return llm.filter(
                subject=subject_item,
                property=predicate_item,
                limit=limit,
                distinct=True,
            )

        elif relation == 'CountryLandBordersCountry'.lower() and (
            (
                (data['Relation'] in relation_filter)
                or (relation_filter is None or relation_filter.__len__() == 0)
            )
        ):
            log.write(f'Relation: {data["Relation"]}\n')
            log.write(
                f'Subject: {data["SubjectEntityID"]} {data["SubjectEntity"]}'
                + '\n'
            )
            instance_of = wd.P(31, 'is a')
            predicate_item = wd.P(47, 'shares border with')

            pat = FilterPattern(
                subject_item,
                predicate_item,
                [instance_of(wd.Q(6256, 'country'))],
            )

            country_borders = countries_border.get(
                wd.get_entity_label(subject_item).lower()
            )

            context = None
            if use_context:
                if (
                    wd.get_entity_label(subject_item).lower()
                    in countries_border
                ):
                    country_borders = countries_border.get(
                        wd.get_entity_label(subject_item).lower()
                    )
                    if len(country_borders) > 0:
                        context = (
                            f'The countries that share a land border with {wd.get_entity_label(subject_item)} are: '
                            + ', '.join(country_borders)
                        )
                    else:
                        context = f'No country shares a land border with {wd.get_entity_label(subject_item)}.'

            if context:
                llm.context = context
                log.write(f'Context: {context}\n')

            if task:
                prompt_template = (
                    DEFAULT_ENFORCED_CONTEXT.copy()
                    if context
                    else DEFAULT_PROMPT_TEMPLATE.copy()
                )
                prompt_template['user'] = (
                    prompt_template['user']
                    .replace('{task}', task)
                    .replace('{subject_entity}', '{subject}')
                    .replace('{mask_token}', '_')
                )
                llm.prompt_template = prompt_template
                log.write(f'Prompt template: {prompt_template}\n')
            llm.disambiguation_method = 'baseline'
            log.flush()
            return llm.filter(pattern=pat, distinct=True)

        elif relation == 'PersonHasCityOfDeath'.lower() and (
            (
                (data['Relation'] in relation_filter)
                or (relation_filter is None or relation_filter.__len__() == 0)
            )
        ):
            log.write(f'Relation: {data["Relation"]}\n')
            log.write(
                f'Subject: {data["SubjectEntityID"]} {data["SubjectEntity"]}'
                + '\n'
            )
            predicate_item = wd.P(20, 'death place')
            limit = 1
            instance_of = wd.P(31, 'is a')

            pat = FilterPattern(
                subject_item,
                predicate_item,
                [instance_of(wd.Q(515, 'city'))],
            )

            if use_context:
                context = get_context(
                    subject_item=subject_item,
                    predicate_item=predicate_item,
                    plugins=[
                        'wikipedia-place-of-death',
                        'wikitree-place-of-death',
                    ],
                    extra_language_tags=[
                        'pl',
                        'ru',
                        'de',
                        'es',
                        'uk',
                        'nl',
                        'az',
                        'arz',
                    ],
                )
                if context:
                    llm.context = context
                    log.write(f'Context: {context}\n')
                else:
                    context = f'Person {data["SubjectEntity"]} does not have a city of death.'
                    llm.context = context

            if task:
                prompt_template = (
                    DEFAULT_ENFORCED_CONTEXT.copy()
                    if context
                    else DEFAULT_PROMPT_TEMPLATE.copy()
                )
                prompt_template['user'] = (
                    prompt_template['user']
                    .replace('{task}', task)
                    .replace('{subject_entity}', '{subject}')
                    .replace('{mask_token}', '_')
                )
                llm.prompt_template = prompt_template
                log.write(f'Prompt template: {prompt_template}\n')
            llm.disambiguation_method = 'baseline'
            log.flush()
            return llm.filter(pattern=pat, limit=limit, distinct=True)

        elif relation == 'SeriesHasNumberOfEpisodes'.lower() and (
            (
                (data['Relation'] in relation_filter)
                or (relation_filter is None or relation_filter.__len__() == 0)
            )
        ):
            log.write(f'Relation: {data["Relation"]}\n')
            log.write(
                f'Subject: {data["SubjectEntityID"]} {data["SubjectEntity"]}\n'
            )
            llm.disambiguate = False
            limit = 1

            predicate_item = wd.P(1113, 'number of episodes')

            if use_context:
                context = get_context(
                    subject_item=subject_item,
                    predicate_item=predicate_item,
                    extra_urls=[
                        f'https://www.google.com/search?q={data["SubjectEntity"].replace(" ", "+")}+number+of+episodes'
                    ],
                    extra_language_tags=['ro', 'zh', 'fr'],
                    plugins=[
                        'wikipedia-episodes',
                        'imdb-episodes',
                        'google-episodes',
                    ],
                )

            if context:
                llm.enforce_context = True
                llm.context = context
                log.write(f'Context: {context}\n')

            if task:
                prompt_template = (
                    DEFAULT_ENFORCED_CONTEXT.copy()
                    if context
                    else DEFAULT_PROMPT_TEMPLATE.copy()
                )
                prompt_template['user'] = (
                    DEFAULT_ENFORCED_CONTEXT['user']
                    .replace('{task}', task)
                    .replace('{subject_entity}', '{subject}')
                    .replace('{mask_token}', '_')
                )
                llm.prompt_template = prompt_template
                log.write(f'Prompt template: {prompt_template}\n')

            log.flush()
            return llm.filter(
                subject=subject_item,
                property=predicate_item,
                limit=limit,
                distinct=True,
            )

        return None
    except Exception as e:
        logger.error(
            f'Error while filtering {subject_item} '
            f'and {predicate_item} {e.__str__()}'
        )
        raise e


def from_stmt_to_list(llm_result, lmkbc_challenge_year=2024):
    response_item = []
    response_label = []
    for stmt in llm_result:
        label = (
            stmt.snak.value.value
            if isinstance(stmt.snak.value, String)
            else wd.get_entity_label(stmt.snak.value)
        )
        if not label:
            continue
        value = stmt.snak.value.value.replace(
            'http://www.wikidata.org/entity/', ''
        )
        logging.info(f'Response from llm {value}')
        response_item.append(value)
        response_label.append(label)

    if len(response_item) == 0:
        response_item = [''] if lmkbc_challenge_year == 2023 else []
    if len(response_label) == 0:
        response_label = [''] if lmkbc_challenge_year == 2023 else []
    return response_item, response_label


def run(args):
    llm_name = args.llm
    model_id = args.model_id

    output = (
        args.output
        if args.output
        else (f'{pathlib.Path(__file__).resolve().parent}/predictions.jsonl')
    )

    use_context = args.no_context

    pt = None
    if args.prompt_template:
        try:
            logger.info('Reading prompt template file')
            pt = (
                pd.read_csv(args.prompt_template)
                .set_index('Relation')
                .to_dict()['PromptTemplate']
            )
        except Exception as e:
            raise e

    try:
        logger.info('Reading input file')
        json_object = read_json_l_file(args.input)
    except Exception as e:
        raise e

    for entry in json_object:
        try:
            llm_result = filter(
                llm_name=llm_name,
                model_id=model_id,
                json_data=entry,
                df_prompt_template=pt,
                relation_filter=args.filter_relation,
                use_context=use_context,
            )
            if not llm_result:
                continue

            entities, labels = from_stmt_to_list(llm_result)

            result = {
                'SubjectEntityID': entry['SubjectEntityID'],
                'SubjectEntity': entry['SubjectEntity'],
                'Relation': entry['Relation'],
                'ObjectEntities': labels,
                'ObjectEntitiesID': (
                    labels
                    if (
                        entry['Relation'].lower()
                        == 'seriesHasNumberOfEpisodes'.lower()
                        or entry['Relation'].lower()
                        == 'PersonHasNumberOfChildren'.lower()
                    )
                    else entities
                ),
            }

            print(result)
            log.write(f'Result: {result}\n')
            log.write('\n\n')
            log.flush()

            logger.info(f'Saving the results to \'{output}\'...')
            with open(output, 'a+') as f:
                f.write(json.dumps(result) + '\n')
        except Exception as e:
            raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument(
        '-l',
        '--llm',
        type=str,
        default='bam',
        help='LLM plugin ID (default: bam)',
    )
    parser.add_argument(
        '-m',
        '--model_id',
        type=str,
        default='meta-llama/llama-3-8b-instruct',
        help='Model ID (default: meta-llama/llama-3-8b-instruct',
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help='Input test file (required)',
    )
    parser.add_argument(
        '-fr',
        '--filter_relation',
        action='append',
        help='Relations to filter execution',
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output file',
    )
    parser.add_argument(
        '-lo',
        '--log_output',
        type=str,
        help='Log Output file',
    )
    parser.add_argument(
        '-pt',
        '--prompt_template',
        type=str,
        help='CSV file path containing the prompt templates',
    )
    parser.add_argument(
        '-nc',
        '--no_context',
        action='store_false',
        help='Disable search for context',
    )

    args = parser.parse_args()
    begin = time.time()
    print('Starting evaluation:')
    log_file = (
        args.log_output
        if args.log_output
        else (f'{pathlib.Path(__file__).resolve().parent}/log.txt')
    )
    with open(log_file, 'a+', buffering=1) as log:
        run(args)
    end = time.time()
    excution_time = end - begin
    print(f"Execution time: {excution_time:.4f} seconds")
