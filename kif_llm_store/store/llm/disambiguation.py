# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0


import logging
from concurrent.futures import as_completed, ThreadPoolExecutor

from kif_lib import FilterPattern, Item
from kif_lib.typing import Any, Callable, Optional, Union
from kif_lib.vocabulary import wd

from ..llm.abc import CHAT_CONTENT, LLM
from ..llm.constants import ChatRole, WIKIDATA_SEARCH_API_BASE_URL

# from sentence_transformers import SentenceTransformer, util


LOG = logging.getLogger(__name__)


def llm_based_disambiguation(
    labels: list[str],
    pattern: FilterPattern,
    llm_model: LLM,
    complete_template: Callable[
        [Any], Union[str, dict[ChatRole, CHAT_CONTENT]]
    ],
    create_wikidata_entity=False,
    prompt_template: Optional[str] = None,
    context: Optional[str] = None,
) -> dict[str, Union[str, Item]]:
    """
    :param items: List of items to disambiguate
    :create_wikidata_entity: Should create a wikidata entity if no
    match is found
    :context: Context to use in In-Context prompting
    :prompt_template: Prompt template to use in prompting
    :return: A dictionary containing the disambiguated items and
    their corresponding wikidata entities
    """

    response: dict[str, Union[str, Item]] = {}

    try:
        items_candidates = _get_candidates(labels)
    except Exception as exc:
        LOG.error('Exceptions generated while getting candidates:  %s', exc)

    response = _disambiguate_candidates(
        item_candidates=items_candidates,
        pattern=pattern,
        llm_model=llm_model,
        prompt_template=prompt_template,
        complete_template=complete_template,
        context=context,
        create_wikidata_entity=create_wikidata_entity,
    )
    return response


def baseline_disambiguation(
    labels: list[str],
    create_wikidata_entity=False,
) -> dict[str, Union[str, Item]]:
    response: dict[str, Union[str, Item]] = {}

    for label in labels:
        label = label.strip()

        response[label] = _disambiguation_baseline(
            label, create_wikidata_entity
        )

    return response


def _disambiguation_baseline(
    label, create_wikidata_entity=False
) -> Union[str, Item]:
    import requests

    """
    A simple disambiguation function that returns the Wikidata
    ID of an label.
    """
    label = str(label).strip()

    assert label is not None

    try:
        url = (
            f'{WIKIDATA_SEARCH_API_BASE_URL}'
            f'?action=wbsearchentities'
            f'&search={label}'
            f'&language=en'
            f'&format=json'
        )
        data = requests.get(url).json()
        # Return the first id (Could upgrade this in the future)
        QID = str(data['search'][0]['id'])

        if not QID and create_wikidata_entity:
            return _create_new_wikidata_item(label)

        return wd.Q(name=QID, label=data['search'][0]['label'])

    except Exception as e:
        logging.error(f"Error getting Wikidata ID for `{label}`: {e}")
        return label


def similarity_based_disambiguation(
    labels: list[str],
    pattern: FilterPattern,
    context: Optional[str] = None,
    create_wikidata_entity=False,
) -> dict[str, Union[str, Item]]:

    response: dict[str, Union[str, Item]] = {}
    try:
        item_candidates = _get_candidates(labels)
    except Exception as exc:
        LOG.error('Exceptions generated while getting candidates:  %s', exc)

    candidates_to_process = {}
    for item, candidates in item_candidates.items():
        if candidates:
            candidates_to_process[item] = candidates
        else:
            if create_wikidata_entity:
                response[item] = _create_new_wikidata_item(item)

    candidates_len = len(candidates_to_process)
    if candidates_len > 0:
        with ThreadPoolExecutor(max_workers=candidates_len) as executor:
            future_to_item = {
                executor.submit(
                    _disambiguate_item_similarity_based,
                    item,
                    pattern,
                    candidates,
                    context,
                ): candidates
                for item, candidates in candidates_to_process.items()
            }
            for future in as_completed(future_to_item):
                try:
                    label_qid = future.result()
                    assert label_qid is not None
                    item_label, QID = next(iter(label_qid.items()))
                except Exception as exc:
                    if create_wikidata_entity:
                        response[item_label] = _create_new_wikidata_item(
                            item_label
                        )
                    else:
                        QID = next(
                            iter(candidates_to_process[item_label].keys())
                        )
                        if QID:
                            if candidates_to_process[item_label][QID]['label']:
                                response[item_label] = wd.Q(
                                    name=QID,
                                    label=candidates_to_process[item_label][
                                        QID
                                    ]['label'],
                                )

                    LOG.error(
                        '%s generated an exception: {}'.format(exc), item_label
                    )

    return response


def _get_candidates(labels: list[str]) -> dict[str, Any]:
    assert labels is not None, "No items to disambiguate."
    assert labels.__len__() > 0, "Empty list of items to disambiguate."

    with ThreadPoolExecutor(max_workers=labels.__len__()) as executor:
        future_to_item = {
            executor.submit(_fetch_wikidata_entity, label): label
            for label in labels
        }
        item_candidates = {}
        for future in as_completed(future_to_item):
            label = future_to_item[future]
            if label is None:
                continue
            try:
                data = future.result()
                if data and 'search' in data and data['search'].__len__() > 0:
                    """
                    TODO: When only one candidate check if the label
                      matches with the search term. If yes, no need to
                      disambiguate and use the single candidate as the
                      answer. If not asks the LLM what it thinks about it.
                      It is to ensure we are not using the candidate as
                      the final answer just because there is no other
                      options.
                    """
                    # if data['search'].__len__() == 1:
                    #     response[item] = wd.Q(
                    #         name=data['search'][0]['id'], label=item
                    #     )
                    # else:
                    candidates = {
                        entity['id']: {
                            'label': entity['label'],
                            'description': entity['description'],
                        }
                        for entity in data['search']
                        if 'description' in entity
                    }
                    item_candidates[label] = candidates
                else:
                    # No candidates
                    item_candidates[label] = {}

                    # if label:
                    #     if create_wikidata_entity:
                    #         random_bytes = os.urandom(32)
                    #         hash_object = hashlib.sha256()
                    #         hash_object.update(random_bytes)
                    #         hash_digest = hash_object.hexdigest()

                    #         item_candidates[label] = {
                    #             f'Q_LLM_Store_{hash_digest}': {
                    #                 'label': label,
                    #             }
                    #         }
                    # wd.Q(name=QID, label=label)
            except Exception as exc:
                LOG.error('%s generated an exception: {}'.format(exc), label)

        return item_candidates


def _create_new_wikidata_item(label):
    import hashlib
    import os

    random_bytes = os.urandom(32)
    hash_object = hashlib.sha256()
    hash_object.update(random_bytes)
    hash_digest = hash_object.hexdigest()
    return wd.Q(
        name=f'Q_LLM_Store_{hash_digest}',
        label=label,
    )


def _disambiguate_candidates(
    item_candidates: dict[str, Any],
    llm_model: LLM,
    pattern: FilterPattern,
    create_wikidata_entity: bool,
    complete_template: Callable[
        [Any], Union[str, dict[ChatRole, CHAT_CONTENT]]
    ],
    prompt_template: Optional[Union[str, dict[ChatRole, CHAT_CONTENT]]] = None,
    context: Optional[str] = None,
) -> dict[str, Union[str, Item]]:
    import re

    response: dict[str, Union[str, Item]] = {}

    candidates_to_process = {}
    for item, candidates in item_candidates.items():
        if candidates:
            candidates_to_process[item] = candidates
        else:
            if create_wikidata_entity:
                response[item] = _create_new_wikidata_item(item)

    candidates_len = len(candidates_to_process)
    if candidates_len > 0:
        with ThreadPoolExecutor(max_workers=candidates_len) as executor:
            future_to_item = {
                executor.submit(
                    _disambiguate_item_llm_based,
                    item,
                    pattern,
                    llm_model,
                    candidates,
                    complete_template,
                    prompt_template,
                    context,
                ): candidates
                for item, candidates in candidates_to_process.items()
            }
            for future in as_completed(future_to_item):
                try:
                    label_qid = future.result()
                    assert label_qid is not None
                    item_label, QID = next(iter(label_qid.items()))

                    if not QID:
                        if create_wikidata_entity:
                            response[item_label] = _create_new_wikidata_item(
                                item_label
                            )
                        continue
                    match = re.search(r'Q\d+', QID)

                    if match:
                        QID = match.group()

                        if QID in candidates_to_process[item_label]:
                            response[item_label] = wd.Q(
                                name=QID,
                                label=candidates_to_process[item_label][QID][
                                    'label'
                                ],
                            )
                        else:
                            # Answered an item not listed in the
                            # candidates TODO: we should try again
                            # inside the same session
                            QID = next(
                                iter(candidates_to_process[item_label].keys())
                            )
                            if QID:
                                response[item_label] = wd.Q(
                                    name=QID,
                                    label=candidates_to_process[item_label][
                                        QID
                                    ]['label'],
                                )

                    else:
                        QID = next(
                            iter(candidates_to_process[item_label].keys())
                        )
                        if QID:
                            response[item_label] = wd.Q(
                                name=QID,
                                label=candidates_to_process[item_label][QID][
                                    'label'
                                ],
                            )
                except Exception as exc:
                    if create_wikidata_entity:
                        response[item_label] = _create_new_wikidata_item(
                            item_label
                        )
                    else:
                        QID = next(
                            iter(candidates_to_process[item_label].keys())
                        )
                        if QID:
                            if candidates_to_process[item_label][QID]['label']:
                                response[item_label] = wd.Q(
                                    name=QID,
                                    label=candidates_to_process[item_label][
                                        QID
                                    ]['label'],
                                )

                    LOG.error(
                        '%s generated an exception: {}'.format(exc), item_label
                    )

    return response


def keyword_based_disambiguation(
    labels: list[str],
    keywords: Optional[dict[str, list[str]]],
    create_item: bool = False,
) -> dict[str, Union[str, Item]]:
    import re

    import requests

    """select the entity based on keywords

    :param item:
    :param keywords:
    :return:
    """

    # TODO run in parallel
    response = {}
    for item in labels:
        try:
            url = (
                f'{WIKIDATA_SEARCH_API_BASE_URL}'
                '?action=wbsearchentities'
                f'&search={item}&language=en'
                '&format=json'
            )

            data = requests.get(url).json()

            found = False
            for entity in data['search']:
                if keywords and keywords.get(item):
                    for keyword in keywords[item]:
                        if re.search(keyword, entity['description']):
                            response[item] = entity['concepturi']
                            found = True
                            break
            if not found and not create_item:
                response[item] = entity['concepturi']

        except (KeyError, IndexError):
            response[item] = item
    return response


def _disambiguate_item_llm_based(
    item: str,
    pattern: FilterPattern,
    model: LLM,
    candidates: dict[str, dict[str, str]],
    complete_template: Callable[
        [Any], Union[str, dict[ChatRole, CHAT_CONTENT]]
    ],
    prompt_template: Optional[Union[str, dict[ChatRole, CHAT_CONTENT]]] = None,
    context: Optional[str] = None,
) -> dict[str, Union[str, None]]:
    assert len(candidates) > 0

    # TODO use prompt_template from parameters
    if not prompt_template:
        prompt_template = {
            ChatRole.SYSTEM: '',
            ChatRole.USER: '',
            ChatRole.ASSISTANT: '',
        }

        prompt_template[ChatRole.SYSTEM] += (
            'You are a helpful and honest assistant that '
            'resolves a TASK. Please, respond concisely, with '
            'no further explanation, and truthfully.'
        )

        prompt_template[ChatRole.USER] += '\n\nTASK:\n"{task_template}"'

        task_template = (
            'From the list of CANDIDATES below, use the labels and '
            'descriptions to select the QID that best '
            'replaces completes the relation "{subject} {predicate} {object}".'
        )

        if context:
            prompt_template[
                ChatRole.USER
            ] += ' Use the CONTEXT to support the answer.'

        options = 'CANDIDATES:\n'

        for candidate_QID, label_description in candidates.items():
            options += f'QID: {candidate_QID}:\n'
            options += f'label: {label_description["label"]}\n'
            options += f'description: {label_description["description"]}\n\n'

        prompt_template[ChatRole.USER] += f'\n\n{options}'

        if context:
            prompt_template[ChatRole.USER] += '\n\nCONTEXT:\n"{context}"'

        prompt_template[
            ChatRole.USER
        ] += '\n\nReturn only the QID such as "Q123456".\n'

        prompt = complete_template(
            prompt_template=prompt_template,
            pattern=pattern,
            context=context,
            task_template=task_template,
        )
        response, _ = model.execute_prompt(prompt)
        return {item: response}

    # TODO:
    return {item: None}


def _disambiguate_item_similarity_based(
    item: str,
    pattern: FilterPattern,
    candidates: dict[str, dict[str, str]],
    context: Optional[str] = None,
) -> dict[str, Union[str, None]]:
    assert len(candidates) > 0
    return {item: None}


def _fetch_wikidata_entity(item) -> Union[dict[str, Any], None]:
    import requests

    try:
        if item:
            url = (
                f'{WIKIDATA_SEARCH_API_BASE_URL}'
                '?action=wbsearchentities'
                f'&search={item}'
                '&language=en&format=json'
            )
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
        return None
    except requests.exceptions.RequestException as err:
        LOG.error(
            '%s(): reqquest error:\n%s',
            _fetch_wikidata_entity.__qualname__,
            err.strerror,
        )
        return None
