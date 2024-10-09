# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import ClassVar, Final, List, Tuple

import nest_asyncio
from kif_lib import Entity, Item, KIF_Object, Property
from kif_lib.typing import Any, AsyncIterator, Callable, Iterator, Optional

from ..constants import Label

LOG = logging.getLogger(__name__)

nest_asyncio.apply()


class Disambiguator:

    #: The name of the disambiguation plugin.
    disambiguator_name: ClassVar[str]

    registry: Final[dict[str, type['Disambiguator']]] = {}

    @classmethod
    def _register(
        cls,
        disambiguator: type['Disambiguator'],
        disambiguator_name: str,
    ):
        disambiguator.disambiguator_name = disambiguator_name
        cls.registry[disambiguator.disambiguator_name] = disambiguator

    @classmethod
    def __init_subclass__(cls, disambiguator_name: str):
        Disambiguator._register(cls, disambiguator_name)

    def __new__(
        cls,
        disambiguator_name: str,
        textual_context: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ):
        KIF_Object._check_arg(
            disambiguator_name,
            disambiguator_name in cls.registry,
            f'no such disambiguator plugin "{disambiguator_name}"',
            Disambiguator,
            'disambiguator_name',
            1,
            ValueError,
        )
        return super(Disambiguator, cls).__new__(
            cls.registry[disambiguator_name]
        )  # pyright: ignore

    def __init__(self, textual_context: Optional[str] = None, *args, **kwargs):
        pass

    def adisambiguate_item(
        self,
        labels: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[Entity]]]:
        return self._adisambiguate(
            labels, self._disambiguate_item, *args, **kwargs
        )

    def adisambiguate_property(
        self,
        labels: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[Entity]]]:
        """
        Asynchronously disambiguate a list of labels and return Wikidata
          properties.

        Args:
            labels (List[str]): A list of labels to disambiguate
            **kwargs: Arbitrary additional keyword arguments.

        Returns:
            AsyncIterator[Tuple[str, Optional[Entity]]]: A tuple containing
              the processed label and its corresponding Wikidata property.
        """
        return self._adisambiguate(
            labels, self._disambiguate_property, *args, **kwargs
        )

    def disambiguate_item(
        self,
        labels: List[str],
        context: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[Tuple[Label, Optional[Item]]]:
        """
        Disambiguate a list of labels and return Wikidata items.

        Args:
            labels (List[str]): A list of labels to disambiguate
            **kwargs: Arbitrary additional keyword arguments.

        Returns:
            Iterator[Tuple[str, Optional[Item]]]: A tuple containing
              the processed label and its corresponding Wikidata item.
        """

        async def run_disambiguation():
            results = []
            async for result in self._adisambiguate(
                labels, self._disambiguate_item, *args, **kwargs
            ):
                results.append(result)
            return results

        return asyncio.run(run_disambiguation())

    def disambiguate_property(
        self,
        labels: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[Tuple[Label, Optional[Property]]]:
        async def run_disambiguation():
            results = []
            async for result in self._adisambiguate(
                labels, self._disambiguate_property, *args, **kwargs
            ):
                results.append(result)
            return results

        return asyncio.run(run_disambiguation())

    async def _disambiguate_item(
        self,
        label: str,
        limit: int,
    ) -> Iterator[Tuple[Label, Optional[Item]]]:
        return iter(())

    async def _disambiguate_property(
        self,
        label: str,
        limit: int,
    ) -> Iterator[Tuple[Label, Optional[Property]]]:
        return iter(())

    async def _adisambiguate(
        self,
        labels: List[str],
        disambiguation_fn: Callable[
            [str, Any], AsyncIterator[Optional[Entity]]
        ],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[Entity]]]:
        """
        Asynchronously disambiguates the list of labels by retrieving
          their Wikidata Entity IDs.

        Args:
            labels (List[str]): A list of labels to disambiguate
            disambiguation_fn (Callable): An asynchronous function that yields
              Wikidata entities for a single label.

        Yields:
             AsyncIterator[Tuple[str, Optional[Entity]]]: A tuple containing
              the processed label and its corresponding Wikidata entity.
        """

        labels = [label for label in labels if label and label.strip()]
        semaphore = asyncio.Semaphore(10)

        async with semaphore:
            disambiguation_tasks = [
                asyncio.create_task(disambiguation_fn(label, *args, **kwargs))
                for label in labels
            ]

            try:
                for task in asyncio.as_completed(disambiguation_tasks):
                    label, entity = None, None
                    label, entity = await task
                    yield label, entity
            except Exception as e:
                LOG.error(f'Error processing task for label `{label}`: {e}')
                raise e
