# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import nest_asyncio

from typing import (
    ClassVar,
    Final,
    List,
    Tuple,
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
)

from kif_lib import Entity, Item, KIF_Object, Property

from ..entity_sources import EntitySource


LOG = logging.getLogger(__name__)

nest_asyncio.apply()


class Disambiguator:
    """Disambiguator.

    Parameters:
       disambiguator_name: Disambiguator plugin to instantiate.
       source: Source of entities
    """

    #: The name of the disambiguation plugin.
    disambiguator_name: ClassVar[str]

    registry: Final[dict[str, type['Disambiguator']]] = {}

    __slots__ = '_source'

    _source: EntitySource

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

    def __new__(cls, disambiguator_name: str, *args: Any, **kwargs: Any):
        KIF_Object._check_arg(
            disambiguator_name,
            disambiguator_name in cls.registry,
            f'no such disambiguator plugin "{disambiguator_name}"',
            Disambiguator,
            'disambiguator_name',
            1,
            ValueError,
        )
        return super().__new__(
            cls.registry[disambiguator_name]
        )  # pyright: ignore

    def __init__(self, source: EntitySource, *args, **kwargs):
        self._source = source

    async def alabels_to_items(
        self,
        labels: List[str],
        limit: Optional[int] = 10,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[Item]]]:
        """
        Asynchronously disambiguates a list of labels and return items from
            the target store.

        Args:
            labels (List[str]): A list of labels to disambiguate
            **kwargs: Arbitrary additional keyword arguments.

        Returns:
            AsyncIterator[Tuple[str, Optional[Item]]]: A tuple containing
              the processed label and its corresponding item from the target
              store.
        """
        async for label, item in self.__adisambiguate(
            labels, limit, self._label_to_item, *args, **kwargs
        ):
            yield label, Item(iri=item)

    def labels_to_items(
        self,
        labels: List[str],
        limit: Optional[int] = 10,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, Optional[Item]]]:
        """
        Synchronously disambiguates a list of labels and return items
            from the target store.

        Args:
            labels (List[str]): A list of labels to disambiguate
            **kwargs: Arbitrary additional keyword arguments.

        Returns:
            Iterator[Tuple[str, Optional[Item]]]: A tuple containing
              the processed label and its corresponding item from
                the target store.
        """

        async def run_disambiguation():
            results = []
            async for label, item in self.__adisambiguate(
                labels, limit, self._label_to_item, *args, **kwargs
            ):
                result = label, Item(iri=item)
                results.append(result)
            return results

        return asyncio.run(run_disambiguation())

    async def _label_to_item(
        self, label, *args, **kwargs
    ) -> Tuple[str, Optional[Item]]:
        return (label, None)

    async def alabels_to_properties(
        self,
        labels: List[str],
        limit: Optional[int] = 10,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[Property]]]:
        async for label, property in self.__adisambiguate(
            labels, limit, self._label_to_property, *args, **kwargs
        ):
            result = label, Property(iri=property)
            yield result

    def labels_to_properties(
        self,
        labels: List[str],
        limit: Optional[int] = 10,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, Optional[Property]]]:
        """
        Synchronously disambiguates a list of labels and return properties
            from the target store.

        Args:
            labels (List[str]): A list of labels to disambiguate
            **kwargs: Arbitrary additional keyword arguments.

        Returns:
            Iterator[Tuple[str, Optional[Property]]]: A tuple containing
              the processed label and its corresponding property from
              the target store.
        """

        async def run_disambiguation():
            results = []
            async for label, property in self.__adisambiguate(
                labels, limit, self._label_to_property, *args, **kwargs
            ):
                result = label, Property(iri=property)
                results.append(result)
            return results

        return asyncio.run(run_disambiguation())

    async def _label_to_property(
        self, label, *args, **kwargs
    ) -> Tuple[str, Optional[Property]]:
        return (label, None)

    async def __adisambiguate(
        self,
        labels: List[str],
        limit: int,
        disambiguation_fn: Callable[
            [str, Any], AsyncIterator[Optional[Entity]]
        ],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[Entity]]]:
        """
        Asynchronously disambiguates the list of labels by retrieving
          their Entity IDs from the target store.

        Args:
            labels (List[str]): A list of labels to disambiguate
            disambiguation_fn (Callable): An asynchronous function that yields
                entities from the target store for a single label.

        Yields:
             AsyncIterator[Tuple[str, Optional[Entity]]]: A tuple containing
              the processed label and its corresponding entity from the target
              store.
        """

        labels = [label for label in labels if label and label.strip()]
        semaphore = asyncio.Semaphore(10)

        async with semaphore:
            disambiguation_tasks = [
                asyncio.create_task(
                    disambiguation_fn(label, limit, *args, **kwargs)
                )
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
