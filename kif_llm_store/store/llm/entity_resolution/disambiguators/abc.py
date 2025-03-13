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
    TypeVar,
    Type,
)

from kif_lib import Entity, Item, KIF_Object

from ..entity_sources import EntitySource


LOG = logging.getLogger(__name__)

nest_asyncio.apply()

T = TypeVar("T", bound=Entity)


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

    def __init__(
        self,
        source: EntitySource,
        *args,
        **kwargs,
    ):
        self._source = source

    async def adisambiguate(
        self,
        labels: List[str],
        cls: Type[T] = Item,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[T]]]:
        """
        Asynchronously disambiguates a list of labels and return entities from
            the target store.

        Args:
            labels (List[str]): A list of labels to disambiguate
            **kwargs: Arbitrary additional keyword arguments.

        Returns:
            AsyncIterator[Tuple[str, Optional[Entity]]]: A tuple containing
              the processed label and its corresponding entity from the target
              store.
        """
        async for label, entity in self.__adisambiguate(
            labels, limit, self._disambiguate, *args, **kwargs
        ):
            yield label, cls(iri=entity)

    def disambiguate(
        self,
        labels: List[str],
        cls: Type[T] = Item,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, Optional[T]]]:
        """
        Synchronously disambiguates a list of labels and return entities
            from the target store.

        Args:
            labels (List[str]): A list of labels to disambiguate
            **kwargs: Arbitrary additional keyword arguments.

        Returns:
            Iterator[Tuple[str, Optional[Entity]]]: A tuple containing
              the processed label and its corresponding entity from
                the target store.
        """

        async def run_disambiguation():
            results = []
            async for label, entity in self.__adisambiguate(
                labels, limit, self._disambiguate, *args, **kwargs
            ):
                result = label, cls(iri=entity)
                results.append(result)
            return results

        return asyncio.run(run_disambiguation())

    async def _disambiguate(
        self, label, *args, **kwargs
    ) -> Tuple[str, Optional[T]]:
        return (label, None)

    async def __adisambiguate(
        self,
        labels: List[str],
        limit: int,
        disambiguation_fn: Callable[[str, Any], AsyncIterator[Optional[T]]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, Optional[T]]]:
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
