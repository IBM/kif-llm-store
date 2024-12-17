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

LOG = logging.getLogger(__name__)

nest_asyncio.apply()


class EntitySource:
    """Abstract base class for sources."""

    #: The source plugin registry.
    registry: Final[dict[str, type['EntitySource']]] = {}

    #: The name of this source plugin.
    source_name: ClassVar[str]

    @classmethod
    def _register(
        cls,
        source: type['EntitySource'],
        source_name: str,
    ) -> None:
        source.source_name = source_name
        cls.registry[source.source_name] = source

    @classmethod
    def __init_subclass__(
        cls,
        source_name: str,
    ) -> None:
        EntitySource._register(cls, source_name)

    def __new__(cls, source_name: str, *args: Any, **kwargs: Any):
        KIF_Object._check_arg(
            source_name,
            source_name in cls.registry,
            f"no such source plugin '{source_name}'",
            EntitySource,
            'source_name',
            1,
            ValueError,
        )
        return super().__new__(cls.registry[source_name])  # pyright: ignore

    __slots__ = ('_timeout',)

    def __init__(
        self, *args: Any, timeout: Optional[int] = None, **kwargs: Any
    ) -> None:
        """
        Initializes :class:`Source`.

        Parameters:
           source_name: Name of the source plugin to instantiate.
           args: Arguments to source plugin.
           timeout: Timeout of responses (in seconds).
           kwargs: Keyword arguments to source plugin.
        """

        self._init_timeout(timeout)

    def get_items_from_label(
        self, label: str, limit: Optional[int] = 10
    ) -> dict[str, Optional[Any]]:
        assert label
        assert limit > 0
        return self._get_items_from_label(label, limit)

    def _get_items_from_label(
        self, label: str, limit: Optional[int] = 10
    ) -> dict[str, Optional[Any]]:
        return {label: None}

    def get_properties_from_label(
        self, label: str, limit: Optional[int] = 10
    ) -> dict[str, Optional[Any]]:
        assert label
        assert limit > 0
        return self._get_properties_from_label(label, limit)

    def _get_properties_from_label(
        self, label: str, limit: Optional[int] = 10
    ) -> dict[str, Optional[Any]]:
        return {label: None}

    #: Timeout (in seconds).
    _timeout: Optional[float]

    def _init_timeout(self, timeout: Optional[float] = None) -> None:
        self.timeout = timeout  # type: ignore

    @property
    def timeout(self) -> Optional[float]:
        """The timeout of responses (in seconds)."""
        return self.get_timeout()

    @timeout.setter
    def timeout(self, timeout: Optional[float] = None) -> None:
        self.set_timeout(timeout)

    def get_timeout(self) -> Optional[float]:
        """Gets the timeout of responses (in seconds).


        Returns:
           Timeout or ``None``.
        """
        return self._timeout

    def set_timeout(self, timeout: Optional[float] = None) -> None:
        """Sets the timeout of responses (in seconds).

        Parameters:
           timeout: Timeout.
        """
        self._timeout = timeout


class Disambiguator:
    """Disambiguator.

    Parameters:
       disambiguator_name: Disambiguator plugin to instantiate.
       target_store: Source of entities
    """

    #: The name of the disambiguation plugin.
    disambiguator_name: ClassVar[str]

    registry: Final[dict[str, type['Disambiguator']]] = {}

    __slots__ = '_target_store'

    _target_store: EntitySource

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

    def __init__(self, target_store: EntitySource, *args, **kwargs):
        self._target_store = target_store

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
