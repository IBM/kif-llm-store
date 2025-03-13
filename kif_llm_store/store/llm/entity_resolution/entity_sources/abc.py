# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging

from typing import (
    ClassVar,
    Final,
    Any,
    Optional,
)

from kif_lib import KIF_Object

LOG = logging.getLogger(__name__)


class EntitySource:
    """Abstract base class for sources."""

    #: The source plugin registry.
    registry: Final[dict[str, type['EntitySource']]] = {}

    #: The name of this source plugin.
    source_name: ClassVar[str]

    default_prefix_item_iri: str
    default_prefix_property_iri: str

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
        self,
        *args: Any,
        timeout: Optional[int] = None,
        **kwargs: Any,
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

    def lookup_item_search(
        self, label: str, limit: Optional[int] = 10
    ) -> dict[str, Optional[Any]]:
        assert label
        assert limit > 0
        return self._lookup_item_search(label, limit)

    def parse_entity(self, id) -> Optional[str]:
        return self._parse_entity(id)

    def _parse_entity(self, id: str) -> Optional[str]:
        return id

    def _lookup_item_search(
        self, label: str, limit: Optional[int] = 10
    ) -> dict[str, Optional[Any]]:
        return {label: None}

    def lookup_property_search(
        self, label: str, limit: Optional[int] = 10
    ) -> dict[str, Optional[Any]]:
        assert label
        assert limit > 0
        return self._lookup_property_search(label, limit)

    def _lookup_property_search(
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
