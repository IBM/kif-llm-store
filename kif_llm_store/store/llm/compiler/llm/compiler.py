# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import enum
import itertools

from kif_lib.typing import Final, Iterator
from kif_lib.compiler import Compiler
from typing_extensions import TypeVar

T = TypeVar('T')


class LLM_Compiler(Compiler):
    """Abstract base class for LLM compilers."""

    class Query:
        """Compiled LLM query."""

        def __init__(self, *variables, **kwargs) -> None:
            pass

    class Flags(enum.Flag):
        """Compilation flags."""

        #: Whether to generate debugging information.
        DEBUG = enum.auto()

        #: Whether to match only the best ranked statements.
        BEST_RANK = enum.auto()

        #: Whether to match value snaks.
        VALUE_SNAK = enum.auto()

        #: Whether to match some-value snaks.
        SOME_VALUE_SNAK = enum.auto()

        #: Whether to match no-value snaks.
        NO_VALUE_SNAK = enum.auto()

        #: Whether to push early filters.
        EARLY_FILTER = enum.auto()

        #: Whether to use Wikidata RDF extensions.
        WIKIDATA_EXTENSIONS = enum.auto()

        #: All flags.
        ALL = (
            DEBUG
            | BEST_RANK
            | VALUE_SNAK
            | SOME_VALUE_SNAK
            | NO_VALUE_SNAK
            | EARLY_FILTER
            | WIKIDATA_EXTENSIONS
        )

    #: The default flags.
    default_flags: Final[Flags] = Flags.ALL & ~(
        Flags.DEBUG | Flags.WIKIDATA_EXTENSIONS
    )

    DEBUG = Flags.DEBUG
    BEST_RANK = Flags.BEST_RANK
    VALUE_SNAK = Flags.VALUE_SNAK
    SOME_VALUE_SNAK = Flags.SOME_VALUE_SNAK
    NO_VALUE_SNAK = Flags.NO_VALUE_SNAK
    EARLY_FILTER = Flags.EARLY_FILTER
    WIKIDATA_EXTENSIONS = Flags.WIKIDATA_EXTENSIONS

    __slots__ = (
        '_q',
        '_flags',
    )

    #: The compiled query.
    _q: LLM_Compiler.Query

    #: The compilation flags.
    _flags: LLM_Compiler.Flags

    @abc.abstractmethod
    def __init__(self, flags: Flags | None = None) -> None:
        super().__init__()
        self._q = self.Query()
        if flags is None:
            self._flags = self.default_flags
        else:
            self._flags = self.Flags(flags)

    @property
    def flags(self) -> Flags:
        """The flags set in compiler."""
        return self.get_flags()

    @flags.setter
    def flags(self, flags: Flags) -> None:
        if flags != self._flags:
            self._flags = self.Flags(flags)

    def get_flags(self) -> Flags:
        """Gets the flags set in compiler.

        Returns:
           Compiler flags.
        """
        return self._flags

    def has_flags(self, flags: Flags) -> bool:
        """Tests whether `flags` are set in compiler.

        Parameters:
           flags: Compiler flags.

        Returns:
           ``True`` if successful; ``False`` otherwise.
        """
        return bool(self.flags & flags)

    def set_flags(self, flags: Flags) -> None:
        """Sets `flags` in compiler.

        Parameters:
           flags: Compiler flags.
        """
        self.flags |= flags

    def unset_flags(self, flags: Flags) -> None:
        """Unsets `flags` in compiler.

        Parameters:
           flags: Compiler flags.
        """
        self.flags &= ~flags

    @property
    def query_template(self) -> Query:
        """The compiled query."""
        return self.get_query_template()

    def get_query_template(self) -> Query:
        """Gets the compiled query.

        Returns:
           LLM query.
        """
        return self._q

    def _qvar(self, var: Query.TVariable) -> Query.Variable:
        return self._q.var(var)

    def _qvars(
        self, var: Query.TVariable, *vars_: Query.Variable
    ) -> Iterator[Query.Variable]:
        return map(self._qvar, itertools.chain((var,), vars_))

    def _fresh_qvar(self) -> Query.Variable:
        return self._q.fresh_var()

    def _fresh_qvars(self, n: int) -> Iterator[Query.Variable]:
        return self._q.fresh_vars(n)
