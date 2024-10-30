# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from kif_lib import Descriptor, Store
from kif_lib.model import Filter, Entity
from kif_lib.model.fingerprint import (
    AndFingerprint,
    CompoundFingerprint,
    FullFingerprint,
    OrFingerprint,
    SnakFingerprint,
    ValueFingerprint,
)
from kif_lib.typing import (
    Any,
    Iterable,
    override,
    Self,
    Optional,
)

from typing import Dict, List

from kif_llm_store.store.llm.constants import (
    ONE_VARIABLE_PROMPT_TASK,
    KIF_FilterTypes,
)

from ..llm.compiler import LLM_Compiler


class LogicalComponent:
    def __init__(self, components: List):
        self._components = components

    @property
    def components(self) -> List[Any]:
        return self._components


class OrComponent(LogicalComponent):
    def __init__(self, components: List):
        super().__init__(components)


class AndComponent(LogicalComponent):
    def __init__(self, components: List):
        super().__init__(components)


class Variable:
    _name: str
    _value: Optional[Any]

    def __init__(
        self,
        name: str,
        value: Optional[Any] = None,
        object_class: Any = None,
    ):
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        return self.get_name()

    def get_name(self) -> str:
        return self._name

    @property
    def value(self) -> str:
        return self.get_value()

    def get_value(self) -> str:
        return self._value

    @value.setter
    def value(self, v: Any):
        self.set_value(v)

    def set_value(self, v: Any):
        self._value = v


class LLM_FilterCompiler(LLM_Compiler):
    """LLM filter compiler."""

    __slots__ = (
        '_filter',
        '_instruction',
        '_query_template',
        '_task_sentence_template',
    )

    # The source filter.
    _filter: Filter

    _query_template: str

    _task_sentence_template: str

    _binds: Dict[str, Any]

    _filter_type: KIF_FilterTypes

    _has_where: bool

    _instruction: str

    def __init__(
        self, filter: Filter, flags: Optional[LLM_FilterCompiler.Flags] = None
    ) -> None:
        super().__init__(flags)
        self._filter = filter
        self._has_where = False
        self._binds = {}
        self._instruction = ONE_VARIABLE_PROMPT_TASK

    @property
    def filter(self) -> Filter:
        """The source filter."""
        return self.get_filter()

    def get_filter(self) -> Filter:
        """Gets the source filter.

        Returns:
           Filter.
        """
        return self._filter

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def filter_type(self) -> KIF_FilterTypes:
        """The filter type."""
        return self.get_filter_type()

    def get_filter_type(self) -> KIF_FilterTypes:
        """Gets the filter type.

        Returns:
           KIF_FilterTypes.
        """
        return self._filter_type

    @property
    def has_where(self) -> bool:
        return self._has_where

    @property
    def query_template(self):
        return self.get_query_template()

    def get_query_template(self) -> str:
        """Gets the query.

        Returns:
           Query.
        """
        return self._query_template

    @property
    def task_sentence_template(self):
        return self.get_task_sentence_template()

    def get_task_sentence_template(self) -> str:
        return self._task_sentence_template

    @property
    def binds(self):
        return self.get_binds()

    def get_binds(self) -> Dict[str, Any]:
        """Gets binds.

        Returns:
           Bind.
        """
        return self._binds

    @binds.setter
    def binds(self, v: Dict[str, Any]):
        self.set_binds(v)

    def set_binds(self, v: Dict[str, Any]):
        self._binds = v

    @override
    def compile(
        self,
        target_store: Store,
        task_prompt_template: Optional[str] = None,
    ) -> Self:
        filter = self._filter.normalize()
        self._check_filter_type(filter)
        self._push_filter(filter, target_store, task_prompt_template)
        return self

    def _check_filter_type(self, filter: Filter):
        assert not filter.is_empty()

        def check_filter_slice(
            filter_slices: Iterable,
            value_fingerprint_count=0,
            full_fingerprint_count=0,
            snak_fingerprint_count=0,
        ):
            for fs in filter_slices:
                if isinstance(fs, ValueFingerprint):
                    value_fingerprint_count += 1
                elif isinstance(fs, FullFingerprint):
                    full_fingerprint_count += 1
                elif isinstance(fs, SnakFingerprint):
                    snak_fingerprint_count += 1
                elif isinstance(fs, CompoundFingerprint):
                    for f in fs:
                        (
                            value_fingerprint_count,
                            full_fingerprint_count,
                            snak_fingerprint_count,
                        ) = check_filter_slice(
                            [f],
                            value_fingerprint_count,
                            full_fingerprint_count,
                            snak_fingerprint_count,
                        )

            return (
                value_fingerprint_count,
                full_fingerprint_count,
                snak_fingerprint_count,
            )

        s, p, v = filter.subject, filter.property, filter.value

        vfc, ffc, sfc = check_filter_slice([s, p, v])
        if vfc >= 2:
            self._filter_type = KIF_FilterTypes.ONE_VARIABLE
        else:
            self._filter_type = KIF_FilterTypes.GENERIC

    def _push_filter(
        self,
        filter: Filter,
        target_store: Store,
        task_prompt_template: Optional[str] = None,
    ) -> None:
        if filter.is_empty():
            return  # nothing to do

        ps = filter.subject
        pp = filter.property
        pv = filter.value
        where = []

        def compile(filter, var_count=0):
            where = ''
            if isinstance(filter, FullFingerprint):
                var_count += 1
                var = Variable(f'var{var_count}')
                return var, None, var_count
            if isinstance(filter, ValueFingerprint):
                return filter.value, None, var_count
            if isinstance(filter, SnakFingerprint):
                var_count += 1
                var = Variable(f'var{var_count}')
                property = filter[0][0]
                item = filter[0][1]
                property_label = target_store.get_descriptor(
                    entities=property, mask=Descriptor.AttributeMask.LABEL
                )
                property_label = next(property_label)[1].label.content
                item_label = target_store.get_descriptor(
                    entities=item, mask=Descriptor.AttributeMask.LABEL
                )
                item_label = next(item_label)[1].label.content
                where += f'{var.get_name()} {property_label} {item_label}'  # noqa E501
                return var, where, var_count
            if isinstance(filter, CompoundFingerprint):
                if isinstance(filter, OrFingerprint):
                    local_var_count = var_count
                    local_where = []
                    var = None
                    var_components = []
                    for exp in filter:
                        var, _where, local_var_count = compile(
                            target_store, exp, local_var_count
                        )
                        if _where:
                            local_var_count -= 1
                            local_where.append(_where)
                        else:
                            var_components.append(var)
                    if var_components:
                        var = OrComponent(var_components)
                    var_count = local_var_count + 1
                    return var, ' or '.join(local_where), var_count
                if isinstance(filter, AndFingerprint):
                    local_var_count = var_count
                    local_where = []
                    var = None
                    var_components = []
                    for exp in filter:
                        var, _where, local_var_count = compile(
                            exp, local_var_count
                        )
                        if _where:
                            local_var_count -= 1
                            local_where.append(_where)
                        else:
                            var_components.append(var)
                    if var_components:
                        var = AndComponent(var_components)
                    var_count = local_var_count + 1
                    return var, ' and '.join(local_where), var_count
            raise ValueError('Can\'t compile this filter')

        s, sw, svar = compile(ps)
        p, pw, pvar = compile(pp, svar)
        v, vw, _ = compile(pv, pvar)

        if sw:
            where.append(sw)
        if pw:
            where.append(pw)
        if vw:
            where.append(vw)

        def build_task_prompt_template(
            entity, query_template, replacement
        ) -> str:
            value_template = entity
            if isinstance(entity, Variable):
                value_template = entity.get_name()
                return query_template.replace(replacement, value_template)

            if isinstance(value_template, Entity):
                entity_label = target_store.get_descriptor(
                    entities=value_template,
                    mask=Descriptor.AttributeMask.LABEL,
                )
                value_template = next(entity_label)[1].label.content
            elif isinstance(value_template, OrComponent):
                labels = []
                for component in value_template.components:
                    entity_label = target_store.get_descriptor(
                        entities=component,
                        mask=Descriptor.AttributeMask.LABEL,
                    )
                    labels.append(next(entity_label)[1].label.content)
                value_template = ' or '.join(labels)
            elif isinstance(value_template, AndComponent):
                labels = []
                for component in value_template.components:
                    entity_label = target_store.get_descriptor(
                        entities=component,
                        mask=Descriptor.AttributeMask.LABEL,
                    )
                    labels.append(next(entity_label)[1].label.content)
                value_template = ' and '.join(labels)
            return query_template.replace(replacement, value_template)

        default_task_template = '{subject} {property} {value}'
        value_template = default_task_template
        query_template = self._instruction or ONE_VARIABLE_PROMPT_TASK
        if self._filter_type == KIF_FilterTypes.ONE_VARIABLE:
            self._instruction = ONE_VARIABLE_PROMPT_TASK
            value_template = build_task_prompt_template(
                s, default_task_template, '{subject}'
            )
            value_template = build_task_prompt_template(
                p, value_template, '{property}'
            )
            value_template = build_task_prompt_template(
                v, value_template, '{value}'
            )
            query_template += value_template
        self._task_sentence_template = value_template

        if task_prompt_template:
            if self._filter_type == KIF_FilterTypes.ONE_VARIABLE:
                value_template = build_task_prompt_template(
                    s, task_prompt_template, '{subject}'
                )
                value_template = build_task_prompt_template(
                    p, value_template, '{property}'
                )
                value_template = build_task_prompt_template(
                    v, value_template, '{value}'
                )
                query_template = value_template

        self.binds = {'subject': s, 'property': p, 'value': v}

        if where:
            self._has_where = True
            query_template += '\nwhere'
            for w in where:
                query_template += f'\n{w} and'
            query_template = query_template.rsplit(' and', 1)
            query_template = ''.join(query_template)
        self._query_template = query_template
