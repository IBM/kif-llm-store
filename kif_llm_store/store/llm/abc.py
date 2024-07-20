# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union

from kif_lib.error import Error as KIF_Error
from kif_lib.error import MustBeImplementedInSubclass, ShouldNotGetHere
from kif_lib.model import KIF_Object
from kif_lib.typing import Any, Final, Optional
from typing_extensions import TypeAlias

from ..llm.constants import ChatRole

LOG = logging.getLogger(__name__)


CONVERSATION_ID: TypeAlias = str
LLM_RESPONSE: TypeAlias = str

CHAT_CONTENT: TypeAlias = str


class LLM:
    """LLM factory.

    Parameters:
       llm_name: Name of the llm plugin to instantiate.
       args: Arguments to llm plugin.
       kwargs: Keyword arguments to llm plugin.
    """

    #: The global plugin registry.
    registry: Final[dict[str, type['LLM']]] = dict()

    #: The name of this llm plugin.
    llm_name: str

    #: The description of this llm plugin.
    llm_description: str

    #: The llm model identifier.
    model_id: str

    endpoint: str
    api_key: str

    def execute_prompt(
        self,
        prompt: Union[str, dict[ChatRole, CHAT_CONTENT]],
    ) -> tuple[LLM_RESPONSE, CONVERSATION_ID]:
        assert prompt is not None

        LOG.info('\n\tExecuting the following prompt:\n%s\n\n', prompt)
        response = self._execute_prompt(prompt)
        LOG.info('\n\tPrompt execution response:\n%s\n\n', response[0])
        return response

    def _execute_prompt(
        self,
        prompt: Union[str, dict[ChatRole, CHAT_CONTENT]],
    ) -> tuple[LLM_RESPONSE, CONVERSATION_ID]:
        assert prompt is not None
        raise self._must_be_implemented_in_subclass()

    @classmethod
    def _register(cls, llm: type['LLM'], llm_name: str, llm_description: str):
        llm.llm_name = llm_name
        llm.llm_description = llm_description
        cls.registry[llm.llm_name] = llm

    @classmethod
    def __init_subclass__(cls, llm_name: str, llm_description: str):
        LLM._register(cls, llm_name, llm_description)

    def __new__(cls, llm_name: str, *args: Any, **kwargs: Any):
        KIF_Object._check_arg(
            llm_name,
            llm_name in cls.registry,
            f"no such llm plugin '{llm_name}'",
            LLM,
            'llm_name',
            1,
            ValueError,
        )
        return super(LLM, cls).__new__(cls.registry[llm_name])

    class Error(KIF_Error):
        """Base class for llm errors."""

        pass

    @classmethod
    def _error(cls, details: str) -> Error:
        """Makes a llm error.

        Parameters:
           details: Details.

        Returns:
           LLM error.
        """
        return cls.Error(details)

    @classmethod
    def _must_be_implemented_in_subclass(
        cls, details: Optional[str] = None
    ) -> MustBeImplementedInSubclass:
        """Makes a "must be implemented in subclass" error.

        Parameters:
           details: Details.

        Returns:
           :class:`MustBeImplementedInSubclass` error.
        """
        return KIF_Object._must_be_implemented_in_subclass(details)

    @classmethod
    def _should_not_get_here(
        cls, details: Optional[str] = None
    ) -> ShouldNotGetHere:
        """Makes a "should not get here" error.

        Parameters:
           details: Details.

        Returns:
           :class:`ShouldNotGetHere` error.
        """
        return KIF_Object._should_not_get_here(details)

    def __init__(self, **kwargs):
        pass
