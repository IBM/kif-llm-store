# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union

from genai import Client, Credentials
from genai.exceptions import (
    ApiNetworkException,
    ApiResponseException,
    ValidationError,
)
from genai.schema import (
    AIMessage,
    DecodingMethod,
    HumanMessage,
    LengthPenalty,
    SystemMessage,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)
from kif_lib.typing import Any, Optional, override

from ..llm.constants import ChatRole, LLM_Models
from .abc import CHAT_CONTENT, CONVERSATION_ID, LLM, LLM_RESPONSE

LOG = logging.getLogger(__name__)


class BAM_LLM(
    LLM,
    llm_name=LLM_Models.BAM.value,
    llm_description='Big AI Model from IBM Research',
):
    """BAM LLM.

    Parameters:
        llm_name: LLM plugin to instantiate.
        endpoint: Endpoint to access llm models.
        api_key: API key to access the llm models
        model_id: The identifier of the model you want access from BAM
    """

    _credentials: Credentials
    _client: Client

    _model_id: str

    _decoding_method: Optional[DecodingMethod]
    _beam_width: Optional[int]
    _include_stop_sequence: Optional[bool] = None
    _length_penalty: Optional[LengthPenalty] = None
    _max_new_tokens: Optional[int]
    _min_new_tokens: Optional[int]
    _random_seed: Optional[int]
    _repetition_penalty: Optional[float]
    _return_options: Optional[TextGenerationReturnOptions] = None
    _stop_sequences: Optional[list[str]]
    _temperature: Optional[float]
    _time_limit: Optional[int]
    _top_k: Optional[int]
    _top_p: Optional[float]
    _truncate_input_tokens: Optional[int]
    _typical_p: Optional[float]

    def __init__(
        self,
        llm_name: str,
        api_key: str = '',
        endpoint: str = 'https://bam-api.res.ibm.com',
        model_id: str = 'meta-llama/llama-3-8b-instruct',
        **kwargs: dict[str, Any]
    ):
        assert llm_name == self.llm_name
        super().__init__(**kwargs)

        self.endpoint = endpoint
        self.api_key = api_key
        self._credentials = Credentials(api_key=api_key, api_endpoint=endpoint)
        self._client = Client(credentials=self._credentials)

        self._model_id = model_id

        self._temperature = kwargs.get('temperature', None)
        self._decoding_method = kwargs.get('decoding_method', 'greedy')
        self._max_new_tokens = kwargs.get('max_new_tokens', 100)
        self._min_new_tokens = kwargs.get('min_new_tokens', 1)
        self._return_options = kwargs.get(
            'return_options',
            TextGenerationReturnOptions(
                input_text=True,
                top_n_tokens=None,
                token_ranks=None,
                token_logprobs=None,
                generated_tokens=None,
                input_tokens=None,
            ),
        )
        self._random_seed = kwargs.get('random_seed', None)
        self._repetition_penalty = kwargs.get('repetition_penalty', None)
        self._stop_sequences = kwargs.get('stop_sequences', None)
        self._time_limit = kwargs.get('time_limit', None)
        self._top_k = kwargs.get('top_k', None)
        self._top_p = kwargs.get('top_p', None)
        self._truncate_input_tokens = kwargs.get('truncate_input_tokens', None)
        self._typical_p = kwargs.get('typical_p', None)
        self._beam_width = kwargs.get('beam_width', None)

    @override
    def _execute_prompt(
        self,
        prompt: Union[str, dict[ChatRole, CHAT_CONTENT]],
        conversation_id: Optional[str] = None,
    ) -> tuple[LLM_RESPONSE, CONVERSATION_ID]:
        assert prompt is not None
        parameters = TextGenerationParameters(
            min_new_tokens=self._min_new_tokens,
            random_seed=self._random_seed,
            repetition_penalty=self._repetition_penalty,
            stop_sequences=self._stop_sequences,
            time_limit=self._time_limit,
            top_k=self._top_k,
            top_p=self._top_p,
            truncate_input_tokens=self._truncate_input_tokens,
            typical_p=self._typical_p,
            beam_width=self._beam_width,
            max_new_tokens=self._max_new_tokens,
            decoding_method=self._decoding_method,
            return_options=self._return_options,
            temperature=self._temperature,
        )
        messages = []
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        else:

            content = prompt.get(ChatRole.SYSTEM.value)
            if content:
                messages.append(SystemMessage(content=content))

            content = prompt.get(ChatRole.USER.value)
            if content:
                messages.append(HumanMessage(content=content))

            content = prompt.get(ChatRole.ASSISTANT.value)
            if content:
                messages.append(AIMessage(content=content))

        try:
            response = self._client.text.chat.create(
                model_id=self._model_id,
                parameters=parameters,
                messages=messages,
                conversation_id=conversation_id,
            )

            return (
                response.results[0].generated_text,
                response.conversation_id,
            )

        except (
            ApiResponseException,
            ApiNetworkException,
            ValidationError,
        ) as e:
            LOG.error(e.__str__())
            raise e
