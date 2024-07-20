# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from kif_lib.typing import override
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .abc import LLM


class GPT_LLM(LLM, llm_name='gpt', llm_description='GPT from OpenAI'):
    """GPT LLM.

    Parameters:
        llm_name: LLM plugin to instantiate.
        endpoint: Endpoint to access llm models.
        api_key: API key to access the llm models
        model_id: The identifier of the model you want access from GPT
    """

    _client: OpenAI

    _best_of: Optional[int] = 1
    _echo: Optional[bool]
    _frequency_penalty: Optional[float]
    _logit_bias: Optional[Dict[int, int]]
    _logprobs: Optional[int]
    _max_tokens: Optional[int] = 16
    _n: Optional[int] = 1
    _presence_penalty: Optional[float] = 0.0
    _response_format: Optional[dict[str, str]]
    _seed: Optional[int]
    _stop: Optional[Union[str, List[str]]]
    _stream: Optional[bool]
    _suffix: Optional[str]
    _temperature: Optional[float]
    _top_p: Optional[float]
    _user: Optional[str]

    def __init__(
        self,
        llm_name: str,
        endpoint: str,
        api_key: str,
        model_id: str = 'gpt-3.5-turbo',
        **kwargs: Any
    ):
        assert llm_name == self.llm_name
        super().__init__(**kwargs)

        self.model_id = model_id

        self.endpoint = endpoint
        self.api_key = api_key
        self._client = OpenAI(api_key=api_key, base_url=endpoint)

        self._messages = kwargs.get('messages', [])
        self._frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        self._logit_bias = kwargs.get('logit_bias', None)
        self._logprobs = kwargs.get('logprobs', False)
        self._max_tokens = kwargs.get('max_tokens', None)
        self._n = kwargs.get('n', 1)
        self._presence_penalty = kwargs.get('presence_penalty', 0.0)
        self._response_format = kwargs.get('response_format', None)
        self._seed = kwargs.get('seed', None)
        self._stop = kwargs.get('stop', None)
        self._stream = kwargs.get('stream', False)
        self._temperature = kwargs.get('temperature', 1.0)
        self._top_logprobs = kwargs.get('top_logprobs', None)
        self._top_p = kwargs.get('top_p', 1.0)
        self._tools = kwargs.get('tools', None)
        self._tool_choice = kwargs.get('tool_choice', None)
        self._user = kwargs.get('user', None)

    @override
    def _execute_prompt(self, prompt: str) -> str:

        params = {
            'frequency_penalty': self._frequency_penalty,
            'logit_bias': self._logit_bias,
            'logprobs': self._logprobs,
            'max_tokens': self._max_tokens,
            'n': self._n,
            'presence_penalty': self._presence_penalty,
            'response_format': self._response_format,
            'seed': self._seed,
            'stop': self._stop,
            'stream': self._stream,
            'temperature': self._temperature,
            'top_logprobs': self._top_logprobs,
            'top_p': self._top_p,
            'tools': self._tools,
            'tool_choice': self._tool_choice,
            'user': self._user,
        }

        messages = [{'role': 'user', 'content': prompt}]

        response: ChatCompletion

        try:
            response = self._client.chat.completions.create(
                model=self.model_id, messages=messages, **params
            )
            return response.choices[0].message.content or ''
        except Exception as e:
            raise e
