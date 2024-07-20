# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional

from kif_lib.typing import override
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..llm.constants import LLM_Models
from .abc import LLM

LOG = logging.getLogger(__name__)


class HF_LLM(
    LLM,
    llm_name=LLM_Models.HUGGING_FACE_HUB.value,
    llm_description='Hugging Face LLM Models',
):

    _few_shot = Optional[int]
    _max_new_tokens = Optional[int]

    def __init__(
        self,
        llm_name: str,
        endpoint: str = '',
        api_key: str = '',
        model_id: str = 'facebook/opt-1.3b',
        **kwargs: Any
    ):
        assert llm_name == self.llm_name
        super().__init__(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side='left',
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id, device_map='cpu'
        )

        self.pipe = pipeline(
            task='text-generation',
            model=self.llm,
            tokenizer=self.tokenizer,
        )

        self._max_new_tokens = kwargs.get('max_new_tokens', 20000)

    @override
    def _execute_prompt(self, prompt: str) -> str:
        try:
            result = self.pipe(
                prompt,
                max_new_tokens=self._max_new_tokens,
            )

            return result[0]['generated_text']
        except Exception as e:
            LOG.error(e.__str__())
            raise e
