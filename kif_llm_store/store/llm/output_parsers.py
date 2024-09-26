# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import string
from typing import List, override
from langchain_core.output_parsers import (
    BaseOutputParser,
    CommaSeparatedListOutputParser,
    StrOutputParser,
    SimpleJsonOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)


class CommaSeparatedListOutputParserCleaned(CommaSeparatedListOutputParser):
    @override
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """

        parts = text.split(",")

        # Remove punctuation from each slice
        cleaned_parts = [
            part.translate(str.maketrans('', '', string.punctuation)).strip()
            for part in parts
        ]

        return cleaned_parts


__all__ = (
    'BaseOutputParser',
    'CommaSeparatedListOutputParser',
    'CommaSeparatedListOutputParserCleaned',
    'StrOutputParser',
    'SimpleJsonOutputParser',
    'MarkdownListOutputParser',
    'NumberedListOutputParser',
    'JsonOutputParser',
    'PydanticOutputParser',
)
