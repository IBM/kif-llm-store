# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import datetime

from decimal import Decimal
from kif_lib.typing import override
from typing import List
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


class SemicolonSeparatedListOutputParser(CommaSeparatedListOutputParser):

    @override
    def parse(self, text: str) -> List[str]:
        import string

        '''Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        '''

        parts = text.split(";")

        # Remove punctuation from each slice
        # cleaned_parts = [part.rstrip('?!.;:\"').strip() for part in parts]
        cleaned_parts = [
            part.translate(str.maketrans('', '', string.punctuation)).strip()
            for part in parts
        ]

        return cleaned_parts

    @override
    def get_format_instructions(self) -> str:
        '''Return the format instructions for the semicolon-separated list
        output.
        '''
        return (
            'Your response should be a sequence of semicolon separated noun '
            'phrases, eg: `foo; bar; baz`, or an empty string if there is no '
            'answer.'
        )


class SemicolonSeparatedListOfNumbersOutputParser(
    SemicolonSeparatedListOutputParser
):
    @override
    def parse(self, text: str) -> List[Decimal]:
        import re

        '''Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of numbers.
        '''

        parts = text.split(";")

        numbers = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            cleaned_part = part.rstrip('?!;:').strip()
            cleaned_part = re.sub(r'[^\d\.\-eE]+', '', cleaned_part).replace(
                ',', ''
            )
            try:
                number = int(cleaned_part)
            except ValueError:
                try:
                    number = Decimal(cleaned_part)
                except ValueError:
                    continue
            numbers.append(number)

        return numbers

    @override
    def get_format_instructions(self) -> str:
        '''Return the format instructions for the semicolon-separated list
        output.
        '''
        return (
            'Provide numeric values as complete numerals without '
            'abbreviations (e.g., 1 million as 1000000), separated '
            'by semicolons. Use periods for decimals and avoid thousands '
            'separators.'
        )


class SemicolonSeparatedListOfDateTimeOutputParser(
    SemicolonSeparatedListOutputParser
):
    @override
    def parse(self, text: str) -> List[datetime.datetime]:
        '''Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of date time.
        '''

        parts = text.split(";")

        date_times = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            # TODO: must be implemented
            date_times.append(part)

        return date_times

    @override
    def get_format_instructions(self) -> str:
        '''Return the format instructions for the semicolon-separated list
        output.
        '''
        # TODO: must be implemented
        return '...'


__all__ = (
    'BaseOutputParser',
    'CommaSeparatedListOutputParser',
    'SemicolonSeparatedListOutputParser',
    'SemicolonSeparatedListOfNumbersOutputParser',
    'SemicolonSeparatedListOfDateTimeOutputParser',
    'StrOutputParser',
    'SimpleJsonOutputParser',
    'MarkdownListOutputParser',
    'NumberedListOutputParser',
    'JsonOutputParser',
    'PydanticOutputParser',
)
