# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
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


class SemicolonSeparatedListOutputParser(CommaSeparatedListOutputParser):
    @override
    def parse(self, text: str) -> List[str]:
        '''Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        '''

        parts = text.split(";")

        # Remove punctuation from each slice
        cleaned_parts = [part.rstrip('?!.;:').strip() for part in parts]

        return cleaned_parts

    @override
    def get_format_instructions(self) -> str:
        '''Return the format instructions for the semicolon-separated list output.'''
        return (
            'Your response should be a list of semicolon separated values, '
            'eg: `foo; bar; baz` or `foo;bar;baz`'
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
            cleaned_part = part.rstrip('?!.;:').strip()
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
        '''Return the format instructions for the semicolon-separated list output.'''
        return (
            'Your response should be, if any, a list of numeric values, '
            'separated by semicolons. Ensure that decimal numbers use a dot '
            'instead of a comma, e.g.: `100.5; 200; 3.14159, 256875`.'
        )


__all__ = (
    'BaseOutputParser',
    'CommaSeparatedListOutputParser',
    'SemicolonSeparatedListOutputParser',
    'SemicolonSeparatedListOfNumbersOutputParser',
    'StrOutputParser',
    'SimpleJsonOutputParser',
    'MarkdownListOutputParser',
    'NumberedListOutputParser',
    'JsonOutputParser',
    'PydanticOutputParser',
)
