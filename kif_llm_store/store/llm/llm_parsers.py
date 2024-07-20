# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0


class LLM_Parsers:

    @staticmethod
    def to_list(response: str) -> list[str]:
        import ast
        import re

        """Parse LLM response into a list of strings.

        :param response: The input string containing an array of strings.
        :return: A list of strings.
        """

        def flatten_list(nested_list: list):
            """Flatten a nested list into a single list."""
            flattened_list = []
            for item in nested_list:
                if isinstance(item, list):
                    flattened_list.extend(flatten_list(item))
                else:
                    flattened_list.append(item)
            return flattened_list

        if not response:
            return []

        try:
            result = ast.literal_eval(response)
            return flatten_list(result)
        except (ValueError, TypeError, SyntaxError):
            match = re.search(
                r'\[.*', response
            )  # Match arrays (closed or unclosed)
            if not match:
                return []
            try:
                partial_result = match.group()
                if not partial_result.endswith(']'):
                    # If the array is not properly closed,
                    # remove the last incomplete element
                    partial_result = (
                        re.sub(r',\s*"[^"]*$', '', partial_result) + ']'
                    )
                result = ast.literal_eval(partial_result)
                return flatten_list(result)
            except (ValueError, TypeError, SyntaxError):
                return []
