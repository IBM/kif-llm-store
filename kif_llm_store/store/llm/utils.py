# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union


def is_url(s: str) -> bool:
    from urllib.parse import urlparse

    try:
        result = urlparse(s)
        # Check if the scheme and netloc are present
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def similarity(
    sentence_from: str,
    sentences_to: Union[str, List[str]],
) -> map:
    import torch
    from sentence_transformers import SentenceTransformer, util

    sentences_to = (
        [sentences_to] if isinstance(sentences_to, str) else sentences_to
    )

    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentence_from_embedding = model.encode(
        sentence_from, convert_to_tensor=True
    )
    sentences_to_embeddings = model.encode(
        sentences_to, convert_to_tensor=True
    )

    assert isinstance(sentence_from_embedding, torch.Tensor)
    assert isinstance(sentences_to_embeddings, torch.Tensor)

    similarities = util.pytorch_cos_sim(
        sentence_from_embedding, sentences_to_embeddings
    )
    similarities = similarities.cpu().numpy().flatten()

    ranking = similarities.argsort()[::-1]

    return map(lambda i: (sentences_to[i], similarities[i]), iter(ranking))
