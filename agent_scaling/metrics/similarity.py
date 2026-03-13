from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

try:
    from bert_score import score as bert_score  # type: ignore

    _HAS_BERTSCORE = True
except Exception:
    bert_score = None
    _HAS_BERTSCORE = False

try:
    from litellm import embedding as litellm_embedding

    _HAS_LITELLM_EMBEDDING = True
except Exception:
    litellm_embedding = None
    _HAS_LITELLM_EMBEDDING = False


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bow_vector(text: str) -> Dict[str, float]:
    counts = Counter(tokenize(text))
    return {k: float(v) for k, v in counts.items()}


def _cosine_from_bow(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for k, v in vec_a.items():
        dot += v * vec_b.get(k, 0.0)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_from_dense(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must have the same dimension")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def cosine_similarity(text_a: str, text_b: str) -> float:
    return _cosine_from_bow(_bow_vector(text_a), _bow_vector(text_b))


def pairwise_cosine_similarities(texts: List[str]) -> List[float]:
    sims: List[float] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(cosine_similarity(texts[i], texts[j]))
    return sims


def bertscore_similarity(pairs: List[Tuple[str, str]]) -> List[float]:
    if not _HAS_BERTSCORE:
        raise RuntimeError("bert_score is not available")
    cands = [p[0] for p in pairs]
    refs = [p[1] for p in pairs]
    _, _, f1 = bert_score(cands, refs, lang="en", verbose=False)
    return [float(x) for x in f1]


def _obj_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Unsupported response payload type: {type(value)!r}")


def _extract_embedding_rows(response: Any) -> List[Dict[str, Any]]:
    payload = _obj_to_dict(response)
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Embedding response did not contain a `data` list")
    rows: List[Dict[str, Any]] = []
    for item in data:
        item_dict = _obj_to_dict(item)
        if "embedding" not in item_dict:
            raise RuntimeError("Embedding item missing `embedding` field")
        rows.append(item_dict)
    return rows


def embed_texts_with_litellm(
    texts: List[str], embedding_model: str, batch_size: int = 16
) -> List[List[float]]:
    if not _HAS_LITELLM_EMBEDDING:
        raise RuntimeError(
            "litellm embedding support is unavailable. Install `litellm` with embedding support."
        )
    if not embedding_model:
        raise ValueError("`embedding_model` is required for embedding-based metrics.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = litellm_embedding(model=embedding_model, input=batch)
        except Exception as exc:
            msg = str(exc)
            if "NotFoundError" in msg or "NOT_FOUND" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    f"Embedding model `{embedding_model}` is unavailable for the current provider/API. "
                    "For Gemini via LiteLLM, set `PAPER_REDUNDANCY_EMBEDDING_MODEL=gemini/gemini-embedding-001`."
                ) from exc
            raise
        rows = _extract_embedding_rows(response)
        rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
        for row in rows:
            vector = row.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError("Embedding vector is missing or malformed.")
            embeddings.append([float(v) for v in vector])

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
        )
    return embeddings


def pairwise_embedding_cosine_similarities(
    texts: List[str], embedding_model: str, batch_size: int = 16
) -> List[float]:
    if len(texts) < 2:
        return []
    embeddings = embed_texts_with_litellm(
        texts, embedding_model=embedding_model, batch_size=batch_size
    )
    sims: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(_cosine_from_dense(embeddings[i], embeddings[j]))
    return sims


def sentence_similarity(
    pairs: List[Tuple[str, str]],
    mode: str = "auto",
    require_bert_score: bool = False,
) -> List[float]:
    if mode not in {"auto", "bert_score", "cosine"}:
        raise ValueError(
            f"Unsupported similarity mode `{mode}`. Use one of: auto, bert_score, cosine."
        )

    use_bert = mode == "bert_score" or (mode == "auto" and _HAS_BERTSCORE)
    if use_bert:
        if not _HAS_BERTSCORE:
            if require_bert_score:
                raise RuntimeError(
                    "bert-score is required but not installed. "
                    "Install `bert-score` and its torch dependencies."
                )
        else:
            try:
                return bertscore_similarity(pairs)
            except Exception as exc:
                if require_bert_score:
                    raise RuntimeError(
                        "bert-score failed while strict paper alignment is enabled."
                    ) from exc
    if require_bert_score:
        raise RuntimeError(
            "Strict paper alignment requires BERTScore; falling back to cosine is disabled."
        )

    if mode == "bert_score":
        # Non-strict mode: preserve backward compatibility and continue with cosine.
        return [cosine_similarity(a, b) for a, b in pairs]

    if mode == "cosine" or mode == "auto":
        return [cosine_similarity(a, b) for a, b in pairs]

    return [cosine_similarity(a, b) for a, b in pairs]
