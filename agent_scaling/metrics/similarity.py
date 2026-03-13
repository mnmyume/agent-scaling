from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from bert_score import score as bert_score  # type: ignore

    _HAS_BERTSCORE = True
except Exception:
    bert_score = None
    _HAS_BERTSCORE = False


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


def sentence_similarity(
    pairs: List[Tuple[str, str]], mode: str = "auto"
) -> List[float]:
    if mode == "bert_score" or (mode == "auto" and _HAS_BERTSCORE):
        try:
            return bertscore_similarity(pairs)
        except Exception:
            pass
    return [cosine_similarity(a, b) for a, b in pairs]
