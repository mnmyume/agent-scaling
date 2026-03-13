from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

from .similarity import sentence_similarity, tokenize

def _split_sentences(text: str) -> List[str]:
    # Simple sentence split; fallback to full text if no punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _token_counts_by_agent(texts: List[str]) -> List[Counter]:
    return [Counter(tokenize(t)) for t in texts]


def compute_token_overlap_metrics(
    texts: List[str],
    contradiction_threshold: float = 0.3,
    similarity_mode: str = "bert_score",
    strict_paper_alignment: bool = False,
) -> Dict[str, float]:
    if strict_paper_alignment and similarity_mode not in {"auto", "bert_score"}:
        raise ValueError(
            "Strict paper alignment requires BERTScore-based contradiction detection "
            "(similarity_mode must be `auto` or `bert_score`)."
        )

    if len(texts) == 0:
        return {
            "unique_token_ratio": 0.0,
            "shared_token_ratio": 0.0,
            "contradictory_token_ratio": 0.0,
            "shared_token_entropy_bits": 0.0,
            "contradictory_mass": 0.0,
        }

    token_counts = _token_counts_by_agent(texts)
    total_tokens = sum(sum(c.values()) for c in token_counts)
    if total_tokens == 0:
        return {
            "unique_token_ratio": 0.0,
            "shared_token_ratio": 0.0,
            "contradictory_token_ratio": 0.0,
            "shared_token_entropy_bits": 0.0,
            "contradictory_mass": 0.0,
        }

    token_agent_counts: Dict[str, Dict[int, int]] = {}
    for agent_idx, counts in enumerate(token_counts):
        for token, count in counts.items():
            token_agent_counts.setdefault(token, {})[agent_idx] = count

    unique_mass = 0
    shared_mass = 0
    entropy_weighted_sum = 0.0
    entropy_weight = 0.0
    for token, agent_counts in token_agent_counts.items():
        token_total = sum(agent_counts.values())
        if len(agent_counts) == 1:
            unique_mass += token_total
        else:
            shared_mass += token_total
            total = token_total
            probs = [c / total for c in agent_counts.values() if total > 0]
            entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
            entropy_weighted_sum += entropy * total
            entropy_weight += total

    shared_entropy = (
        entropy_weighted_sum / entropy_weight if entropy_weight > 0 else 0.0
    )

    contradictory_tokens = 0
    if len(texts) > 1:
        sentences_by_agent: List[List[str]] = [
            _split_sentences(t) for t in texts
        ]
        contradictory_sentence_ids: Set[Tuple[int, int]] = set()

        for i, sentences_i in enumerate(sentences_by_agent):
            for j in range(i + 1, len(sentences_by_agent)):
                sentences_j = sentences_by_agent[j]
                if not sentences_i or not sentences_j:
                    continue

                pair_grid: List[Tuple[str, str]] = [
                    (sent_i, sent_j) for sent_i in sentences_i for sent_j in sentences_j
                ]
                sims = sentence_similarity(
                    pair_grid,
                    mode=similarity_mode,
                    require_bert_score=(
                        strict_paper_alignment and similarity_mode in {"auto", "bert_score"}
                    ),
                )
                if not sims:
                    continue

                best_i = [-1.0] * len(sentences_i)
                best_j = [-1.0] * len(sentences_j)
                idx = 0
                for i_idx in range(len(sentences_i)):
                    for j_idx in range(len(sentences_j)):
                        sim = sims[idx]
                        idx += 1
                        if sim > best_i[i_idx]:
                            best_i[i_idx] = sim
                        if sim > best_j[j_idx]:
                            best_j[j_idx] = sim

                for i_idx, sim in enumerate(best_i):
                    if sim < contradiction_threshold:
                        contradictory_sentence_ids.add((i, i_idx))
                for j_idx, sim in enumerate(best_j):
                    if sim < contradiction_threshold:
                        contradictory_sentence_ids.add((j, j_idx))

        for agent_idx, sent_idx in contradictory_sentence_ids:
            contradictory_tokens += len(tokenize(sentences_by_agent[agent_idx][sent_idx]))

    contradictory_ratio = contradictory_tokens / total_tokens

    return {
        "unique_token_ratio": unique_mass / total_tokens,
        "shared_token_ratio": shared_mass / total_tokens,
        "contradictory_token_ratio": contradictory_ratio,
        "shared_token_entropy_bits": shared_entropy,
        "contradictory_mass": contradictory_ratio,
    }
