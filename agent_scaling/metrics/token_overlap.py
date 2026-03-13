from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .similarity import sentence_similarity, tokenize

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]|")


def _split_sentences(text: str) -> List[str]:
    # Simple sentence split; fallback to full text if no punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _token_counts_by_agent(texts: List[str]) -> List[Counter]:
    return [Counter(tokenize(t)) for t in texts]


def compute_token_overlap_metrics(
    texts: List[str],
    contradiction_threshold: float = 0.3,
    similarity_mode: str = "auto",
) -> Dict[str, float]:
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
        sentence_pairs: List[Tuple[str, str]] = []
        sentence_sources: List[Tuple[int, str]] = []
        sentences_by_agent: List[List[str]] = [
            _split_sentences(t) for t in texts
        ]
        for i, sentences_i in enumerate(sentences_by_agent):
            for j in range(i + 1, len(sentences_by_agent)):
                sentences_j = sentences_by_agent[j]
                for sent_i in sentences_i:
                    for sent_j in sentences_j:
                        sentence_pairs.append((sent_i, sent_j))
                        sentence_sources.append((i, sent_i))
                        sentence_sources.append((j, sent_j))

        if sentence_pairs:
            sims = sentence_similarity(sentence_pairs, mode=similarity_mode)
            contradictory_sentence_map: Dict[Tuple[int, str], bool] = {}
            pair_idx = 0
            for (sent_i, sent_j), sim in zip(sentence_pairs, sims):
                is_contradictory = sim < contradiction_threshold
                if is_contradictory:
                    # mark both sentences as contradictory
                    contradictory_sentence_map[(sentence_sources[2 * pair_idx][0], sent_i)] = True
                    contradictory_sentence_map[(sentence_sources[2 * pair_idx + 1][0], sent_j)] = True
                pair_idx += 1

            for agent_idx, sentences in enumerate(sentences_by_agent):
                for sent in sentences:
                    if contradictory_sentence_map.get((agent_idx, sent)):
                        contradictory_tokens += len(tokenize(sent))

    contradictory_ratio = contradictory_tokens / total_tokens

    return {
        "unique_token_ratio": unique_mass / total_tokens,
        "shared_token_ratio": shared_mass / total_tokens,
        "contradictory_token_ratio": contradictory_ratio,
        "shared_token_entropy_bits": shared_entropy,
        "contradictory_mass": contradictory_ratio,
    }
