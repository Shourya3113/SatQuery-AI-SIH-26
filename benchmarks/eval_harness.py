"""
SatQuery AI - Benchmark Evaluation Harness
Owner: Pradipti (Research, Benchmarks, QA & Pitch Lead)
Calculates mIoU for grounding, BLEU/CIDEr for VQA, and F1-score for change detection.
"""

import numpy as np
from typing import Dict, Any


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Intersection over Union (Jaccard Index) for binary segmentation masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    union_sum = np.sum(union)
    if union_sum == 0:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0
    return float(np.sum(intersection) / union_sum)


def calculate_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes F1-score (Dice coefficient) for change detection masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    total_positives = np.sum(pred_mask) + np.sum(gt_mask)
    if total_positives == 0:
        return 1.0
    return float(2.0 * np.sum(intersection) / total_positives)


def tokenize(text: str) -> list[str]:
    """Standard word tokenizer for evaluation benchmarks."""
    import re
    return re.findall(r"\w+", text.lower())


def calculate_bleu(hypothesis: str, reference: str, n_grams: int = 4) -> float:
    """
    Calculates BLEU score (Papineni et al., 2002) with uniform n-gram weights and brevity penalty.
    """
    import math
    from collections import Counter

    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, n_grams + 1):
        hyp_ngrams = [tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)

        clipped_matches = sum(min(count, ref_counts.get(ng, 0)) for ng, count in hyp_counts.items())
        precisions.append(clipped_matches / len(hyp_ngrams))

    # Brevity penalty
    c = len(hyp_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - (r / c))

    if any(p == 0 for p in precisions):
        return 0.0

    log_sum = sum((1.0 / n_grams) * math.log(p) for p in precisions)
    return float(bp * math.exp(log_sum))
