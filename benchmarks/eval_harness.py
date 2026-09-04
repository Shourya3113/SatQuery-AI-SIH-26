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
