"""
SatQuery AI - Benchmark Metric Evaluation Tests
Owner: Pradipti (Research, Benchmarks, QA & Pitch Lead)
Tests mIoU, F1-score, and BLEU scoring for remote sensing models.
"""

import numpy as np
import pytest
from benchmarks.eval_harness import calculate_iou, calculate_f1_score, calculate_bleu


def test_calculate_iou_perfect_match():
    mask_a = np.ones((10, 10), dtype=np.uint8)
    mask_b = np.ones((10, 10), dtype=np.uint8)
    iou = calculate_iou(mask_a, mask_b)
    assert iou == 1.0


def test_calculate_iou_partial_match():
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_b = np.zeros((10, 10), dtype=np.uint8)
    mask_a[0:5, 0:5] = 1  # 25 pixels
    mask_b[0:5, 0:2] = 1  # 10 pixels (all intersect)
    iou = calculate_iou(mask_a, mask_b)
    assert iou == pytest.approx(10.0 / 25.0)


def test_calculate_f1_score():
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_b = np.zeros((10, 10), dtype=np.uint8)
    mask_a[0:5, 0:5] = 1  # 25
    mask_b[0:5, 0:5] = 1  # 25
    f1 = calculate_f1_score(mask_a, mask_b)
    assert f1 == 1.0


def test_calculate_bleu_exact_match():
    hyp = "High resolution optical remote sensing image showing agricultural fields"
    ref = "High resolution optical remote sensing image showing agricultural fields"
    score = calculate_bleu(hyp, ref)
    assert score == pytest.approx(1.0, rel=1e-3)


def test_calculate_bleu_partial_match():
    hyp = "Optical remote sensing image with urban buildings"
    ref = "High resolution optical satellite image with urban buildings and roads"
    score = calculate_bleu(hyp, ref, n_grams=2)
    assert 0.0 < score < 1.0
