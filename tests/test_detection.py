"""Tests for detection.py: classify_by_id, summarize, extract_detections."""
from __future__ import annotations

import pytest

from detection import Detection, classify_by_id, extract_detections, summarize


# ---------------------------------------------------------------------------
# Classify by ID
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_id", [1, 2, 4, 11])
def test_classify_safe_ids(cls_id: int) -> None:
    assert classify_by_id(cls_id) == "safe"


@pytest.mark.parametrize("cls_id", [5, 6, 7])
def test_classify_violation_ids(cls_id: int) -> None:
    assert classify_by_id(cls_id) == "violation"


@pytest.mark.parametrize("cls_id", [0, 3, 8, 9, 10, 15, 23])
def test_classify_context_ids(cls_id: int) -> None:
    assert classify_by_id(cls_id) == "context"


@pytest.mark.parametrize("cls_id", [99, -1, 1000])
def test_classify_unknown_defaults_to_context(cls_id: int) -> None:
    assert classify_by_id(cls_id) == "context"


# ---------------------------------------------------------------------------
# Summary Helpers
# ---------------------------------------------------------------------------

def _det(status: str) -> Detection:
    return Detection(
        class_name="test",
        class_id=0,
        confidence=0.9,
        status=status,
        bbox=(0, 0, 10, 10),
    )


# ---------------------------------------------------------------------------
# Summary Test
# ---------------------------------------------------------------------------

def test_summarize_empty_list() -> None:
    s = summarize([])
    assert s.total == 0
    assert s.safe == 0
    assert s.violations == 0
    assert s.context == 0
    assert s.compliance_rate is None


def test_summarize_all_safe() -> None:
    s = summarize([_det("safe"), _det("safe")])
    assert s.safe == 2
    assert s.violations == 0
    assert s.compliance_rate == pytest.approx(100.0)


def test_summarize_all_violation() -> None:
    s = summarize([_det("violation"), _det("violation")])
    assert s.violations == 2
    assert s.safe == 0
    assert s.compliance_rate == pytest.approx(0.0)


def test_summarize_mixed() -> None:
    dets = [_det("safe")] * 3 + [_det("violation")]
    s = summarize(dets)
    assert s.safe == 3
    assert s.violations == 1
    assert s.compliance_rate == pytest.approx(75.0)


def test_summarize_only_context_has_no_rate() -> None:
    s = summarize([_det("context"), _det("context")])
    assert s.context == 2
    assert s.safe == 0
    assert s.violations == 0
    assert s.compliance_rate is None


def test_summarize_counts_total() -> None:
    dets = [_det("safe"), _det("violation"), _det("context")]
    s = summarize(dets)
    assert s.total == 3


# ---------------------------------------------------------------------------
# Extract_detections using lightweight fakes 
# ---------------------------------------------------------------------------

class _FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: tuple[int, int, int, int]) -> None:
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


_NAMES: dict[int, str] = {2: "Hardhat", 5: "NO-Hardhat", 8: "Person"}


def test_extract_detections_returns_correct_fields() -> None:
    box = _FakeBox(cls_id=2, conf=0.9, xyxy=(10, 20, 100, 200))
    dets = extract_detections([_FakeResult([box])], _NAMES, confidence_threshold=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_name == "Hardhat"
    assert d.class_id == 2
    assert d.status == "safe"
    assert d.bbox == (10, 20, 100, 200)
    assert d.confidence == pytest.approx(0.9)


def test_extract_detections_filters_low_confidence() -> None:
    low = _FakeBox(cls_id=2, conf=0.3, xyxy=(0, 0, 10, 10))
    high = _FakeBox(cls_id=5, conf=0.8, xyxy=(0, 0, 10, 10))
    dets = extract_detections([_FakeResult([low, high])], _NAMES, confidence_threshold=0.5)
    assert len(dets) == 1
    assert dets[0].class_id == 5


def test_extract_detections_unknown_class_name_fallback() -> None:
    box = _FakeBox(cls_id=99, conf=0.9, xyxy=(0, 0, 10, 10))
    dets = extract_detections([_FakeResult([box])], _NAMES, confidence_threshold=0.5)
    assert dets[0].class_name == "class_99"


def test_extract_detections_empty_results() -> None:
    dets = extract_detections([], _NAMES, confidence_threshold=0.5)
    assert dets == []


def test_extract_detections_result_without_boxes() -> None:
    class _NoBoxes:
        boxes = None

    dets = extract_detections([_NoBoxes()], _NAMES, confidence_threshold=0.5)
    assert dets == []
