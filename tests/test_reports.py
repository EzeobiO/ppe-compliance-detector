#Tests for reports.py: render_markdown, render_text, write_report_tempfile.
from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from detection import ComplianceSummary, Detection, Status
from reports import render_markdown, render_text, write_report_tempfile


def _det(status: Status, class_name: str = "Hardhat", conf: float = 0.9) -> Detection:
    return Detection(
        class_name=class_name,
        class_id=0,
        confidence=conf,
        status=status,
        bbox=(0, 0, 100, 100),
    )


def _summary(safe: int = 0, violations: int = 0, context: int = 0) -> ComplianceSummary:
    relevant = safe + violations
    rate = (safe / relevant * 100.0) if relevant > 0 else None
    return ComplianceSummary(
        total=safe + violations + context,
        safe=safe,
        violations=violations,
        context=context,
        compliance_rate=rate,
    )


# render_markdown

def test_render_markdown_empty_detections() -> None:
    result = render_markdown([], _summary(), 0.1)
    assert "No detections" in result


def test_render_markdown_contains_compliance_rate() -> None:
    dets = [_det("safe"), _det("violation")]
    result = render_markdown(dets, _summary(safe=1, violations=1), 0.05)
    assert "50%" in result


def test_render_markdown_contains_summary_table_headers() -> None:
    dets = [_det("safe"), _det("violation")]
    result = render_markdown(dets, _summary(safe=1, violations=1), 0.05)
    assert "Proper PPE" in result
    assert "Violations" in result


def test_render_markdown_contains_detection_class_names() -> None:
    dets = [_det("safe", "Hardhat"), _det("violation", "NO-Hardhat")]
    result = render_markdown(dets, _summary(safe=1, violations=1), 0.05)
    assert "Hardhat" in result
    assert "NO-Hardhat" in result


def test_render_markdown_excellent_label() -> None:
    dets = [_det("safe"), _det("safe")]
    result = render_markdown(dets, _summary(safe=2, violations=0), 0.05)
    assert "EXCELLENT" in result


def test_render_markdown_critical_label() -> None:
    dets = [_det("violation"), _det("violation")]
    result = render_markdown(dets, _summary(safe=0, violations=2), 0.05)
    assert "CRITICAL" in result


# render_text

def test_render_text_empty_detections() -> None:
    result = render_text([], _summary(), 0.1)
    assert "No detections" in result


def test_render_text_contains_timestamp_header() -> None:
    dets = [_det("safe")]
    result = render_text(dets, _summary(safe=1), 0.05)
    assert "SCAN TIME:" in result


def test_render_text_contains_compliance_rate() -> None:
    dets = [_det("safe"), _det("violation")]
    result = render_text(dets, _summary(safe=1, violations=1), 0.05)
    assert "50%" in result


def test_render_text_contains_detection_class_names() -> None:
    dets = [_det("safe", "Safety Vest"), _det("violation", "NO-Hardhat")]
    result = render_text(dets, _summary(safe=1, violations=1), 0.05)
    assert "Safety Vest" in result
    assert "NO-Hardhat" in result


def test_render_text_contains_inference_time() -> None:
    dets = [_det("safe")]
    result = render_text(dets, _summary(safe=1), 0.123)
    assert "INFERENCE TIME:" in result


# write_report_tempfile

@pytest.fixture
def written_report() -> Iterator[tuple[str, str]]:
    content = "PPE Test Report\nLine two."
    path = write_report_tempfile(content)
    yield path, content
    if os.path.exists(path):
        os.remove(path)


def test_write_report_tempfile_file_exists(written_report: tuple[str, str]) -> None:
    path, _ = written_report
    assert os.path.exists(path)


def test_write_report_tempfile_content_matches(written_report: tuple[str, str]) -> None:
    path, content = written_report
    with open(path, encoding="utf-8") as f:
        assert f.read() == content


def test_write_report_tempfile_has_txt_extension(written_report: tuple[str, str]) -> None:
    path, _ = written_report
    assert path.endswith(".txt")
