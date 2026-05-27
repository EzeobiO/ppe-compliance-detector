"""Detection extraction, ID-based classification, annotation, and compliance scoring.

Key design choice: classification is done by class ID against an explicit
allowlist (see ``config.py``) rather than substring matching on class
names. The previous string-based approach mis-classified ``NO-Mask`` as
a non-violation because the safe/violation logic relied on the order of
substring checks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

import cv2
import numpy as np

from config import (
    BOX_THICKNESS_DEFAULT,
    BOX_THICKNESS_VIOLATION,
    CLASS_CONTEXT,
    CLASS_SAFE_COMPLIANT,
    CLASS_VIOLATION,
    COLOR_CONTEXT,
    COLOR_SAFE,
    COLOR_VIOLATION,
    LABEL_FONT_SCALE,
    LABEL_FONT_THICKNESS,
    LABEL_PADDING_PX,
    VIOLATION_TAG_FONT_SCALE,
)

Status = Literal["safe", "violation", "context"]


@dataclass(frozen=True)
class Detection:
    """One detection produced by the YOLO model, post-thresholding."""

    class_name: str
    class_id: int
    confidence: float
    status: Status
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class ComplianceSummary:
    """Aggregate counts and the computed compliance rate for a frame.

    ``compliance_rate`` is ``None`` when there are no PPE-relevant
    detections (i.e. only context classes, or nothing at all). This
    avoids the misleading "100%" the previous implementation reported
    on empty frames.
    """

    total: int
    safe: int
    violations: int
    context: int
    compliance_rate: float | None


_STATUS_COLOR: dict[Status, tuple[int, int, int]] = {
    "safe": COLOR_SAFE,
    "violation": COLOR_VIOLATION,
    "context": COLOR_CONTEXT,
}


def classify_by_id(class_id: int) -> Status:
    """Classify a detection by class ID against the config's allowlists.

    Unknown IDs default to ``"context"`` so a model with extra classes
    (or a fallback non-PPE model) doesn't poison the compliance score
    with spurious violations.
    """
    if class_id in CLASS_VIOLATION:
        return "violation"
    if class_id in CLASS_SAFE_COMPLIANT:
        return "safe"
    if class_id in CLASS_CONTEXT:
        return "context"
    return "context"


def extract_detections(
    results: Iterable,
    class_names: dict[int, str],
    confidence_threshold: float,
) -> list[Detection]:
    """Extract structured ``Detection`` records from YOLO results.

    Detections below ``confidence_threshold`` are filtered out.
    ``class_names`` is the model's own ``model.names`` dict; if a class
    ID isn't in it (shouldn't happen but be defensive), the name falls
    back to ``"class_{id}"``.
    """
    detections: list[Detection] = []
    for result in results:
        if getattr(result, "boxes", None) is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_name=class_names.get(cls_id, f"class_{cls_id}"),
                    class_id=cls_id,
                    confidence=conf,
                    status=classify_by_id(cls_id),
                    bbox=(x1, y1, x2, y2),
                )
            )
    return detections


def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw color-coded bounding boxes and labels on a copy of the image."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = _STATUS_COLOR[det.status]
        thickness = (
            BOX_THICKNESS_VIOLATION if det.status == "violation"
            else BOX_THICKNESS_DEFAULT
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.class_name}: {det.confidence:.0%}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - LABEL_PADDING_PX),
            (x1 + label_w + 5, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
            (255, 255, 255), LABEL_FONT_THICKNESS,
        )
        if det.status == "violation":
            cv2.putText(
                annotated, "VIOLATION", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, VIOLATION_TAG_FONT_SCALE,
                color, 2,
            )
    return annotated


def summarize(detections: list[Detection]) -> ComplianceSummary:
    """Compute counts and the compliance rate for a list of detections."""
    counts = Counter(d.status for d in detections)
    safe = counts.get("safe", 0)
    violations = counts.get("violation", 0)
    context = counts.get("context", 0)
    relevant = safe + violations
    rate: float | None = (safe / relevant * 100.0) if relevant > 0 else None
    return ComplianceSummary(
        total=len(detections),
        safe=safe,
        violations=violations,
        context=context,
        compliance_rate=rate,
    )
