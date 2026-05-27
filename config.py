"""Configuration constants for the PPE Compliance Detector.

Centralising these values keeps the magic numbers out of the logic and the
class-ID definitions in one place that's easy to audit against the dataset's
data.yaml.
"""
from __future__ import annotations

# ----- Model -----
MODEL_PATH: str = "ppe_detector_best.pt"
DEFAULT_CONFIDENCE: float = 0.35

# ----- Class IDs (Roboflow Construction Site Safety v30, 25-class variant) -----
# Verified against ppe_detector_best.pt via model.names at runtime.
# Source: data.yaml from version 30 of
# https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
#
#  0: Excavator        |  8: Person          | 16: mini-van
#  1: Gloves           |  9: SUV             | 17: sedan
#  2: Hardhat          | 10: Safety Cone     | 18: semi
#  3: Ladder           | 11: Safety Vest     | 19: trailer
#  4: Mask             | 12: bus             | 20: truck and trailer
#  5: NO-Hardhat       | 13: dump truck      | 21: truck
#  6: NO-Mask          | 14: fire hydrant    | 22: van
#  7: NO-Safety Vest   | 15: machinery       | 23: vehicle
#                                            | 24: wheel loader
CLASS_SAFE_COMPLIANT: frozenset[int] = frozenset({1, 2, 4, 11})
CLASS_VIOLATION: frozenset[int] = frozenset({5, 6, 7})
CLASS_CONTEXT: frozenset[int] = frozenset({0, 3, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24})

# ----- Colors (BGR, for OpenCV) -----
COLOR_SAFE: tuple[int, int, int] = (0, 255, 0)        # Green
COLOR_VIOLATION: tuple[int, int, int] = (0, 0, 255)   # Red
COLOR_CONTEXT: tuple[int, int, int] = (0, 165, 255)   # Orange

# ----- UI / streaming -----
FPS_WINDOW_SIZE: int = 30
ALERT_COOLDOWN_SECONDS: float = 2.0
MAX_HISTORY_ENTRIES: int = 10

# ----- OpenCV drawing -----
BOX_THICKNESS_VIOLATION: int = 3
BOX_THICKNESS_DEFAULT: int = 2
LABEL_FONT_SCALE: float = 0.6
LABEL_FONT_THICKNESS: int = 2
LABEL_PADDING_PX: int = 10
VIOLATION_TAG_FONT_SCALE: float = 0.5

# ----- Compliance thresholds (percent) -----
COMPLIANCE_EXCELLENT_PCT: float = 90.0
COMPLIANCE_WARNING_PCT: float = 70.0
