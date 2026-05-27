"""YOLO model loading with explicit error handling.

The previous app.py used a bare `except:` on model loading, which silently
fell back to vanilla YOLOv8s for any error — including unrelated bugs.
The fallback model doesn't have PPE classes, so the app would appear to
work but produce useless detections. This module makes the failure modes
explicit and logged.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from ultralytics import YOLO

from config import MODEL_PATH

logger = logging.getLogger(__name__)

ModelType = Literal["custom", "fallback"]


def load_model(model_path: str = MODEL_PATH) -> tuple[YOLO, ModelType]:
    """Load the PPE detection model.

    Tries the custom fine-tuned weights first; if they're missing or fail
    to load, falls back to vanilla YOLOv8s and logs a prominent warning.
    The caller receives a ``ModelType`` flag so the UI can display the
    correct banner.

    Returns:
        A ``(model, model_type)`` tuple where ``model_type`` is either
        ``"custom"`` or ``"fallback"``.

    Raises:
        RuntimeError: when neither the custom nor the fallback model
            can be loaded (e.g. no network access for the YOLOv8s
            download and no local cache).
    """
    custom_path = Path(model_path)

    if custom_path.exists():
        try:
            model = YOLO(str(custom_path))
            logger.info("Loaded custom PPE model from %s", custom_path)
            return model, "custom"
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Custom model at %s exists but failed to load: %s. "
                "Falling back to vanilla YOLOv8s.",
                custom_path,
                exc,
            )
    else:
        logger.warning(
            "Custom PPE model not found at %s. Falling back to vanilla "
            "YOLOv8s — detections will NOT be PPE-specific.",
            custom_path,
        )

    try:
        return YOLO("yolov8s.pt"), "fallback"
    except Exception as exc:  # last-resort: nothing usable
        raise RuntimeError(
            "Neither the custom PPE model nor the fallback YOLOv8s "
            f"could be loaded: {exc}"
        ) from exc
