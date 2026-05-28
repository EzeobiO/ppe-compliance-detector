"""Main entry point for the PPE Compliance Detector.
Loads the model, builds the Gradio UI, and launches the app.
"""
from __future__ import annotations

import logging

from gradio.themes import Soft
from gradio.themes.utils import fonts

from model import load_model
from ui import build_demo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    model, model_type = load_model()
    class_names: dict[int, str] = model.names  # type: ignore[assignment]
    logger.info("Model type: %s | Classes: %d", model_type, len(class_names))
    demo = build_demo(model, model_type, class_names)
    theme = Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="slate",
        font=[fonts.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )
    demo.launch(
        theme=theme,
        css=".gradio-container { max-width: 1400px !important; }",
    )


if __name__ == "__main__":
    main()
