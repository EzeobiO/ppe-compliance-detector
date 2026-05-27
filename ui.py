"""Gradio Blocks UI definition for the PPE Compliance Detector.

All detection logic is delegated to ``detection`` and ``reports``; this
module is purely responsible for wiring UI components to those functions.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import gradio as gr
from gradio.themes.utils import fonts

from config import (
    ALERT_COOLDOWN_SECONDS,
    DEFAULT_CONFIDENCE,
    FPS_WINDOW_SIZE,
    MAX_HISTORY_ENTRIES,
)
from detection import annotate, extract_detections, summarize
from reports import render_markdown, render_text, write_report_tempfile

logger = logging.getLogger(__name__)

# Plays a short 880 Hz beep via Web Audio API when the hidden #_alert_trigger
# Number component receives a new non-zero value.
_AUDIO_JS = """
<script>
(function () {
    var _lastTrigger = 0;
    setInterval(function () {
        var el = document.querySelector("#_alert_trigger input");
        if (!el) return;
        var val = parseFloat(el.value) || 0;
        if (val !== 0 && val !== _lastTrigger) {
            _lastTrigger = val;
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.frequency.value = 880;
                gain.gain.setValueAtTime(0.3, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.3);
            } catch (e) {}
        }
    }, 250);
})();
</script>
"""


def _render_history(scan_history: list[str]) -> str:
    if not scan_history:
        return "*No violation events recorded yet.*"
    lines = ["**Recent Violation Events:**", ""]
    for entry in reversed(scan_history):
        lines.append(f"- {entry}")
    return "\n".join(lines)


def build_demo(
    model: Any,
    model_type: str,
    class_names: dict[int, str],
) -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Captures ``model``, ``model_type``, and ``class_names`` via closure so
    the internal handlers have no mutable module-level state.
    """

    def _detect_image(
        image: Any,
        confidence: float,
    ) -> tuple[Any, str, str | None]:
        if image is None:
            return None, "Upload an image to begin.", None
        start = time.time()
        results = model(image, verbose=False)
        inference_time = time.time() - start
        detections = extract_detections(results, class_names, confidence)
        annotated_img = annotate(image, detections)
        summary = summarize(detections)
        md = render_markdown(detections, summary, inference_time)
        text = render_text(detections, summary, inference_time)
        path = write_report_tempfile(text)
        return annotated_img, md, path

    def _detect_webcam(
        frame: Any,
        confidence: float,
        state: dict[str, Any] | None,
    ) -> tuple[Any, str, str, float, dict[str, Any]]:
        if frame is None:
            blank_state: dict[str, Any] = {
                "frame_times": [],
                "last_alert_time": 0.0,
                "scan_history": [],
            }
            return None, "Waiting for webcam...", _render_history([]), 0.0, blank_state

        if state is None:
            state = {"frame_times": [], "last_alert_time": 0.0, "scan_history": []}

        start = time.time()
        results = model(frame, verbose=False)
        inference_time = time.time() - start

        detections = extract_detections(results, class_names, confidence)
        annotated_img = annotate(frame, detections)
        summary = summarize(detections)

        state["frame_times"].append(inference_time)
        if len(state["frame_times"]) > FPS_WINDOW_SIZE:
            state["frame_times"].pop(0)
        mean_time = sum(state["frame_times"]) / len(state["frame_times"])
        fps = 1.0 / mean_time if mean_time > 0 else 0.0

        rate_str = (
            f"{summary.compliance_rate:.0f}%"
            if summary.compliance_rate is not None
            else "N/A"
        )
        if summary.compliance_rate is None:
            status_text = "**Scanning...**"
        elif summary.compliance_rate >= 90:
            status_text = "**COMPLIANT**"
        elif summary.compliance_rate >= 70:
            status_text = "**WARNING**"
        else:
            status_text = "**VIOLATIONS DETECTED**"

        live_stats = (
            f"{status_text}\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Compliance | {rate_str} |\n"
            f"| Violations | {summary.violations} |\n"
            f"| FPS | {fps:.1f} |"
        )

        now = time.time()
        alert_trigger = 0.0
        if (
            summary.violations > 0
            and (now - state["last_alert_time"]) >= ALERT_COOLDOWN_SECONDS
        ):
            alert_trigger = now
            state["last_alert_time"] = now

        if summary.violations > 0:
            ts = datetime.now().strftime("%H:%M:%S")
            state["scan_history"].append(
                f"{ts} — {summary.violations} violation(s), {rate_str} compliance"
            )
            if len(state["scan_history"]) > MAX_HISTORY_ENTRIES:
                state["scan_history"].pop(0)

        history_md = _render_history(state["scan_history"])
        return annotated_img, live_stats, history_md, alert_trigger, state

    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="slate",
        font=[fonts.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="PPE Compliance Detector", theme=theme) as demo:
        gr.Markdown(
            "# PPE Compliance Detector\n\n"
            "AI-powered construction site safety monitoring. Upload site photos or use "
            "your webcam to detect PPE violations instantly."
        )
        if model_type == "custom":
            gr.Markdown("**Using custom-trained PPE detection model**")
        else:
            gr.Markdown(
                "**Using fallback YOLOv8s model** — upload `ppe_detector_best.pt` "
                "for PPE-specific detection"
            )

        confidence_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
            label="Confidence Threshold",
            info="Lower = more detections, Higher = more accurate",
        )

        with gr.Tabs():
            with gr.TabItem("Upload Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Site Photo",
                            type="numpy",
                            sources=["upload", "clipboard"],
                        )
                        detect_btn = gr.Button(
                            "Analyze Safety Compliance",
                            variant="primary",
                            size="lg",
                        )
                    with gr.Column():
                        image_output = gr.Image(label="Detection Results")
                report_md = gr.Markdown(label="Compliance Report")
                download_file = gr.File(label="Download Report (.txt)")
                gr.Examples(
                    examples=[
                        ["examples/site_01.jpg", 0.35],
                        ["examples/site_02.jpg", 0.35],
                        ["examples/site_03.jpg", 0.35],
                        ["examples/site_04.jpg", 0.35],
                        ["examples/site_05.jpg", 0.35],
                        ["examples/site_06.jpg", 0.35],
                    ],
                    inputs=[image_input, confidence_slider],
                    outputs=[image_output, report_md, download_file],
                    fn=_detect_image,
                    cache_examples=False,
                    label="Try Example Images",
                )

                detect_btn.click(
                    fn=_detect_image,
                    inputs=[image_input, confidence_slider],
                    outputs=[image_output, report_md, download_file],
                )

            with gr.TabItem("Live Monitoring"):
                gr.Markdown(
                    "> **Real-time PPE monitoring.** Point your camera at workers to "
                    "check compliance instantly."
                )
                webcam_state = gr.State(None)
                alert_trigger_num = gr.Number(
                    value=0, visible=False, elem_id="_alert_trigger"
                )

                with gr.Row():
                    with gr.Column():
                        webcam_input = gr.Image(
                            label="Camera Feed",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                        )
                    with gr.Column():
                        webcam_output = gr.Image(label="Live Detection")
                        live_stats_md = gr.Markdown("**Waiting for camera...**")
                        history_md = gr.Markdown(_render_history([]))

                gr.HTML(_AUDIO_JS)

                webcam_input.stream(
                    fn=_detect_webcam,
                    inputs=[webcam_input, confidence_slider, webcam_state],
                    outputs=[
                        webcam_output,
                        live_stats_md,
                        history_md,
                        alert_trigger_num,
                        webcam_state,
                    ],
                )

        gr.Markdown(
            "---\n\n"
            "### Detection Legend\n\n"
            "| Color | Meaning |\n"
            "|-------|--------|\n"
            "| 🟢 Green | Proper PPE (hardhat, mask, gloves, safety vest) |\n"
            "| 🔴 Red | Violation (NO-Hardhat, NO-Mask, NO-Safety Vest) |\n"
            "| 🟠 Orange | Context (person, cone, machinery, vehicles) |"
        )

    return demo
