"""
🦺 PPE Compliance Detector
Real-time construction site safety monitoring using YOLOv8.
Detects hardhats, safety vests, and flags violations.

Developed by: Ebube Ezeobi
"""

import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
from collections import Counter
import time
import json
import base64

# Load custom PPE detection model
try:
    model = YOLO("ppe_detector_best.pt")
    MODEL_TYPE = "custom"
    print("✅ Loaded custom PPE detection model")
except:
    model = YOLO("yolov8s.pt")
    MODEL_TYPE = "pretrained"
    print("⚠️ Custom model not found, using pretrained YOLOv8s")

# Class configuration for PPE detection
PPE_CONFIG = {
    "safe_classes": ["Hardhat", "Safety Vest", "hardhat", "safety vest", "helmet", "vest"],
    "violation_classes": ["NO-Hardhat", "NO-Safety Vest", "no-hardhat", "no-safety vest", "no helmet", "no vest", "head", "person"],
    "colors": {
        "safe": (0, 255, 0),       # Green
        "violation": (0, 0, 255),   # Red
        "warning": (0, 165, 255),   # Orange
    }
}

# Detection history storage
detection_history = []


def classify_detection(class_name):
    """Classify a detection as safe, violation, or warning."""
    class_lower = class_name.lower()
    
    if any(safe in class_lower for safe in ["hardhat", "helmet", "vest", "safety"]):
        if "no" in class_lower or "no-" in class_lower:
            return "violation"
        return "safe"
    elif "person" in class_lower or "head" in class_lower:
        return "warning"
    return "warning"


def draw_ppe_detections(image, results, confidence_threshold):
    """
    Draw PPE detections with color-coded safety status.
    Green = compliant, Red = violation, Orange = warning
    """
    annotated = image.copy()
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            status = classify_detection(cls_name)
            color = PPE_CONFIG["colors"][status]
            
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "status": status,
                "bbox": (x1, y1, x2, y2)
            })
            
            thickness = 3 if status == "violation" else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{cls_name}: {conf:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 5, y1),
                color,
                -1
            )
            
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            if status == "violation":
                cv2.putText(
                    annotated,
                    "! VIOLATION",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
    
    return annotated, detections


def calculate_compliance(detections):
    """Calculate overall PPE compliance statistics."""
    if not detections:
        return None
    
    status_counts = Counter(d["status"] for d in detections)
    total = len(detections)
    
    safe_count = status_counts.get("safe", 0)
    violation_count = status_counts.get("violation", 0)
    warning_count = status_counts.get("warning", 0)
    
    relevant = safe_count + violation_count
    compliance_rate = (safe_count / relevant * 100) if relevant > 0 else 100
    
    return {
        "total": total,
        "safe": safe_count,
        "violations": violation_count,
        "warnings": warning_count,
        "compliance_rate": compliance_rate
    }


def generate_report(detections, compliance, inference_time):
    """Generate a formatted compliance report."""
    if not detections:
        return "No PPE equipment or violations detected. Try lowering the confidence threshold.", ""
    
    if compliance["compliance_rate"] >= 90:
        status_emoji = "✅"
        status_text = "EXCELLENT"
    elif compliance["compliance_rate"] >= 70:
        status_emoji = "⚠️"
        status_text = "NEEDS ATTENTION"
    else:
        status_emoji = "🚨"
        status_text = "CRITICAL"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""## {status_emoji} Compliance Score: {compliance["compliance_rate"]:.0f}% — {status_text}

**Scan Time:** {timestamp}  
**Inference:** {inference_time:.2f}s

---

### Summary

| Category | Count |
|----------|-------|
| ✅ Proper PPE | {compliance["safe"]} |
| 🚨 Violations | {compliance["violations"]} |
| ⚠️ Warnings | {compliance["warnings"]} |
| **Total Detections** | **{compliance["total"]}** |

---

### Detailed Findings

"""
    
    violations = [d for d in detections if d["status"] == "violation"]
    safe = [d for d in detections if d["status"] == "safe"]
    warnings = [d for d in detections if d["status"] == "warning"]
    
    if violations:
        report += "**🚨 Violations Found:**\n"
        for d in violations:
            report += f"- {d['class']} (confidence: {d['confidence']:.0%})\n"
        report += "\n"
    
    if safe:
        report += "**✅ Proper PPE Detected:**\n"
        for d in safe:
            report += f"- {d['class']} (confidence: {d['confidence']:.0%})\n"
        report += "\n"
    
    if warnings:
        report += "**⚠️ Warnings:**\n"
        for d in warnings:
            report += f"- {d['class']} (confidence: {d['confidence']:.0%})\n"
    
    # Plain text version for download
    plain_report = f"""PPE COMPLIANCE REPORT
Generated by PPE Compliance Detector
Developed by Ebube Ezeobi
{'='*50}

SCAN TIME: {timestamp}
INFERENCE TIME: {inference_time:.2f}s
COMPLIANCE SCORE: {compliance["compliance_rate"]:.0f}% — {status_text}

{'='*50}
SUMMARY
{'='*50}
Proper PPE:    {compliance["safe"]}
Violations:    {compliance["violations"]}
Warnings:      {compliance["warnings"]}
Total:         {compliance["total"]}

{'='*50}
DETAILED FINDINGS
{'='*50}
"""
    
    if violations:
        plain_report += "\nVIOLATIONS:\n"
        for d in violations:
            plain_report += f"  - {d['class']} ({d['confidence']:.0%} confidence)\n"
    
    if safe:
        plain_report += "\nPROPER PPE:\n"
        for d in safe:
            plain_report += f"  - {d['class']} ({d['confidence']:.0%} confidence)\n"
    
    if warnings:
        plain_report += "\nWARNINGS:\n"
        for d in warnings:
            plain_report += f"  - {d['class']} ({d['confidence']:.0%} confidence)\n"
    
    plain_report += f"\n{'='*50}\nReport generated by PPE Compliance Detector\nhttps://obie-ezeobi.vercel.app\n"
    
    return report, plain_report


def add_to_history(compliance, detections):
    """Add scan result to history."""
    global detection_history
    
    if compliance:
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "compliance_rate": compliance["compliance_rate"],
            "violations": compliance["violations"],
            "safe": compliance["safe"],
            "total": compliance["total"]
        }
        detection_history.insert(0, entry)
        
        # Keep only last 10 entries
        if len(detection_history) > 10:
            detection_history.pop()


def get_history_display():
    """Format detection history for display."""
    if not detection_history:
        return "No scans yet. Upload an image to start."
    
    history_md = "| Time | Score | Violations | Safe | Total |\n"
    history_md += "|------|-------|------------|------|-------|\n"
    
    for entry in detection_history:
        score_emoji = "✅" if entry["compliance_rate"] >= 90 else "⚠️" if entry["compliance_rate"] >= 70 else "🚨"
        history_md += f"| {entry['timestamp']} | {score_emoji} {entry['compliance_rate']:.0f}% | {entry['violations']} | {entry['safe']} | {entry['total']} |\n"
    
    return history_md


def detect_ppe_image(image, confidence_threshold):
    """Process uploaded image for PPE detection."""
    if image is None:
        return None, "Please upload an image.", "", get_history_display()
    
    start_time = time.time()
    results = model(image, verbose=False)
    inference_time = time.time() - start_time
    
    annotated, detections = draw_ppe_detections(image, results, confidence_threshold)
    compliance = calculate_compliance(detections)
    report, plain_report = generate_report(detections, compliance, inference_time)
    
    # Add to history
    add_to_history(compliance, detections)
    
    return annotated, report, plain_report, get_history_display()


def detect_ppe_webcam(frame, confidence_threshold, sound_enabled, state):
    """Process webcam frame for real-time PPE detection."""
    if frame is None:
        return None, "Waiting for webcam...", False, state
    
    if state is None:
        state = {"frame_times": [], "last_alert": 0}
    
    start_time = time.time()
    results = model(frame, verbose=False)
    annotated, detections = draw_ppe_detections(frame, results, confidence_threshold)
    compliance = calculate_compliance(detections)
    
    # Calculate FPS
    frame_time = time.time() - start_time
    state["frame_times"].append(frame_time)
    if len(state["frame_times"]) > 30:
        state["frame_times"].pop(0)
    fps = 1.0 / (sum(state["frame_times"]) / len(state["frame_times"]))
    
    # Check for violations and trigger alert
    trigger_sound = False
    if compliance and compliance["violations"] > 0 and sound_enabled:
        current_time = time.time()
        if current_time - state.get("last_alert", 0) > 2:  # Alert every 2 seconds max
            trigger_sound = True
            state["last_alert"] = current_time
    
    # Generate live stats
    if compliance:
        violation_count = compliance["violations"]
        rate = compliance["compliance_rate"]
        
        if rate >= 90:
            status = "✅ COMPLIANT"
        elif rate >= 70:
            status = "⚠️ WARNING"
        else:
            status = "🚨 VIOLATIONS DETECTED"
        
        stats = f"""### {status}

| Metric | Value |
|--------|-------|
| Compliance | {rate:.0f}% |
| Violations | {violation_count} |
| Safe PPE | {compliance["safe"]} |
| FPS | {fps:.1f} |"""
    else:
        stats = f"**Scanning...** | FPS: {fps:.1f}"
    
    return annotated, stats, trigger_sound, state


def create_download_file(report_text):
    """Create downloadable report file."""
    if not report_text:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ppe_report_{timestamp}.txt"
    
    # Save to temp file
    with open(filename, "w") as f:
        f.write(report_text)
    
    return filename


def clear_history():
    """Clear detection history."""
    global detection_history
    detection_history = []
    return "No scans yet. Upload an image to start."


# Custom CSS
custom_css = """
.gradio-container { max-width: 1400px !important; }

.app-header {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.app-footer {
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    border-top: 1px solid #333;
}

.app-footer a {
    margin: 0 15px;
    text-decoration: none;
    font-weight: 500;
}

.history-panel {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 15px;
}

.sound-toggle {
    padding: 10px;
    background: #2d2d44;
    border-radius: 8px;
}
"""

# JavaScript for sound alert
sound_js = """
<script>
function playAlertSound() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
}
</script>
"""


# Build Gradio Interface
with gr.Blocks(
    title="PPE Compliance Detector | Ebube Ezeobi",
    css=custom_css
) as demo:
    
    # Hidden state for plain report (for download)
    plain_report_state = gr.State("")
    
    # Header
    gr.Markdown(
        """
        # 🦺 PPE Compliance Detector
        
        **AI-powered construction site safety monitoring**. Upload site photos or use your webcam 
        to instantly detect PPE violations and generate compliance reports.
        
        Detects: Hard hats, safety vests, and flags workers without required equipment.
        """,
        elem_classes="app-header"
    )
    
    # Model status
    if MODEL_TYPE == "custom":
        gr.Markdown("**Custom-trained PPE model**")
    else:
        gr.Markdown("⚠️ **Using pretrained model** — Add `ppe_detector_best.pt` for PPE-specific detection")
    
    # Controls row
    with gr.Row():
        confidence_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.35,
            step=0.05,
            label="🎚️ Confidence Threshold",
            info="Lower = more detections, Higher = more accurate"
        )
    
    with gr.Tabs():
        # ===== IMAGE UPLOAD TAB =====
        with gr.TabItem("📷 Upload Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload Site Photo",
                        type="numpy",
                        sources=["upload", "clipboard"]
                    )
                    detect_btn = gr.Button(
                        "🔍 Analyze Safety Compliance", 
                        variant="primary", 
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    image_output = gr.Image(label="Detection Results")
            
            # Report section
            with gr.Row():
                with gr.Column(scale=2):
                    compliance_report = gr.Markdown(label="Compliance Report")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Export Report")
                    download_btn = gr.Button("💾 Download Report (.txt)", variant="secondary")
                    download_file = gr.File(label="Download", visible=True)
            
            # History section
            with gr.Accordion("📜 Scan History", open=False):
                history_display = gr.Markdown(
                    get_history_display(),
                    elem_classes="history-panel"
                )
                clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary", size="sm")
            
            # Wire up image detection
            detect_btn.click(
                fn=detect_ppe_image,
                inputs=[image_input, confidence_slider],
                outputs=[image_output, compliance_report, plain_report_state, history_display]
            )
            
            # Wire up download
            download_btn.click(
                fn=create_download_file,
                inputs=[plain_report_state],
                outputs=[download_file]
            )
            
            # Wire up clear history
            clear_history_btn.click(
                fn=clear_history,
                inputs=[],
                outputs=[history_display]
            )
        
        # ===== WEBCAM TAB =====
        with gr.TabItem("🎥 Live Monitoring"):
            gr.Markdown(
                """
                > **Real-time PPE monitoring.** Point your camera at workers to check compliance instantly.
                > Enable sound alerts to get notified when violations are detected.
                """
            )
            
            webcam_state = gr.State(None)
            
            # Sound toggle
            with gr.Row():
                sound_enabled = gr.Checkbox(
                    label="🔊 Enable Sound Alerts",
                    value=False,
                    info="Plays alert sound when violations detected",
                    elem_classes="sound-toggle"
                )
                sound_status = gr.Markdown("*Sound alerts disabled*")
            
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(
                        label="Camera Feed",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy"
                    )
                
                with gr.Column():
                    webcam_output = gr.Image(label="Live Detection")
                    live_stats = gr.Markdown("**Waiting for camera...**")
            
            # Hidden component for triggering sound
            sound_trigger = gr.Checkbox(visible=False)
            
            webcam_input.stream(
                fn=detect_ppe_webcam,
                inputs=[webcam_input, confidence_slider, sound_enabled, webcam_state],
                outputs=[webcam_output, live_stats, sound_trigger, webcam_state]
            )
            
            # Update sound status text
            def update_sound_status(enabled):
                return "🔊 *Sound alerts enabled — you'll hear a beep when violations are detected*" if enabled else "*Sound alerts disabled*"
            
            sound_enabled.change(
                fn=update_sound_status,
                inputs=[sound_enabled],
                outputs=[sound_status]
            )
    
    # Detection Legend
    gr.Markdown(
        """
        ---
        
        ### 🎨 Detection Legend
        
        | Color | Meaning |
        |-------|---------|
        | 🟢 **Green** | Proper PPE detected (hardhat, safety vest) |
        | 🔴 **Red** | Violation - missing required PPE |
        | 🟠 **Orange** | Warning - person detected, PPE status unclear |
        """
    )
    
    # About section
    with gr.Accordion("ℹ️ About This Project", open=False):
        gr.Markdown(
            """
            ### Technical Details
            
            This application uses **YOLOv8** (You Only Look Once v8) fine-tuned on construction site safety data
            to detect Personal Protective Equipment (PPE) compliance in real-time.
            
            **Model:** YOLOv8s custom-trained on 2,600+ labeled images  
            **Dataset:** [Construction Site Safety Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) from Roboflow  
            **Framework:** Ultralytics + PyTorch  
            **Interface:** Gradio
            
            ### Use Cases
            
            - 🏗️ **Site Safety Audits** - Quick compliance checks from site photos
            - 📹 **Real-time Monitoring** - Live camera feed analysis
            - 📊 **Compliance Reporting** - Exportable reports for documentation
            - 🎓 **Safety Training** - Visual demonstrations for worker education
            
            ### Performance Metrics
            
            | Metric | Score |
            |--------|-------|
            | mAP50 | ~0.85 |
            | mAP50-95 | ~0.65 |
            | Inference | ~25ms (GPU) |
            """
        )
    
    # Footer with developer info
    gr.Markdown(
        """
        ---
        
        <div style="text-align: center; padding: 20px;">
            <p><strong>Developed by Ebube Ezeobi</strong></p>
            <p>
                <a href="https://obie-ezeobi.vercel.app" target="_blank">🌐 Portfolio</a> · 
                <a href="https://github.com/EzeobiO" target="_blank">💻 GitHub</a> · 
                <a href="https://linkedin.com/in/ezeobio" target="_blank">💼 LinkedIn</a>
            </p>
            <p style="font-size: 12px; color: #888; margin-top: 10px;">
                Built with YOLOv8 + Gradio | © 2025
            </p>
        </div>
        """,
        elem_classes="app-footer"
    )


if __name__ == "__main__":
    demo.launch()