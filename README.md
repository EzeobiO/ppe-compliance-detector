---
title: PPE Compliance Detector
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.4.0
python_version: '3.11'
app_file: app.py
pinned: false
license: mit
tags:
  - object-detection
  - yolov8
  - computer-vision
  - safety
  - ppe-detection
  - construction
---

# PPE Compliance Detector

Real-time construction site safety monitoring powered by a custom-trained YOLOv8s model. Detects Personal Protective Equipment (PPE) compliance and flags violations instantly.

**Developed by [Ebube Ezeobi](https://obieezeobi.vercel.app)**

[![CI](https://github.com/EzeobiO/ppe-compliance-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/EzeobiO/ppe-compliance-detector/actions/workflows/ci.yml)
[![Deploy](https://github.com/EzeobiO/ppe-compliance-detector/actions/workflows/deploy.yml/badge.svg)](https://github.com/EzeobiO/ppe-compliance-detector/actions/workflows/deploy.yml)
[![Hugging Face](https://img.shields.io/badge/🤗%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/Ezeobi-O/ppe-compliance-detector)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-Custom%20Trained-blue)
![Gradio](https://img.shields.io/badge/Gradio-6.4.0-orange)
![Python](https://img.shields.io/badge/Python-3.11-green)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## What It Detects

| Detection | Status | Visual |
|-----------|--------|--------|
| Hardhat | Compliant | Green box |
| Mask | Compliant | Green box |
| Safety Vest | Compliant | Green box |
| NO-Hardhat | Violation | Red box |
| NO-Mask | Violation | Red box |
| NO-Safety Vest | Violation | Red box |
| Person, Safety Cone, machinery, vehicle | Context | Orange box |

---

## Features

| Feature | Description |
|---------|-------------|
| **Image Upload** | Analyze site photos for PPE compliance |
| **Live Webcam** | Real-time monitoring with FPS counter |
| **Compliance Scoring** | Percentage-based safety rating (safe ÷ relevant detections) |
| **Markdown Report** | In-UI breakdown of all detections and violations |
| **Download Report** | Export compliance report as a plain-text `.txt` file |
| **Scan History** | Session log of the last 10 violation events |
| **Confidence Slider** | Adjustable detection threshold (default 0.35) |

---

## Quick Start

### Run Locally

```bash
git clone https://github.com/EzeobiO/ppe-compliance-detector
cd ppe-compliance-detector
pip install -r requirements.txt
python app.py
```

The app will open at `http://localhost:7860`.

### Try the Demo

1. Upload a construction site image (or enable webcam)
2. View detected PPE with color-coded bounding boxes
3. Read the compliance score and detailed report
4. Download the report as a `.txt` file

---

## Model Performance

Trained on [Roboflow Construction Site Safety v30](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety).

| Metric | Score |
|--------|-------|
| mAP50 | ~0.85 |
| mAP50-95 | ~0.65 |
| Inference | ~25 ms (T4 GPU) |

See [`MODEL_CARD.md`](MODEL_CARD.md) for full training details and limitations.

---

## Project Structure

```
ppe-compliance-detector/
├── app.py                   # Entry point: load model, launch UI (~25 lines)
├── config.py                # All constants and class-ID allowlists
├── model.py                 # load_model() with explicit error handling
├── detection.py             # Detection dataclass, classify_by_id, annotate, summarize
├── reports.py               # render_markdown, render_text, write_report_tempfile
├── ui.py                    # Gradio Blocks definition (build_demo)
├── tests/
│   ├── test_detection.py    # Tests for classify_by_id, summarize, extract_detections
│   └── test_reports.py      # Tests for render_markdown, render_text, write_report_tempfile
├── .github/workflows/
│   ├── ci.yml               # Ruff lint + pytest on every push / PR (reusable)
│   └── deploy.yml           # Deploys to HF Space on push to main (calls ci.yml)
├── ppe_detector_best.pt     # Fine-tuned model weights
├── train_ppe_detector.ipynb # Training notebook (Google Colab)
├── MODEL_CARD.md            # Model card: dataset, classes, metrics, limitations
├── requirements.txt         # Pinned dependencies
└── LICENSE                  # MIT
```

---

## Deployment
The live Space: [huggingface.co/spaces/Ezeobi-O/ppe-compliance-detector](https://huggingface.co/spaces/Ezeobi-O/ppe-compliance-detector)

---

## Testing

```bash
pip install ruff pytest
ruff check .
pytest -v
```

---

## Technical Stack

| Component | Choice |
|-----------|--------|
| Detection model | YOLOv8s fine-tuned, 25 classes |
| Training | 50 epochs, image size 640, batch 16, T4 GPU |
| UI framework | Gradio 6.4.0 (Blocks) |
| Dataset | Roboflow Construction Site Safety v30 (~2,600 images) |

---

## Compliance Score

| Score | Status | Meaning |
|-------|--------|---------|
| ≥ 90% | Excellent | Standards maintained |
| 70–89% | Needs Attention | Violations present |
| < 70% | Critical | Immediate action required |
| N/A | No PPE detections | Only context classes in frame |

---

## Developer

**Ebube Ezeobi**  
Computer Science @ Kennesaw State University — Concentration: Artificial Intelligence

- 🌐 [Portfolio](https://obieezeobi.vercel.app)
- 💻 [GitHub](https://github.com/EzeobiO)
- 💼 [LinkedIn](https://linkedin.com/in/ezeobio)

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Roboflow](https://roboflow.com/) for the dataset and tooling
- [Gradio](https://gradio.app/) for the UI framework
- [Hugging Face](https://huggingface.co/) for hosting

---

## License

MIT License — see [`LICENSE`](LICENSE).