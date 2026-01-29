# PPE Compliance Detector

Real-time construction site safety monitoring powered by a custom-trained YOLOv8 model. Detects Personal Protective Equipment (PPE) compliance and flags safety violations instantly.

**Developed by [Ebube Ezeobi](https://obie-ezeobi.vercel.app)**

![YOLOv8](https://img.shields.io/badge/YOLOv8-Custom%20Trained-blue)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🎯 What It Detects

| Detection | Status | Visual |
|-----------|--------|--------|
| Hard Hat | ✅ Compliant | 🟢 Green box |
| Safety Vest | ✅ Compliant | 🟢 Green box |
| NO Hard Hat | 🚨 Violation | 🔴 Red box |
| NO Safety Vest | 🚨 Violation | 🔴 Red box |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 **Image Upload** | Analyze site photos for safety compliance |
| 🎥 **Live Webcam** | Real-time monitoring with FPS counter |
| 📊 **Compliance Scoring** | Instant percentage-based safety rating |
| 📋 **Detailed Reports** | Breakdown of all detections and violations |
| 💾 **Export Reports** | Download compliance reports as text files |
| 📜 **Scan History** | Track recent scans and results |
| 🔊 **Sound Alerts** | Audio notification on violation detection |
| 🎚️ **Adjustable Threshold** | Fine-tune detection sensitivity |

---

## 🚀 Quick Start

### Try the Demo
1. Upload a construction site image (or enable webcam)
2. View detected PPE with color-coded bounding boxes
3. Get instant compliance score and detailed report
4. Download report for documentation

### Run Locally
```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/ppe-compliance-detector
cd ppe-compliance-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at `http://localhost:7860`

---

## 📈 Model Performance

Trained on the [Construction Site Safety Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) from Roboflow.

| Metric | Score |
|--------|-------|
| mAP50 | ~0.85 |
| mAP50-95 | ~0.65 |
| Inference | ~25ms (GPU) |

---

## 🛠️ Technical Stack

- **Model:** YOLOv8s (fine-tuned)
- **Training:** 50 epochs on T4 GPU
- **Framework:** Ultralytics + PyTorch
- **Interface:** Gradio
- **Dataset:** 2,600+ labeled construction site images

---

## 💼 Use Cases

| Industry | Application |
|----------|-------------|
| 🏗️ Construction | Site compliance audits |
| 🏭 Manufacturing | Safety monitoring |
| 📹 Security | Real-time violation alerts |
| 🎓 Training | Safety awareness demos |
| 📊 Documentation | Compliance record keeping |

---

## 📁 Project Structure

```
ppe-detection/
├── app.py                    # Gradio application
├── requirements.txt          # Python dependencies
├── ppe_detector_best.pt      # Trained model weights
├── train_ppe_detector.ipynb  # Training notebook (Colab)
└── README.md                 # Documentation
```

---

## 🏋️ Train Your Own Model

1. Open `train_ppe_detector.ipynb` in Google Colab
2. Enable T4 GPU runtime
3. Add your Roboflow API key
4. Run all cells (~45-60 min training)
5. Download `ppe_detector_best.pt`
6. Upload to this Space

---

## 📊 Compliance Score Interpretation

| Score | Status | Action Required |
|-------|--------|-----------------|
| 90-100% | ✅ Excellent | Maintain standards |
| 70-89% | ⚠️ Warning | Address violations |
| <70% | 🚨 Critical | Immediate action needed |

---

## 👨‍💻 Developer

**Ebube Ezeobi**  
Computer Science @ Kennesaw State University  
Concentration: Artificial Intelligence

- 🌐 [Portfolio](https://obie-ezeobi.vercel.app)
- 💻 [GitHub](https://github.com/EzeobiO)
- 💼 [LinkedIn](https://linkedin.com/in/ezeobio)
- 📧 ezeobiebube9@gmail.com

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Roboflow](https://roboflow.com/) for the dataset and tools
- [Gradio](https://gradio.app/) for the UI framework
- [Hugging Face](https://huggingface.co/) for hosting

---

## 📝 License

MIT License — free for personal and commercial use.

---

<p align="center">
  <strong>Built for safety. Powered by AI.</strong> 🦺
</p>
