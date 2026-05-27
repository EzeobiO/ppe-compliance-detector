# Model Card — PPE Compliance Detector

## Model Details

| Field | Value |
|-------|-------|
| **Base model** | YOLOv8s (pretrained on COCO) |
| **Fine-tuned weights** | `ppe_detector_best.pt` |
| **Framework** | Ultralytics YOLOv8 |
| **Task** | Object detection |
| **Input** | RGB images, resized to 640 × 640 |

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Early stopping patience | 10 |
| Augmentations | Default Ultralytics (mosaic, HSV, scale, translate, fliplr) |
| Hardware | Google Colab T4 GPU |
| Training time | ~45–60 min |

---

## Dataset

**Roboflow Construction Site Safety** (version 30)  
Source: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

- ~2,600 labeled construction site images
- Train / val / test split provided by Roboflow
- 25 classes (see below)

---

## Classes

| ID | Class | Role |
|----|-------|------|
| 0 | Excavator | Context |
| 1 | Gloves | ✅ Safe / PPE |
| 2 | Hardhat | ✅ Safe / PPE |
| 3 | Ladder | Context |
| 4 | Mask | ✅ Safe / PPE |
| 5 | NO-Hardhat | 🚨 Violation |
| 6 | NO-Mask | 🚨 Violation |
| 7 | NO-Safety Vest | 🚨 Violation |
| 8 | Person | Context |
| 9 | SUV | Context |
| 10 | Safety Cone | Context |
| 11 | Safety Vest | ✅ Safe / PPE |
| 12 | bus | Context |
| 13 | dump truck | Context |
| 14 | fire hydrant | Context |
| 15 | machinery | Context |
| 16 | mini-van | Context |
| 17 | sedan | Context |
| 18 | semi | Context |
| 19 | trailer | Context |
| 20 | truck and trailer | Context |
| 21 | truck | Context |
| 22 | van | Context |
| 23 | vehicle | Context |
| 24 | wheel loader | Context |

**Compliance scoring:** detections in the Safe/PPE and Violation roles contribute to the compliance rate; Context detections do not.

---

## Performance

Evaluated on the Roboflow-provided test split.

| Metric | Score |
|--------|-------|
| mAP50 | ~0.85 |
| mAP50-95 | ~0.65 |
| Inference (T4 GPU) | ~25 ms |

Training curves and the confusion matrix are saved as `Training Results.png` and `Confusion Matrix.png` in the repository root.

---

## Intended Use

- Construction site safety audits
- Safety training demonstrations and awareness programs
- Research into automated PPE compliance monitoring

---

## Out-of-Scope Use

- **Legally binding compliance enforcement** — model output is probabilistic and should not replace human safety inspection
- **Individual identification** — the model detects PPE classes, not persons; do not use it for surveillance
- **Adverse conditions** — not evaluated on night-time imagery, heavy occlusion, or extreme weather

---

## Known Limitations

The following limitations are based on general properties of the training setup. Test your deployment against site-specific conditions and fill in the placeholders below from direct observation.

- **[FILL IN BASED ON YOUR TESTING]** Color or texture biases (e.g., reflective vests vs. standard orange)
- **[FILL IN BASED ON YOUR TESTING]** Minimum detection distance / resolution requirements
- **[FILL IN BASED ON YOUR TESTING]** Behavior on heavily occluded workers or crowded scenes
- Confidence threshold must be tuned per scene; the default of 0.35 may produce false positives in complex backgrounds

---

## Developer

**Ebube Ezeobi** — Computer Science (AI concentration), Kennesaw State University  
[Portfolio](https://obie-ezeobi.vercel.app) · [GitHub](https://github.com/EzeobiO) · [LinkedIn](https://linkedin.com/in/ezeobio)
