# What I Learned Building the PPE Compliance Detector

## The Starting Point

I wanted a project that sat at the intersection of computer vision and a real-world problem I could explain quickly to anyone. Workplace safety felt like the right target: it is measurable, it is visual, and it has an obvious cost when it fails. The idea was simple -- take site photos or a live camera feed, run detection, and tell the user whether workers are wearing the right equipment.

Before writing any code I spent time narrowing the scope. PPE detection is a mature research problem, and the goal was not to beat state-of-the-art but to build something end-to-end that a recruiter or a site manager could actually open in a browser.

## Choosing the Dataset

I settled on the Roboflow Construction Site Safety dataset version 30 because it already had 25 annotated classes covering the full safety picture: Hardhat, NO-Hardhat, Mask, NO-Mask, Safety Vest, NO-Safety Vest, along with context classes like Person, machinery, and various vehicles. Having both the positive class (Hardhat) and the violation class (NO-Hardhat) in the same dataset meant I could directly compute a compliance ratio without post-processing tricks.

The dataset is roughly 2,600 labeled images split into train, validation, and test by Roboflow. I pulled it using the Roboflow Python SDK inside a Colab notebook so I did not have to manage downloads manually.

## Model Selection and the Speed vs. Accuracy Trade-off

I chose YOLOv8s, the small variant of Ultralytics YOLOv8. My reasoning was:

- It is fast enough to run inference in under 20 ms per image on a T4 GPU, which matters for the live webcam feature.
- Its pretrained COCO weights already recognize vehicles, people, and equipment-like objects, so fine-tuning converges quickly on a construction site dataset.
- The Ultralytics API is straightforward, which meant less time debugging the training loop and more time building the app.

YOLOv8m or YOLOv8l would have produced higher mAP but the jump in inference latency was not worth it for a demo that needed to feel real-time.

## Training on Google Colab

I ran training on a Colab T4 GPU because it is free and immediately accessible. The configuration was 100 maximum epochs with `patience=10` for early stopping. In practice, training halted at epoch 55 because validation mAP50 had not improved in 10 consecutive epochs. The best checkpoint, saved automatically as `best.pt`, came from epoch 45.

Final validation metrics on the 114-image validation split:

- mAP50: 0.614
- mAP50-95: 0.423
- Precision: 0.802
- Recall: 0.492

Some classes like Gloves and Safety Cone are underrepresented in the dataset and drag the aggregate scores down. The safety-critical violation classes (NO-Hardhat, NO-Mask, NO-Safety Vest) all achieved mAP50 values between 0.55 and 0.67, which is usable for a demo context.

One thing I miscalculated early on was thinking I needed more epochs. Setting patience=10 and watching the curves showed me that the model was plateauing by epoch 45 regardless of how many more epochs I let it run. Seeing that inflection point on the loss curve was a useful intuition builder.

## Exporting and Storing Weights

After training, I exported `best.pt` from Colab with `files.download` and committed it directly to the repository tracked by Git LFS. The file is around 22 MB, which is small enough to live in the repo without causing friction for anyone who clones it.

## Designing the App Architecture

My first draft was a single script with everything mixed together. That lasted about 30 minutes before I realized the inference code, the Gradio layout, and the report generation had nothing to do with each other and should not share a namespace.

I refactored into a five-module layout:

- `config.py` holds every magic number and class-ID set in one place. Any time I needed to know whether a class ID was a violation or a context class, the answer was one import away.
- `model.py` handles loading the weights and surfacing a clear error if the file is missing.
- `detection.py` owns the `Detection` dataclass, `classify_by_id`, the annotation drawing logic, and `summarize_detections`. Keeping detection as a pure data transformation -- inputs in, labeled boxes out -- made it trivially testable.
- `reports.py` turns a list of `Detection` objects into a Markdown string and a plain-text file. Separating rendering from computation meant I could write a test for the Markdown output without needing a model or an image.
- `ui.py` contains the Gradio Blocks definition. The single function `build_demo` returns a fully wired interface and takes `model`, `model_type`, and `class_names` as arguments. No globals.
- `app.py` is the entry point: ten lines that load the model, call `build_demo`, attach the theme, and launch.

This structure turned out to be the most educational part of the project. Writing tests for `detect` and `reports` forced me to make the interfaces clean. If I could not construct a fake input without loading a real model, the function was doing too much.

## Building the Gradio Interface

I chose Gradio because it handles webcam streaming, image upload, and file downloads with very little boilerplate, and because Hugging Face Spaces serves Gradio apps natively.

Features I added progressively:

- Image upload tab with annotated output, compliance score, and Markdown report
- Webcam tab with live annotation and a session-averaged FPS counter
- Confidence slider so users can tune the detection threshold
- Download button that writes the compliance report to a temp file and serves it as a `.txt`
- Scan history that logs the last 10 detected violation events to a session-scoped list

The hardest part of the UI work was the webcam loop. Gradio streams frames by calling the generator function on each tick. I had to be careful about where the FPS deque lived (inside the function closure, seeded on each tab switch) to avoid state leaking between sessions.

I also spent time picking example images that actually show interesting detections: a crowded site with a mix of compliant workers and violations, a close-up helmet shot, and a vehicle-only frame to demonstrate the context class behavior.

## CI/CD and Deployment

I set up two GitHub Actions workflows:

- `ci.yml` runs Ruff lint and pytest on every push and pull request.
- `deploy.yml` calls `ci.yml` as a reusable workflow and, if it passes, syncs the repo to a Hugging Face Space using `huggingface-hub` CLI.

This meant every push to main that passed tests was automatically live at the HF Space URL. The main thing I had to figure out was setting `HF_TOKEN` as a GitHub Actions secret and structuring the deploy step to only push after CI passed, not in parallel with it.

I also had to handle a Gradio API change where the `theme` parameter moved from `launch()` to the `Blocks` constructor between versions. The fix was passing the theme into `demo.launch()` rather than the `gr.Blocks()` call, which resolved the deprecation warning that was causing HF to reject the build.

## What the Numbers Actually Mean

The gap between the inflated numbers I initially put in the documentation (mAP50 ~0.85) and the real numbers (0.614) taught me something important: I had been reading mAP50 from the wrong place. Training logs print per-batch validation scores that are computed differently from a full post-training `model.val()` pass. The model.val() call is the number that matters for documentation.

0.614 mAP50 across 25 classes on a moderately small dataset is reasonable. It is not a production safety system. It is a project that demonstrates I can train a detection model, evaluate it honestly, build a full-stack app around it, and deploy it with CI/CD. That is what it is meant to show.

## What I Would Do Differently

If I rebuilt this project, I would:

- Track a proper test set evaluation (model.val(split='test')) separately from validation, so the reported metrics reflect held-out performance.
- Run a few ablations: YOLOv8n versus YOLOv8s, different image sizes, and augmentation strengths.
- Use a larger dataset if available. The 25-class split across ~2,600 images means some classes (like fire hydrant and semi) may have only a handful of instances in the training set.
- Set up proper experiment tracking with a tool like Weights and Biases rather than relying on the Ultralytics CSV logs.

## Summary

The project covered the full lifecycle of a computer vision application: problem scoping, dataset selection, model training and evaluation, modular application code, testing, CI/CD, and deployment. The most durable skills I came away with were understanding YOLO's training loop (what patience actually does, why best.pt diverges from final.pt), designing a module structure that is testable from the start, and building Gradio apps that work correctly in a streaming context.
