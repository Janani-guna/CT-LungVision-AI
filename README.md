# Cancer Detection from CT Scan Imaging

End-to-end experiments for classifying **lung CT scans** into cancer classes vs normal.
This repo includes traditional ML baselines (Decision Tree / Logistic Regression) and a compact **TinyVGG-style CNN** implemented in notebooks, plus a lightweight app script for local inference.

> Dataset: **Chest CT-Scan images (Kaggle)** — 4 classes *(adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal)*. ([Kaggle][1], [arXiv][2], [Nature][3])

---

## Table of contents

* [Project overview](#project-overview)
* [Repo structure](#repo-structure)
* [Data](#data)
* [Quickstart](#quickstart)
* [Training](#training)
* [Evaluation](#evaluation)
* [Inference / Demo app](#inference--demo-app)
* [Notes & tips](#notes--tips)
* [Roadmap](#roadmap)
* [Citations](#citations)
* [License](#license)

---

## Project overview

This project explores multiple modeling approaches for CT image classification:

* **Classical ML**: Decision Tree & Logistic Regression baselines (useful sanity checks, quick to train).
* **Deep Learning**: **TinyVGG-style** CNN trained from scratch / fine-tuned.
* **Deployment sketch**: a small local app script to try single-image predictions.

These approaches are reflected by the files and notebooks in this repo (see below). The original description and current README mention both **TinyVGG** and **Decision Trees**, which is why both paths are kept here. ([GitHub][4])

---

## Repo structure

```
.
├── TinyVGGModel.ipynb               # TinyVGG training (baseline)
├── TinyVGGModelModified.ipynb       # Variants / tweaks
├── DecisionTreeNoteBook.ipynb       # Classical ML baseline
├── model_deployment_app.py          # Simple local inference app (see below)
├── app.py                           # (alt) demo script
├── model_LogR.sav                   # Saved Logistic Regression model
├── requirement.txt                  # Python dependencies
├── README.md
└── docs/ (dataset dictionaries)
    ├── cxr_abnormalities.dictionary.d040722.pdf
    ├── participant.dictionary.d040722.pdf
    └── sct_image_series.dictionary.d040722.pdf
```

> GitHub reports the repo is mostly Jupyter notebooks with a bit of Python glue. ([GitHub][4])

---

## Data

* **Source**: Kaggle — *Chest CT-Scan images* by Mohamed Hany.
  Link: [https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
  Typical structure: images grouped by **four classes** (Adenocarcinoma, Large Cell, Squamous Cell, Normal). Some mirrors/derivatives provide train/val/test splits out of the box. ([Kaggle][1], [Nature][3])

### Expected local layout

Place (or symlink) the dataset like this (adjust paths in notebooks as needed):

```
data/
  train/
    adenocarcinoma/
    large cell carcinoma/
    squamous cell carcinoma/
    normal/
  valid/
    adenocarcinoma/
    large cell carcinoma/
    squamous cell carcinoma/
    normal/
  test/
    adenocarcinoma/
    large cell carcinoma/
    squamous cell carcinoma/
    normal/
```

If your download only has a single folder of class subdirs, you can create train/valid/test splits with a small script or via `torchvision.datasets.ImageFolder` + `random_split`.

---

## Quickstart

1. **Environment**

```bash
# clone
git clone https://github.com/Janani-guna/CT-LungVision-AI
cd CT-LungVision-AI

# create venv (recommended)
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows (powershell)
# .\.venv\Scripts\Activate.ps1

# install deps
pip install --upgrade pip
pip install -r requirement.txt
```

> The file is named `requirement.txt` (singular) in this repo.

2. **Open the notebooks**

Use Jupyter or VS Code to run:

* `TinyVGGModel.ipynb` or `TinyVGGModelModified.ipynb`
* `DecisionTreeNoteBook.ipynb`

Update dataset paths at the top of each notebook.

---

## Training

### TinyVGG (PyTorch)

The TinyVGG setup follows the well-known VGG pattern of **stacked 3×3 conv + ReLU + max-pool** blocks with a small classifier head. It’s intentionally compact for quick iteration on limited data. (Background on VGG/TinyVGG: see references.) ([PyImageSearch][5], [viso.ai][6], [FreeCodeCamp][7])

Common knobs you can tweak in the notebook:

* Input size (`224×224` is conventional for VGG-style models)
* Data augmentation (flip, rotate, slight zoom, CLAHE if desired)
* Learning rate / scheduler, weight decay, early stopping
* Class weights if classes are imbalanced

### Classical ML baselines

`DecisionTreeNoteBook.ipynb` loads features (either simple pixel/intensity stats or embeddings) and trains a Decision Tree; `model_LogR.sav` holds a trained Logistic Regression model you can reuse for quick comparisons.

---

## Evaluation

Typical metrics to track:

* **Accuracy, Precision, Recall, F1, ROC-AUC**, plus per-class support
* **Confusion matrix** (helps reveal class confusion, e.g. *adenocarcinoma* vs *squamous*)

For medical imaging work, consider patient-level splits (avoid leakage across splits), calibration curves, and threshold selection tuned to your use case.

---

## Inference / Demo app

There are two small scripts included; open the file to confirm which framework it uses in your environment:

* **`model_deployment_app.py`** – a simple local app for single-image prediction (check imports to see if it uses Streamlit or Flask, then run accordingly).

  * Streamlit style: `streamlit run model_deployment_app.py`
  * Flask style: `python model_deployment_app.py` and visit the shown URL.

* **`app.py`** – alternate demo script with similar intent.

Make sure the model weights you want to use (TinyVGG `.pth` or classical `.sav`) are loaded in the script, and the preprocessing matches your training pipeline.

---

## Notes & tips

* **Reproducibility**: fix random seeds and note your exact train/val/test split.
* **Class imbalance**: try class-weighted loss, balanced sampling, or modest augmentation.
* **Generalization**: keep a held-out test set from the start; avoid peeking via repeated tuning.
* **Ethics & scope**: this is a **research/learning** project; **do not** use it for clinical decisions.

---

## Roadmap

* Lift notebooks into Python modules and a CLI (`train.py`, `infer.py`)
* Add **Grad-CAM**/saliency maps for explainability
* Training logs + tensorboard, early-stopping & checkpointing
* Clear export path: `TinyVGG → .pth` and matching loader in the app
* Unit tests for transforms and inference preprocessing

---

## Citations

* **Dataset:** Kaggle *Chest CT-Scan images* (4 classes; common train/val/test layout). ([Kaggle][1], [Nature][3])
* **Repo context:** file list and description indicating TinyVGG and Decision Tree variants. ([GitHub][4])
* **Background on VGG/TinyVGG:** compact VGG-style CNNs with stacked 3×3 conv blocks. ([PyImageSearch][5], [viso.ai][6], [FreeCodeCamp][7])
* **Independent references using the same Kaggle dataset and 4-class setup.** ([arXiv][2])

---

## License

No license file is present in this repo. If you plan to reuse parts of this code, please open an issue to discuss terms.

---

### Project Status

This repository is an adapted version of the publicly available project referenced in citation [4].

[1]: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images?utm_source=chatgpt.com "Chest CT-Scan images Dataset"
[2]: https://arxiv.org/pdf/2304.04814?utm_source=chatgpt.com "Lung Cancer Diagnosis of CT scan Images Using CNN ..."
[3]: https://www.nature.com/articles/s41598-025-97645-5?utm_source=chatgpt.com "Explainable AI for lung cancer detection via a custom CNN ..."
[4]: https://github.com/FreshPrince99/Cancer-detection-through-CT-scan-imaging "Original public project source"
[5]: https://pyimagesearch.com/2021/05/22/minivggnet-going-deeper-with-cnns/?utm_source=chatgpt.com "MiniVGGNet: Going Deeper with CNNs"
[6]: https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/?utm_source=chatgpt.com "Very Deep Convolutional Networks (VGG) Essential Guide"
[7]: https://www.freecodecamp.org/news/implement-vgg-from-scratch-with-pytorch-deep-learning-theory/?utm_source=chatgpt.com "Implement VGG From Scratch with PyTorch"
