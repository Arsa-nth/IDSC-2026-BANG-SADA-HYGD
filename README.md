# HYGD Advanced Glaucoma Detection - IDSC 2026

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Competition](https://img.shields.io/badge/IDSC-2026-gold)

This repository contains a high-performance Deep Learning implementation for early Glaucoma detection using the **HYGD (Hillel-Yaffe Glaucoma Dataset)** from PhysioNet. Developed by team **BANG SADA** for the **IDSC 2026** competition.

## 📖 Quick View (Recommended)
If GitHub has trouble rendering the notebook due to its size, you can view the full version with all outputs (graphs, Grad-CAM, and results) here:
👉 **[View Notebook on NBViewer](https://nbviewer.org/github/Arsa-nth/IDSC-2026-BANG-SADA-HYGD/blob/main/IDSC%202026_BANG%20SADA_HYGD.ipynb)**

## 👥 Team Members (BANG SADA)
1. **Adinda Sekaring Wana**
2. **Bonfilio Renato Lawaziduhu Fau**
3. **Keenan Gadi Palwono**
4. **Nathanael Komang Bagus Prakarsa**

## 🚀 Technical Architecture
The system utilizes a state-of-the-art ensemble approach tailored for medical imaging:
* **Backbone Ensemble:** A hybrid fusion of **EfficientNet-B4** and **ConvNeXt-Base**.
* **Loss Function:** Implementation of **Quality-Weighted Focal Loss** to handle class imbalance and prioritize high-confidence medical labels.
* **Optimization:** Multi-Scale **Test Time Augmentation (TTA)** featuring 135 augmentations per architecture to ensure robust inference.
* **Validation Strategy:** 5-Fold Cross-Validation with Out-of-Fold (OOF) Ensemble to minimize overfitting.

## 📊 Model Performance
Based on the final notebook execution:
* **Ensemble OOF AUC:** `0.9888`
* **Youden Threshold:** `0.595` (Optimized for overall accuracy)
* **Clinical Threshold:** `0.424` (Optimized for high sensitivity in clinical screening)
* **Primary Metric:** GON+ Probability (Glaucomatous Optic Neuropathy)

## 🛠️ Project Structure
The notebook is organized into 13 comprehensive sections:
1. **Configuration & Imports:** Environment setup (PyTorch, Timm, Albumentations).
2. **Dataset Load & Sanity Check:** Ensuring data integrity from PhysioNet.
3. **EDA (Exploratory Data Analysis):** Dataset-level and single-image level analysis (EDA 1-20).
4. **Preprocessing Pipeline:** Medical-grade augmentation and normalization.
5. **Model Components:** Architecture definition and custom loss functions.
6. **Training Stage:** From ResNet-50 baseline to advanced transformer-based models.
7. **Inference & Explainability:** Visualizing decision-making through **Grad-CAM** (heatmaps showing retinal areas used for diagnosis).

## 🖥️ Getting Started
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install torch torchvision timm albumentations opencv-python pandas matplotlib