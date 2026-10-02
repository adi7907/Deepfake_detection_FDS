# Deepfake Image Detection (Assignment 2)

**Course:** Foundations of Data Science (FDS)  
**Topic:** Deep Learning approach for Deepfake Image Detection  
**Reference Paper:** *MesoNet: a Compact Facial Video Forgery Detection Network*, Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen (IEEE WIFS 2018)

## 📌 Project Overview
This repository contains an end-to-end Data Science and Machine Learning pipeline for detecting AI-generated images (deepfakes). The methodology implemented follows the **MesoNet** architecture, a lightweight mesoscopic Convolutional Neural Network (CNN) specifically designed to detect digital forgeries by extracting micro-texture artifacts rather than relying on high-level semantic features.

## 🚀 Methodology
1. **Data Preprocessing:** Images are resized to 128x128 pixels, normalized, and converted to RGB.
2. **Architecture:** The `Meso4` architecture is constructed using four consecutive 2D convolutional blocks (with Batch Normalization and Max Pooling), followed by a dense classification layer with Dropout.
3. **Training & Validation:** The model is evaluated using 3-Fold Stratified Cross-Validation to ensure generalization and robustness against overfitting.
4. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix are generated to critically assess the model's forensic capabilities.

## 📁 Repository Structure
* `Deepfake_Detection_MesoNet.ipynb`: The primary Jupyter Notebook containing the full execution pipeline (Dataset Loading, Model Building, Cross-Validation, and Visualization).
* `Assignment2_Report.md` / `.pdf`: The detailed assignment report documenting the research, architecture, and empirical findings.

## ⚙️ How to Run
This notebook is fully compatible with **Google Colab**.
1. Upload `Deepfake_Detection_MesoNet.ipynb` to Google Colab.
2. Mount your Google Drive and ensure your dataset is placed in folders named `real` and `fake`.
3. Update the `base_path` variable in the notebook to point to your dataset.
4. Run all cells to train the MesoNet model and generate the final accuracy reports and visualizations.
