# 🩺 Breast Cancer Detection Using Thermography (DMR-IR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=flat-square&logo=kaggle)

A deep learning project for classifying breast thermography images into **Sick**, **Normal**, and **Unknown** categories using a custom CNN with a self-attention mechanism and Grad-CAM visualization.

![Description of image](image.png)


---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## 🔍 Overview

Breast cancer is one of the leading causes of cancer-related mortality worldwide. This project explores the use of **infrared thermography (DMR-IR)** as a non-invasive screening tool, combined with convolutional neural networks to automate the classification of thermal breast images.

A lightweight custom CNN with an integrated **Simple Attention mechanism** is trained on the BCD Dataset to distinguish between cancerous, normal, and ambiguous (unknown) thermal patterns.

---

## 📂 Dataset

**Source:** [Breast Cancer Detection Using Thermography – Kaggle](https://www.kaggle.com/datasets/)

| Class          | Count |
|----------------|-------|
| Normal         | 162   |
| Sick           | 100   |
| Unknown Class  | 100   |
| **Total**      | **362** |

- No duplicate or null entries were found in the dataset.
- Class imbalance was addressed using **oversampling** of minority classes, yielding a balanced dataset of **324 samples**.
- Data split: **80% train / 10% validation / 10% test** with stratification.

---

## 🔄 Project Pipeline

```
Raw Images
    │
    ▼
Data Loading & EDA (Pandas, Seaborn)
    │
    ▼
Label Encoding & Class Balancing (Oversampling)
    │
    ▼
Image Preprocessing (ImageDataGenerator, 224×224 RGB)
    │
    ▼
Custom CNN + Attention Layer
    │
    ▼
Training (20 epochs, Adam, Sparse Categorical Cross-Entropy)
    │
    ▼
Evaluation (Accuracy, F1-Score, Confusion Matrix, Grad-CAM)
```

---

## 🧠 Model Architecture

A custom **Functional API CNN** was built with the following structure:

| Layer                        | Output Shape       | Parameters |
|------------------------------|--------------------|------------|
| Input                        | (224, 224, 3)      | —          |
| Conv2D (32 filters, 3×3)     | (224, 224, 32)     | 896        |
| MaxPooling2D                 | (112, 112, 32)     | —          |
| Conv2D (64 filters, 3×3)     | (112, 112, 64)     | 18,496     |
| MaxPooling2D                 | (56, 56, 64)       | —          |
| Conv2D (64 filters, 3×3)     | (56, 56, 64)       | 36,928     |
| MaxPooling2D                 | (28, 28, 64)       | —          |
| Conv2D (64 filters, 3×3)     | (28, 28, 64)       | 36,928     |
| MaxPooling2D                 | (14, 14, 64)       | —          |
| **SimpleAttention**          | (14, 14, 64)       | 64         |
| Concatenate (x + attention)  | (14, 14, 128)      | —          |
| GlobalAveragePooling2D       | (128,)             | —          |
| Dense (Softmax, 3 classes)   | (3,)               | 387        |

**Total Parameters:** 93,635 (~365 KB)

The `SimpleAttention` layer applies channel-wise feature weighting using a learned Dense layer, then concatenates the weighted output with the original feature map — enhancing the model's focus on thermally abnormal regions.

---

## 📊 Results

### Training Summary (20 Epochs)

| Metric              | Value  |
|---------------------|--------|
| Final Train Accuracy| ~83%   |
| Final Val Accuracy  | ~81%   |
| Final Train Loss    | ~0.42  |
| Final Val Loss      | ~0.47  |

### Test Set Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0 (Sick)      | 1.00      | 0.56   | 0.71     | 9       |
| 1 (Unknown)   | 1.00      | 1.00   | 1.00     | 7       |
| 2 (Normal)    | 0.81      | 1.00   | 0.89     | 17      |
| **Accuracy**  |           |        | **0.88** | **33**  |
| Macro Avg     | 0.94      | 0.85   | 0.87     | 33      |
| Weighted Avg  | 0.90      | 0.88   | 0.87     | 33      |

### Confusion Matrix

The model perfectly classified the **Unknown** and **Normal** classes, with minor misclassifications in the **Sick** class (4 out of 9 misclassified as Normal).

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- GPU recommended (project was run on 2× Tesla T4 via Kaggle)

### Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-thermography.git
cd breast-cancer-thermography
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

```
tensorflow>=2.10
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
```

---

## 🚀 Usage

1. Download the dataset from Kaggle and place it at:
   ```
   /kaggle/input/breast-cancer-detection-usingthermography/BCD_Dataset/
   ```

2. Run the notebook or script end-to-end:
   ```bash
   jupyter notebook breast_cancer_thermography.ipynb
   ```

3. The script will:
   - Load and explore the dataset
   - Balance classes via oversampling
   - Train the CNN model for 20 epochs
   - Output accuracy/loss curves, classification report, and confusion matrix

---

## 🛠️ Technologies Used

| Tool/Library      | Purpose                              |
|-------------------|--------------------------------------|
| TensorFlow / Keras| Model building and training          |
| OpenCV (cv2)      | Image loading and color conversion   |
| Pandas            | Data handling and preprocessing      |
| NumPy             | Numerical operations                 |
| Scikit-learn      | Train/test split, evaluation metrics |
| Seaborn / Matplotlib | Data visualization               |
| Kaggle (Tesla T4 × 2) | GPU-accelerated training        |

---

## 📁 Project Structure

```
breast-cancer-thermography/
│
├── BCD_Dataset/
│   ├── Sick/
│   ├── normal/
│   └── Unknown_class/
│
├── breast_cancer_thermography.ipynb   # Main notebook
├── README.md
└── requirements.txt
```

---

## 🔮 Future Work

- Integrate **transfer learning** (e.g., EfficientNet, ResNet50) for improved feature extraction
- Explore **multi-scale attention** or **CBAM** (Convolutional Block Attention Module)
- Apply **data augmentation** (rotation, flipping, zoom) to improve generalization
- Deploy model as a lightweight web app using **Streamlit** or **Gradio**
- Expand dataset with more clinical thermography samples for better real-world reliability

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset sourced from the [Kaggle BCD Dataset](https://www.kaggle.com/)
- Inspired by research into non-invasive breast cancer screening using infrared imaging
- Built and trained on the Kaggle GPU environment

---

> *"Early detection saves lives."* — This project is intended for research and educational purposes only, and is not a substitute for professional medical diagnosis.
