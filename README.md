# 🧠 From Brain Signals to Visual Art  
## EEG Emotion-to-Image Framework using Evolutionary-Optimized DNNs and Diffusion Models

This repository implements an **end-to-end EEG-based emotion recognition and visualization framework** that classifies emotional states from brain signals and transforms them into artistic images using **deep learning**, **evolutionary optimization**, and **diffusion models**.

---

## 🔍 Overview

EEG signals are highly non-linear, non-stationary, noisy, and subject-dependent, making emotion recognition a challenging task.  
This project builds upon a **pre-extracted EEG feature dataset** and introduces an optimized classification and visualization pipeline by combining:

- **Evolutionary feature selection (GA)**
- **Hyperparameter optimization (PSO)**
- **Deep Neural Networks (DNNs)**
- **Emotion-to-image generation via Stable Diffusion**

The final system bridges **affective computing** and **generative AI**, enabling intuitive visualization of EEG-derived emotions.

---

## 📊 Dataset

- **Source:** Kaggle  
  https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions

- **Description:**  
  The dataset contains **pre-extracted EEG features** computed from raw EEG recordings acquired with a Muse headband (TP9, AF7, AF8, TP10).

- **Note:**  
  Raw EEG acquisition, preprocessing, windowing, and feature extraction were performed by the dataset authors.  
  This project focuses on **feature selection, classification, optimization, inference, and visualization**.

---

## 🎯 Emotion Classes

The classifier predicts three valence-based emotional states:

- **Positive**
- **Neutral**
- **Negative**

Predicted class probabilities are also used to generate **blended emotional images**.

---

## 🧩 Features and Optimization

- **Initial feature space:** 2549 handcrafted EEG features  
- **Feature selection:** Genetic Algorithm (GA)  
- **Classifier:** Deep Neural Network (DNN)  
- **Hyperparameter optimization:** Particle Swarm Optimization (PSO)  

The combined GA + PSO approach improves both **classification accuracy** and **computational efficiency**.

---

## 🎨 Emotion-to-Image Generation

Emotion predictions are mapped to emotion-conditioned text prompts and passed to **Stable Diffusion** to generate images representing:

- Dominant emotional states
- Mixed emotional distributions via proportional blending

This provides an expressive alternative to traditional numerical EEG analysis.

---

## 📊 Results (Summary)

| Configuration | Accuracy |
|--------------|----------|
| Baseline     | 91.2%    |
| GA Only      | 95.6%    |
| PSO Only     | 96.8%    |
| **GA + PSO** | **98.43%** |

---

## 🛠️ Technologies

- Python  
- NumPy / SciPy  
- Scikit-learn  
- TensorFlow / Keras  
- Genetic Algorithms (GA)  
- Particle Swarm Optimization (PSO)  
- Stable Diffusion  

---

## 🚀 How to Run

### 1️⃣ Clone the repository and install dependencies
```bash
git clone https://github.com/danielluca00/EEG_Emotion_Classifier.git
cd EEG_Emotion_Classifier
pip install -r requirements.txt
```
### 2️⃣ Train and evaluate the model
```bash
python main.py
```
### 3️⃣ Run inference on custom EEG feature data
```bash
python inference.py
```
### 4️⃣ Generate emotion-driven images from inference results
```bash
python emotion_to_image.py
```
