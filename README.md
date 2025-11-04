# 🧠 EEG Emotion Classifier

This project implements a **Deep Neural Network (DNN)** model to classify **human emotions** based on **EEG (Electroencephalography) signals**.  
The dataset used contains EEG features (e.g., FFT coefficients) already preprocessed and extracted from raw EEG signals.  
The goal is to recognize emotional states (e.g., *positive*, *neutral*, *negative*) and later transform them into artistic representations such as images or audio.

---

## 🚀 Project Overview

- **Input:** EEG features extracted from multiple electrodes (e.g., FFT values)
- **Output:** Emotion class (Positive / Neutral / Negative)
- **Model Type:** Deep Neural Network (feed-forward)
- **Frameworks:** TensorFlow / Keras / Scikit-learn
- **Use case:** Emotion recognition and artistic representation of EEG signals

---

## 🧩 Model Architecture

The DNN model consists of multiple fully-connected layers with **ReLU activation**, **Batch Normalization**, and **Dropout** for regularization.

| Layer Type | Units | Activation | Dropout |
|-------------|--------|-------------|----------|
| Dense + BN  | 2548  | ReLU | 0.25 |
| Dense + BN  | 3822  | ReLU | 0.27 |
| Dense + BN  | 5096  | ReLU | 0.30 |
| Dense + BN  | 3822  | ReLU | 0.27 |
| Dense + BN  | 2548  | ReLU | 0.25 |
| Output      | 3     | Softmax | — |

---

## ⚙️ Installation

### 1. Clone the Repository

You can clone the repository using Git. Open your terminal and run the following command:

```bash
git clone https://github.com/danielluca00/EEG_Emotion_Classifier.git
```

### 2. Run the Project

Run the main script:

```bash
python main.py
```

