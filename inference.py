import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from utils.feature_selection import load_selected_features
import joblib

# === Paths ===
DEFAULT_MODEL_PATH = "results/best_dnn_model.h5"
DEFAULT_FEATURES_PATH = "selected_features/latest_features.json"
OUTPUT_PATH = "results/inference_output.csv"

# === Label map ===
emotion_map = {0: "Negative", 1: "Neutral", 2: "Positive"}


def preprocess_data(df, selected_indices=None, scaler_path="models/scaler.pkl"):
    """
    Preparing EEG data for inference using the TRAINING SCALER
    """
    X = df.select_dtypes(include=[np.number])

    # ⬅️ Load the scaler used during training
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Training scaler doesn't exist: {scaler_path}")

    scaler = joblib.load(scaler_path)

    # ⬅️ Apply TRAINING SCALER
    X_scaled = scaler.transform(X)

    if selected_indices is not None:
        X_scaled = X_scaled[:, selected_indices]

    return X_scaled, X.columns.tolist()


def run_inference():
    print("\n=== EEG Emotion Classification (Inference) ===")

    # === Load pre-trained model ===
    model_path = input(f"\n📂 Insert pre-trained model path [{DEFAULT_MODEL_PATH}]: ").strip()
    if model_path == "":
        model_path = DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        print(f"❌ Error: model not found in {model_path}")
        return

    print(f"🔹 Loading model from: {model_path}")
    model = load_model(model_path)

    # === Load selected feature set ===
    selected_indices = None
    use_features = input("\nUse selected feature set from GA? (y/n): ").strip().lower()

    if use_features == "y":
        features_path = input(f"📁 Insert feature set path [{DEFAULT_FEATURES_PATH}]: ").strip()
        if features_path == "":
            features_path = DEFAULT_FEATURES_PATH

        if os.path.exists(features_path):
            selected_indices = load_selected_features(features_path)
            print(f"✅ Loaded {len(selected_indices)} features selected from GA.")
        else:
            print("⚠️ Features file not found, all availables features will be used.")

    # === Load new EEG file ===
    eeg_path = input("\n🧠 Insert EEG CSV file path to classify: ").strip()
    if not os.path.exists(eeg_path):
        print("❌ Error: EEG file not found.")
        return

    df = pd.read_csv(eeg_path)
    X_new, feature_names = preprocess_data(df, selected_indices)

    # === Prediction ===
    print("\n🚀 Classifying EGG signals...")
    pred_probs = model.predict(X_new)
    pred_classes = np.argmax(pred_probs, axis=1)
    pred_labels = [emotion_map[c] for c in pred_classes]

    # Predicted label
    df["Predicted_Emotion"] = pred_labels

    # Confidence of the class with highest probability
    df["Confidence"] = np.max(pred_probs, axis=1)

    # === Probability of each class ===
    for class_idx, class_name in emotion_map.items():
        df[f"Prob_{class_name}"] = pred_probs[:, class_idx]

    # === Save results ===
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Classification completed! Results are saved in:\n   {OUTPUT_PATH}\n")

    print("📊 Distribution of predicted emotions:")
    print(df["Predicted_Emotion"].value_counts())

    print("\nResults example (first 5 rows):")
    print(df[[
        "Predicted_Emotion",
        "Confidence",
        "Prob_Negative",
        "Prob_Neutral",
        "Prob_Positive"
    ]].head())


if __name__ == "__main__":
    run_inference()
