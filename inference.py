import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from utils.feature_selection import load_selected_features

# === Percorsi predefiniti ===
DEFAULT_MODEL_PATH = "results/best_dnn_model.h5"
DEFAULT_FEATURES_PATH = "selected_features/latest_features.json"
OUTPUT_PATH = "results/inference_output.csv"

# === Mappa etichette ===
emotion_map = {0: "Negative", 1: "Neutral", 2: "Positive"}


def preprocess_data(df, selected_indices=None):
    """
    Prepara i dati EEG per l'inference:
    - Rimuove colonne non numeriche
    - Applica lo StandardScaler
    - Seleziona solo le feature ottimizzate, se specificate
    """
    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if selected_indices is not None:
        X_scaled = X_scaled[:, selected_indices]

    return X_scaled, X.columns.tolist()


def run_inference():
    print("\n=== EEG Emotion Classification (Inference) ===")

    # === Carica modello preaddestrato ===
    model_path = input(f"\n📂 Inserisci percorso modello pre-addestrato [{DEFAULT_MODEL_PATH}]: ").strip()
    if model_path == "":
        model_path = DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        print(f"❌ Errore: modello non trovato in {model_path}")
        return

    print(f"🔹 Caricamento modello da: {model_path}")
    model = load_model(model_path)

    # === Caricamento feature set selezionato ===
    selected_indices = None
    use_features = input("\nUsare set di feature selezionate dal GA? (y/n): ").strip().lower()

    if use_features == "y":
        features_path = input(f"📁 Inserisci percorso file feature [{DEFAULT_FEATURES_PATH}]: ").strip()
        if features_path == "":
            features_path = DEFAULT_FEATURES_PATH

        if os.path.exists(features_path):
            selected_indices = load_selected_features(features_path)
            print(f"✅ Caricate {len(selected_indices)} feature selezionate dal GA.")
        else:
            print("⚠️ File di feature non trovato, verranno usate tutte le feature disponibili.")

    # === Carica nuovo file EEG ===
    eeg_path = input("\n🧠 Inserisci percorso file CSV EEG da classificare: ").strip()
    if not os.path.exists(eeg_path):
        print("❌ Errore: file EEG non trovato.")
        return

    df = pd.read_csv(eeg_path)
    X_new, feature_names = preprocess_data(df, selected_indices)

    # === Predizione ===
    print("\n🚀 Classificazione in corso...")
    pred_probs = model.predict(X_new)
    pred_classes = np.argmax(pred_probs, axis=1)
    pred_labels = [emotion_map[c] for c in pred_classes]

    df["Predicted_Emotion"] = pred_labels
    df["Confidence"] = np.max(pred_probs, axis=1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Classificazione completata! Risultati salvati in:\n   {OUTPUT_PATH}\n")
    print("📊 Distribuzione emozioni predette:")
    print(df["Predicted_Emotion"].value_counts())

    print("\nEsempio risultati:")
    print(df[["Predicted_Emotion", "Confidence"]].head())


if __name__ == "__main__":
    run_inference()
