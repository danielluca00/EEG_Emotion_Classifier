import os
import pandas as pd
from diffusers import StableDiffusionPipeline
import torch

# === Percorsi ===
INFERENCE_FILE = "results/inference_output.csv"
OUTPUT_DIR = "generated_art"

# === Prompt artistici per emozione ===
EMOTION_PROMPTS = {
    "Positive": (
        "abstract digital art, vibrant warm colors, glowing light, "
        "ethereal atmosphere, soft shapes, dreamlike composition, "
        "emotionally uplifting, high detail, 8k"
    ),
    "Neutral": (
        "minimalist geometric composition, balanced tones, soft gray palette, "
        "calm and still, abstract contemporary art, clean lines, high detail, 8k"
    ),
    "Negative": (
        "dark abstract art, cold blue and black tones, chaotic textures, "
        "emotional tension, stormy atmosphere, sharp shapes, dramatic lighting, 8k"
    )
}


def load_emotion():
    """Legge il file inference e calcola l'emozione dominante."""
    if not os.path.exists(INFERENCE_FILE):
        print(f"❌ File di inference non trovato: {INFERENCE_FILE}")
        return None

    df = pd.read_csv(INFERENCE_FILE)

    # Conta la classe più frequente
    dominant_emotion = df["Predicted_Emotion"].value_counts().idxmax()
    print(f"\n🎭 Emozione dominante rilevata: {dominant_emotion}")

    return dominant_emotion


def create_image_from_emotion(emotion):
    """Genera l'immagine usando Stable Diffusion e il prompt associato."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prompt = EMOTION_PROMPTS.get(emotion, "abstract art")

    print(f"\n🎨 Generazione immagine per emozione '{emotion}'...")
    print(f"📝 Prompt: {prompt}")

    # Caricamento modello
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe.to("cuda")

    # Generazione
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    output_path = os.path.join(OUTPUT_DIR, f"emotion_{emotion.lower()}.png")
    image.save(output_path)

    print(f"\n✅ Immagine generata salvata in:\n   {output_path}\n")


def main():
    print("\n=== EEG → Emotion → Art Generator ===")

    emotion = load_emotion()
    if emotion is None:
        return

    create_image_from_emotion(emotion)


if __name__ == "__main__":
    main()
