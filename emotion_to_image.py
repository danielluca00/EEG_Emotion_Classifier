import os
import math
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch

try:
    from diffusers import StableDiffusionPipeline
except Exception as e:
    raise ImportError(
        "diffusers not available. Install it with: pip install diffusers[torch] transformers accelerate safetensors"
    ) from e

# Paths
INFERENCE_FILE = "results/inference_output.csv"
OUTPUT_DIR = "generated_art"
INDIV_DIR = os.path.join(OUTPUT_DIR, "individuals")
BLEND_DIR = os.path.join(OUTPUT_DIR, "blended")
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Generation hyperparameters
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
SEED = None  # None -> random; set int for reproducibility
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, ugly, watermark, text, signature, deformed, poorly drawn, blurry, underexposed, "
    "overexposed, too dark, black image, corrupted"
)

# Prompt per emozione (hai scelto quelli che funzionavano meglio)
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
        "emotional tension, stormy atmosphere, sharp shapes, dramatic lighting, high detail, 8k"
    )
}

# Minimum weight threshold: emotions with weight below this will be ignored to save time
MIN_WEIGHT_THRESHOLD = 0.01


def ensure_dirs():
    os.makedirs(INDIV_DIR, exist_ok=True)
    os.makedirs(BLEND_DIR, exist_ok=True)


def load_inference_file(path=INFERENCE_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Inference file not found: {path}")
    df = pd.read_csv(path)
    if "Predicted_Emotion" not in df.columns:
        raise ValueError("Inference file must contain a 'Predicted_Emotion' column.")
    return df


def compute_weights(df):
    counts = df["Predicted_Emotion"].value_counts(normalize=False)
    # ensure keys for all emotions
    weights = {k: float(counts.get(k, 0)) for k in EMOTION_PROMPTS.keys()}
    total = sum(weights.values()) or 1.0
    normalized = {k: v / total for k, v in weights.items()}
    return normalized


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    # fallback to mps for mac silicon if available
    if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(device):
    # Use fp16 on cuda for speed & memory saving
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading Stable Diffusion model {MODEL_ID} on {device} (dtype={torch_dtype})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        safety_checker=None
    )

    # enable attention slicing to reduce memory usage if needed
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    pipe.to(device)
    return pipe


def generate_image(pipe, prompt, negative_prompt, width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                   steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    out = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    image = out.images[0]
    return image


def pil_to_array(img: Image.Image):
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def array_to_pil(arr: np.ndarray):
    arr_clipped = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_clipped)


def blend_images(weighted_images):
    """
    weighted_images: list of tuples (weight, PIL.Image)
    returns blended PIL.Image
    """
    if not weighted_images:
        raise ValueError("No images to blend.")

    # normalize weights
    weights = np.array([w for w, _ in weighted_images], dtype=np.float32)
    total = weights.sum()
    if total <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / total

    # convert to arrays and weighted sum
    accum = None
    for (w, pil_img) in zip(weights, [img for _, img in weighted_images]):
        arr = pil_to_array(pil_img)
        if accum is None:
            accum = w * arr
        else:
            accum += w * arr

    blended = array_to_pil(accum)
    return blended


def safe_generate_and_save(pipe, emotion, weight, seed_base=None):
    """
    Generate an image for the given emotion and save it.
    Returns (weight, PIL.Image, path)
    """
    prompt = EMOTION_PROMPTS.get(emotion, "abstract art")
    # small prompt variations to increase diversity and reduce black outputs
    prompt_variation = prompt + " --stylize 100"
    negative = NEGATIVE_PROMPT

    # derive seed for reproducibility based on base and emotion
    seed = None
    if seed_base is not None:
        seed = int((hash(emotion) ^ seed_base) & 0xFFFFFFFF)

    try:
        img = generate_image(
            pipe,
            prompt_variation,
            negative,
            steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=seed
        )
    except Exception as e:
        # fallback: try fewer steps or CPU if something goes wrong
        print(f"  Warning: generation for {emotion} failed with error: {e}. Retrying with CPU and fewer steps...")
        cpu_pipe = None
        try:
            cpu_pipe = pipe.to("cpu")
            img = generate_image(
                cpu_pipe,
                prompt_variation,
                negative,
                steps=max(10, NUM_INFERENCE_STEPS // 2),
                guidance_scale=GUIDANCE_SCALE,
                seed=seed
            )
            # move original pipe back to device
            _ = pipe.to(choose_device())
        except Exception as e2:
            raise RuntimeError(f"Generation failed twice for emotion {emotion}: {e2}") from e

    # save individual image
    safe_name = emotion.lower()
    fname = f"{safe_name}_{int(weight*1000)}.png"
    path = os.path.join(INDIV_DIR, fname)
    img.save(path)
    print(f"  Saved image for '{emotion}' (weight={weight:.3f}) -> {path}")
    return (weight, img, path)


def main():
    ensure_dirs()

    print("\n=== EEG → Emotion → Art Generator (SD15) ===")
    try:
        df = load_inference_file(INFERENCE_FILE)
    except Exception as e:
        print(f"Error loading inference file: {e}")
        return

    weights = compute_weights(df)
    print("\nDetected emotion distribution (normalized):")
    for k, v in weights.items():
        print(f"  {k}: {v*100:.2f}%")

    # select emotions with weight above threshold
    selected = {k: v for k, v in weights.items() if v >= MIN_WEIGHT_THRESHOLD}
    if not selected:
        print("No significant emotions found (all below threshold). Exiting.")
        return

    device = choose_device()
    print(f"\nUsing device: {device}")
    pipe = load_pipeline(device)

    # If a single emotion dominates strongly (e.g. >0.75), generate only that one
    dominant_emotion, dominant_weight = max(weights.items(), key=lambda x: x[1])
    seed_base = None if SEED is None else int(SEED)

    weighted_images = []
    if dominant_weight >= 0.75:
        print(f"\nDominant emotion '{dominant_emotion}' >= 75% — generating single focused image.")
        weight, img, path = safe_generate_and_save(pipe, dominant_emotion, 1.0, seed_base=seed_base)
        final_img = img
        final_path = os.path.join(BLEND_DIR, f"final_{dominant_emotion.lower()}.png")
        final_img.save(final_path)
        print(f"\n✅ Final image saved: {final_path}")
        return

    # Otherwise, generate per-emotion images and blend
    print("\nGenerating images for each emotion and blending according to weights...")
    for emotion, w in selected.items():
        # skip negligible weights
        if w < MIN_WEIGHT_THRESHOLD:
            continue
        wt, img, path = safe_generate_and_save(pipe, emotion, w, seed_base=seed_base)
        weighted_images.append((wt, img))

    if not weighted_images:
        print("No images generated. Exiting.")
        return

    # Blend
    print("\nBlending images...")
    blended = blend_images(weighted_images)
    # save blended result
    # construct filename describing weights
    weights_tag = "_".join([f"{k[:3]}{int(v*100)}" for k, v in weights.items()])
    final_fname = f"blended_{weights_tag}.png"
    final_path = os.path.join(BLEND_DIR, final_fname)
    blended.save(final_path)
    print(f"\n✅ Blended image saved: {final_path}")

    # also save a simple JSON metadata file
    meta = {
        "weights": weights,
        "individual_images": [p for (_, _, p) in [(w, i, p) for (w, i, p) in [(t[0], t[1], os.path.join(INDIV_DIR, f"{t[1].size[0]}x{t[1].size[1]}.png")) for t in weighted_images]]],
        "final_image": final_path
    }
    meta_path = os.path.join(BLEND_DIR, f"meta_{int(1000*sum(weights.values()))}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
