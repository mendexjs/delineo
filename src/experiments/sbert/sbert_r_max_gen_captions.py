import os
import json
import time
from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai import errors
from joblib import Parallel, delayed

# Using oauth2 config see https://ai.google.dev/palm_docs/oauth_quickstart
# Read default config from /home/your-user/.config/gcloud/application_default_credentials.json


PROJECT_ID = "hybrid-cabinet-482822-p5"
LOCATION = "global"
MODEL_ID = "gemini-2.0-flash"

IMAGES_DIR = "../validation_samples"
OUTPUT_JSONL = "sbert_r_max_samples.jsonl"

NUM_SAMPLES_PER_IMAGE = 20
MAX_RETRIES = 15
N_JOBS = 4

SYSTEM_INSTRUCTION = """
You are an expert UI/UX designer creating training captions for a mobile UI generation model.
Your task is to analyze the provided mobile screenshot and write a concise, professional prompt that a working product designer would naturally write to describe the interface, focusing on its structure, components, and overall context. Do NOT attempt to transcribe text that appears blurry, distorted, or nonsensical; instead, describe its presence, placement, and general purpose.

CRITICAL ARCHITECTURE RULE:
Your output MUST be strictly divided into two logical halves:

The Aesthetic Prefix (First ~20-40 words): Global style, theme, color palette, and overall visual trend.
The Component Body (Remaining words): The layout and UI components described using standard industry terminology, natural phrasing, and the presence and purpose of key text elements.
Strict Output Rules:

Start: Always start with "High-fidelity mobile UI, [Theme/Vibe], [Color Palette]..." 
Length & Flow: Be concise and natural (approx. 50-120 words max). Do not over-describe every pixel. Use professional shorthand and focus on the visual hierarchy.
Content Guidelines for Canny ControlNet:
The Canny map will provide the spatial lines. Your prompt MUST provide the component names, fills, and describe the presence and general nature of key text elements.

Designer Vocabulary: Use terms like: hero image, primary CTA, secondary button, bottom nav, FAB, modal, card component, list view, segmented control, whitespace, padding, elevation, drop shadow, opacity.
Colors & Materials: Use clear, professional descriptors ("matte dark mode," "glassmorphism," "muted gray," "accent primary color").
Text Elements: Describe the presence, placement, and general purpose of text elements (e.g., 'a prominent title', 'body text below a button', 'a numerical price tag'). Do NOT attempt to transcribe text that appears blurry, distorted, or nonsensical. Focus on the structural role of text within the UI.
Imagery: Briefly state what the placeholder images represent (e.g., "avatar," "lifestyle hero image").
"""

def process_single_seed(client, pil_image, seed_val):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=256,
        seed=seed_val 
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[pil_image, "Describe this mobile UI according to the system instructions."],
                config=config
            )
            return {
                "seed": seed_val,
                "generated_caption": response.text.strip(),
                "error": None
            }
            
        except errors.APIError as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg:
                sleep_time = (2 ** attempt) + (seed_val % 3) 
                time.sleep(sleep_time)
            elif "invalid_grant" in error_msg:
                # Erro fatal de autenticação
                return {"seed": seed_val, "generated_caption": None, "error": "AUTH_EXPIRED"}
            else:
                time.sleep(2)
                
        except Exception as e:
            time.sleep(2)
            
    return {"seed": seed_val, "generated_caption": None, "error": "MAX_RETRIES_EXCEEDED"}

def main():
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    # 1. Identificar Imagens Já Processadas (Lógica de Resume)
    processed_images = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    processed_images.add(data.get("filename"))
        print(f"Encontradas {len(processed_images)} imagens já processadas. Elas serão ignoradas.")

    pending_images = []
    for filename in os.listdir(IMAGES_DIR):
        if filename.lower().endswith(('_output.png', '_output.jpg', '_output.jpeg')):
            if filename not in processed_images:
                pending_images.append(filename)


    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as out_file:
        
        for image_filename in tqdm(pending_images, desc="Processando Imagens"):
            img_path = os.path.join(IMAGES_DIR, image_filename)
            
            try:
                pil_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"\nErro ao carregar {image_filename}: {e}")
                continue

            results = Parallel(n_jobs=N_JOBS, backend="threading")(
                delayed(process_single_seed)(client, pil_image, seed_val)
                for seed_val in tqdm(range(1, NUM_SAMPLES_PER_IMAGE + 1), desc=f"Processing {image_filename}", leave=False)
            )

            valid_variations = [
                {"seed": res["seed"], "generated_caption": res["generated_caption"]} 
                for res in results if res["generated_caption"] is not None
            ]
            
            if len(valid_variations) == 0:
                print(f"\nFalha total ao processar a imagem {image_filename}. Nenhuma semente gerada.")
                continue

            # Salvar no JSONL
            output_data = {
                "filename": image_filename,
                "variations": valid_variations
            }
            
            out_file.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            out_file.flush()
            
            # Alerta caso algumas threads tenham falhado por limite de taxa
            if len(valid_variations) < NUM_SAMPLES_PER_IMAGE:
                print(f"\nAviso: {image_filename} salvou apenas {len(valid_variations)}/{NUM_SAMPLES_PER_IMAGE} variações válidas.")

    print("\n" + "="*40)
    print("PROCESSO FINALIZADO")
    print("="*40)

if __name__ == "__main__":
    main()