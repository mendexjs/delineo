import os
import json
import time
from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai import errors
from joblib import Parallel, delayed

# Configurações do GCP e Modelo
PROJECT_ID = "hybrid-cabinet-482822-p5"
LOCATION = "global"
MODEL_ID = "gemini-2.0-flash"

# Caminhos do seu experimento final
EXPERIMENT_DIR = "./best_results"
OUTPUT_JSONL = f"{EXPERIMENT_DIR}/captions.jsonl"

MAX_RETRIES = 15
N_JOBS = 4

SYSTEM_INSTRUCTION = """
You are an expert UI/UX designer creating training captions for a mobile UI generation model. 
Your task is to analyze the provided mobile screenshot and write a concise, professional prompt that a working product designer would naturally write to describe the interface. 

**CRITICAL ARCHITECTURE RULE:**
Your output MUST be strictly divided into two logical halves:
1.  **The Aesthetic Prefix (First ~20-40 words):** Global style, theme, color palette, and overall visual trend. 
2.  **The Component Body (Remaining words):** The layout and UI components described using standard industry terminology, natural phrasing, and key text values.

**Strict Output Rules:**
1.  **Start:** Always start with "High-fidelity mobile UI, [Theme/Vibe], [Color Palette]..." unless rejecting.
2.  **Length & Flow:** Be concise and natural (approx. 50-120 words max). Do not over-describe every pixel. Use professional shorthand and focus on the visual hierarchy.
3.  **Rejection (Quality Control):** If the image meets ANY of the following criteria, you must abort the description and return exactly and ONLY the phrase: "NOISY UI"
    * **Non-UI:** Blank screens, error messages, or purely text documents.
    * **Outdated UI:** Heavy skeuomorphism, archaic OS aesthetics (e.g., Windows 95) or outdated gradients.
    * **Amateur UI:** Broken layouts, bad padding/alignment, poor contrast, or chaotic hierarchy.

**Content Guidelines for Canny ControlNet:**
The Canny map will provide the spatial lines. Your prompt MUST provide the component names, fills, and key text.
* **Designer Vocabulary:** Use terms like: hero image, primary CTA, secondary button, bottom nav, FAB, modal, card component, list view, segmented control, whitespace, padding, elevation, drop shadow, opacity.
* **Colors & Materials:** Use clear, professional descriptors ("matte dark mode," "glassmorphism," "muted gray," "accent primary color").
* **Exact Text Rendering:** If prominent text exists (like headers or CTAs), enclose it in single quotes (e.g., CTA 'Sign Up').
* **Imagery:** Briefly state what the placeholder images represent (e.g., "avatar," "lifestyle hero image").

**Example (Simple UI):**
High-fidelity mobile UI, ultra-minimalist dark mode aesthetic, flat design, high contrast. A clean login screen with generous whitespace and a strong visual hierarchy. Center stage is a stark white text field for 'Email Address', sitting just above a pill-shaped primary CTA in electric lime reading 'Enter'. The top-center features a minimal geometric triangle logo. Flat elevation with no drop shadows.

**Example (Complex UI):**
High-fidelity mobile UI, modern travel aesthetic, airy lighting, glassmorphism elements, clean white background with soft yellow accents. A travel discovery feed featuring a transparent header overlapping a full-width nature hero image. Below is a frosted glass search bar with a 'Filter' icon. A horizontal scrollable section contains circular 'Story' avatars with gradient borders. The main vertical list view uses large, rounded-corner card components. A sample card shows a forest photo, bold serif title 'Alpine Lodge', a yellow star rating badge, and a bottom-right price tag '$120/night'. A minimalist line-art bottom nav includes Home, Search, Saved, Profile.
"""

def process_seed_image(client, img_path, seed_id):
    """Processa uma única imagem (seed) gerada e retorna sua caption."""
    try:
        # Carregamento da imagem movido para dentro da thread por segurança
        pil_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        return {"seed": seed_id, "caption": None, "error": f"Erro ao carregar imagem: {e}"}

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.5,
        top_p=0.9,
        max_output_tokens=256,
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[pil_image, "Describe this mobile UI according to the system instructions."],
                config=config
            )
            return {
                "seed": seed_id,
                "caption": response.text.strip(),
                "error": None
            }
            
        except errors.APIError as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg:
                sleep_time = (2 ** attempt) + (int(seed_id) % 3) 
                time.sleep(sleep_time)
            elif "invalid_grant" in error_msg:
                return {"seed": seed_id, "caption": None, "error": "AUTH_EXPIRED"}
            else:
                time.sleep(2)
                
        except Exception as e:
            time.sleep(2)
            
    return {"seed": seed_id, "caption": None, "error": "MAX_RETRIES_EXCEEDED"}

def main():
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    # 1. Lógica de Resume: Identificar samples (pastas) já processadas
    processed_samples = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    processed_samples.add(data.get("sample_id"))
        print(f"[*] Encontrados {len(processed_samples)} samples já processados no JSONL. Eles serão ignorados.")

    # 2. Mapear as subpastas no diretório do experimento
    pending_samples = []
    for item in os.listdir(EXPERIMENT_DIR):
        item_path = os.path.join(EXPERIMENT_DIR, item)
        # Verifica se é um diretório e se não está no JSONL
        if os.path.isdir(item_path) and item not in processed_samples:
            pending_samples.append(item)

    print(f"[*] Iniciando extração de captions para {len(pending_samples)} samples...\n")

    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as out_file:
        
        for sample_id in tqdm(pending_samples, desc="Processando Samples"):
            sample_dir = os.path.join(EXPERIMENT_DIR, sample_id)
            
            # Coletar todas as imagens seed_X.png dentro desta pasta
            seed_images = []
            for filename in os.listdir(sample_dir):
                if filename.startswith("seed_") and filename.endswith(".png"):
                    # Extrair o número da seed (ex: "seed_42.png" -> 42)
                    seed_val = int(filename.replace("seed_", "").replace(".png", ""))
                    img_path = os.path.join(sample_dir, filename)
                    seed_images.append((img_path, seed_val))
            
            if not seed_images:
                continue

            # 3. Paralelizar chamadas para a Vertex AI para as imagens deste sample
            results = Parallel(n_jobs=N_JOBS, backend="threading")(
                delayed(process_seed_image)(client, img_path, seed_val)
                for img_path, seed_val in seed_images
            )

            # Filtrar apenas os resultados válidos
            valid_captions = [
                {"seed": res["seed"], "caption": res["caption"]} 
                for res in results if res["caption"] is not None
            ]
            
            if len(valid_captions) == 0:
                print(f"\n[Aviso] Falha total ao processar as imagens do sample {sample_id}.")
                continue

            # 4. Estruturar e salvar no JSONL
            output_data = {
                "sample_id": sample_id,
                "captions": valid_captions
            }
            
            out_file.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            out_file.flush()
            
            if len(valid_captions) < len(seed_images):
                print(f"\nAviso: {sample_id} salvou apenas {len(valid_captions)}/{len(seed_images)} captions válidas.")

    print("\n" + "="*40)
    print("EXTRAÇÃO DE CAPTIONS FINALIZADA")
    print("="*40)

if __name__ == "__main__":
    main()