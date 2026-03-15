import os
import json
import random
import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
from tqdm import tqdm

JSONL_PATH = "../ui_validation_captions.jsonl"
IMG_BASE_DIR = "../validation_samples"
CONTROLNET_PATH = "/scratch/delineo_final/controlnet"
LORA_PATH = "/scratch/delineo_final/lora"
OUTPUT_BASE_DIR = "./delineo_final_eval_results"

# ==============================================================================
# --- HIPERPARAMETERS ---
# ==============================================================================

TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
DEVICE = "cuda"

NUM_INFERENCE_STEPS = 80
GUIDANCE_SCALE = 9.5
CONTROLNET_CONDITIONING_SCALE = 0.75
LORA_SCALE = 0.75

# 10 Seeds fixas para reprodutibilidade
SEEDS = [42, 1234, 777, 2001, 1994, 8080, 5555, 9999, 1111, 3333]

NEGATIVE_PROMPT = (
    "ugly, noisy, chaotic hierarchy, heavy skeuomorphism, broken layout, "
    "deformed, blurry text, noisy text, phone frame, deformed body, disfigured, "
    "poorly drawn face, bad anatomy, extra limbs, missing limbs, floating limbs, "
    "grid, collage, tiny text, mutation, mutated, disgusting, amputation, tiling, "
    "low quality, unnatural, unprofessional, poorly composed, disconnected limbs"
)

# ==============================================================================

def chunked(lst, n):
    """Divide uma lista em pedaços (batches) de tamanho n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_jsonl(path: str) -> List[Dict]:
    """Lê o arquivo JSONL e retorna uma lista de dicionários."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_hyperparameters(output_dir: str):
    """Salva todas as configurações em um arquivo YAML para rastreabilidade."""
    config = {
        "JSONL_PATH": JSONL_PATH,
        "CONTROLNET_PATH": CONTROLNET_PATH,
        "LORA_PATH": LORA_PATH,
        "TARGET_WIDTH": TARGET_WIDTH,
        "TARGET_HEIGHT": TARGET_HEIGHT,
        "NUM_INFERENCE_STEPS": NUM_INFERENCE_STEPS,
        "GUIDANCE_SCALE": GUIDANCE_SCALE,
        "CONTROLNET_CONDITIONING_SCALE": CONTROLNET_CONDITIONING_SCALE,
        "LORA_SCALE": LORA_SCALE,
        "SEEDS": SEEDS,
        "NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "EXECUTION_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    yaml_path = os.path.join(output_dir, "hyperparameters.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def load_sketch_img(fname):
    base_id = fname.replace("_output.png", "")
    possible_inputs = [
        f"{base_id}_input.png",
        # Swire designer ids
        f"{base_id}_1_input.png",
        f"{base_id}_2_input.png",
        f"{base_id}_3_input.png",
        f"{base_id}_4_input.png"
    ]

    existing_inputs = [
        f for f in possible_inputs 
        if os.path.exists(os.path.join(IMG_BASE_DIR, f))
    ]

    if not existing_inputs:
        print(f"Nenhum arquivo de sketch encontrado para a base: {base_id}")
        return None
    
    chosen_sketch = random.choice(existing_inputs)
    return load_image(os.path.join(IMG_BASE_DIR, chosen_sketch))


def main():
    # 1. Preparar Diretório de Saída e YAML
    time_str = datetime.now().strftime("%d-%m_%H-%M")
    output_dir = f"{OUTPUT_BASE_DIR}_{time_str}"
    os.makedirs(output_dir, exist_ok=True)
    save_hyperparameters(output_dir)
    print(f"[*] Resultados e YAML serão salvos em: {output_dir}")

    # 2. Carregar Dados de Entrada
    print(f"[*] Lendo captions de: {JSONL_PATH}")
    dataset = load_jsonl(JSONL_PATH)
    print(f"[*] Total de amostras para processar: {len(dataset)}")

    # 3. Inicializar Modelos (ControlNet + SD3.5 + LoRA)
    print("[*] Carregando Modelos para a VRAM...")
    controlnet = SD3ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    
    pipe.set_progress_bar_config(disable=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    # Carregar LoRA
    pipe.enable_xformers_memory_efficient_attention()
    pipe.load_lora_weights(
        LORA_PATH, 
        weight_name="pytorch_lora_weights.safetensors", 
        adapter_name="ui_style"
    )
    pipe.set_adapters("ui_style", adapter_weights=[LORA_SCALE])

    # 4. Loop de Inferência
    print(f"[*] Iniciando Geração Final (Processando 1 amostra por vez, gerando {len(SEEDS)} seeds)...")
    for item in tqdm(dataset, desc="Processando Amostras"):
        
        caption = item["caption"]
        fname = item["filename"]
        
        control_img = load_sketch_img(fname)
        if not control_img:
            continue

        generators = [torch.Generator(DEVICE).manual_seed(s) for s in SEEDS]

        with torch.inference_mode():
            out = pipe(
                prompt=[caption],
                negative_prompt=[NEGATIVE_PROMPT],
                control_image=[control_img],
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                num_images_per_prompt=len(SEEDS),
                generator=generators,
                height=TARGET_HEIGHT,
                width=TARGET_WIDTH,
            ).images

        # 5. Salvar Imagens Organizadas por Amostra
        sample_name = Path(fname).stem
        sample_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        for s_idx, seed in enumerate(SEEDS):
            gen_img = out[s_idx]
            save_path = os.path.join(sample_dir, f"seed_{seed}.png")
            gen_img.save(save_path)

        torch.cuda.empty_cache()

    print(f"\n[+] Inferência concluída com sucesso! Resultados em: {output_dir}")

if __name__ == "__main__":
    main()