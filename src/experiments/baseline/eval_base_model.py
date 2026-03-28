import os
import json
import random
import torch
from datetime import datetime
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline, StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
from tqdm import tqdm

# ==============================================================================
# --- CONFIGURAÇÕES ---
# ==============================================================================

IMG_BASE_DIR = "../validation_samples"
JSONL_PATH = "../ui_validation_captions.jsonl"
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
original_canny_controlnet = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
DEVICE = "cuda"

# Amostras hardcoded para o teste (sem a extensão)
SAMPLES_TO_TEST = ['5334_output', '2213_output', 'iphone_IMG_899828_output', '12922_output', 'Android_Android_15_output', 'iphone_IMG_8773_output']
SEEDS = [42, 1234]
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280

# Select the mode, to infer base line samples using image-editing (i2i), controlnet (cn) or default text-to-image (t2i)
MODE = "t2i" # t2i | i2i | cn
I2I_STRENGTH = 0.9
CN_STRENGTH = 0.7
GUIDANCE_SCALE = 5
NUM_INFERENCE_STEPS = 100

# ==============================================================================

def load_sketch_img(fname):
    """Carrega o sketch de entrada resolvendo os IDs de designer do SWIRE."""
    base_id = fname.replace("_output.png", "")
    possible_inputs = [
        f"{base_id}_input.png",
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
        print(f"[Aviso] Nenhum arquivo de sketch encontrado para a base: {base_id}")
        return None
    
    chosen_sketch = random.choice(existing_inputs)
    print(f"  -> Sketch carregado: {chosen_sketch}")
    return load_image(os.path.join(IMG_BASE_DIR, chosen_sketch))

def load_captions(jsonl_path):
    """Lê o arquivo JSONL e retorna um dicionário mapeando filename -> caption."""
    captions_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                captions_dict[data["filename"]] = data["caption"]
    return captions_dict

def get_pipeline():
    match MODE:
        case 'cn':
            controlnet = SD3ControlNetModel.from_pretrained(original_canny_controlnet, torch_dtype=torch.bfloat16)
            return StableDiffusion3ControlNetPipeline.from_pretrained(
                MODEL_ID, controlnet=controlnet, torch_dtype=torch.bfloat16
            ).to(DEVICE)
        case 'i2i':
            return StableDiffusion3Img2ImgPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16
            ).to(DEVICE)
    return StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)

def get_params(input_img):
    match MODE:
        case 'cn':
            return {
                "controlnet_conditioning_scale": CN_STRENGTH,
                "control_image": input_img
            }

        case 'i2i':
            return {
                "strength": I2I_STRENGTH,
                "image": input_img
            }

    return {}



def main():
    # 1. Preparar Diretório de Saída
    time_str = datetime.now().strftime("%d-%m_%H-%M")
    output_dir = f"./{MODE}_base_model_test_{time_str}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Resultados base serão salvos em: {output_dir}")

    # 2. Carregar Captions
    captions_dict = load_captions(JSONL_PATH)

    print(f"[*] Carregando o modelo base {MODEL_ID} na {DEVICE}...")
    pipe = get_pipeline()
    
    # Otimizações para a A100
    pipe.enable_xformers_memory_efficient_attention()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # 4. Loop de Inferência
    for sample_id in tqdm(SAMPLES_TO_TEST, desc=f"Evaluating sample"):
        print(f"\n[*] Processando amostra: {sample_id}")
        filename = f"{sample_id}.png"
        
        # Recuperar caption
        caption = captions_dict.get(filename)
        if not caption:
            print(f"[Aviso] Caption não encontrada para {filename}. Pulando...")
            continue
            
        # Carregar imagem de referência (sketch)
        sketch_img = load_sketch_img(filename)
        if not sketch_img:
            continue
            
        # Preparar os geradores para processamento em lote
        # Isso permite gerar as duas imagens de uma vez na sua A100
        generators = [torch.Generator(DEVICE).manual_seed(s) for s in SEEDS]

        print("  -> Gerando inferências no modelo base...")
        with torch.inference_mode():
            out = pipe(
                prompt=[caption],
                height=TARGET_HEIGHT,
                width=TARGET_WIDTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                num_images_per_prompt=len(SEEDS),
                generator=generators,
                **get_params(sketch_img)
            ).images

        # 5. Salvar Imagens
        for i, seed in enumerate(SEEDS):
            gen_img = out[i]
            save_path = os.path.join(output_dir, f"{sample_id}_seed_{seed}.png")
            gen_img.save(save_path)
            print(f"  -> Salvo: {save_path}")

        # Limpar cache da VRAM
        torch.cuda.empty_cache()

    print(f"\n[+] Teste do modelo base concluído! Arquivos em {output_dir}")

if __name__ == "__main__":
    main()