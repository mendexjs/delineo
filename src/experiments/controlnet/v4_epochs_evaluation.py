import os
from pathlib import Path
import re
import glob
import time
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import List, Tuple
import argparse

import open_clip
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image

# --- Caminhos e Configurações ---
ENCODER_PATH = "/scratch/delineo_outputs/encoder/ui_semantic_encoder_L14.pth"
FIXED_LORA_PATH = "/scratch/delineo_outputs/lora/delineo_lora_v5/checkpoint-1000"
CONTROLNET_ROOT_DIR = "/scratch/sd35-delineo-finetuned-v4"

TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
PROMPT_BATCH_SIZE = 3
DEVICE = "cuda"

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield i, lst[i:i + n]

def load_semantic_encoder(path):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, preprocess

def get_similarity(model, preprocess, img1, img2):
    with torch.no_grad():
        im1 = preprocess(img1).unsqueeze(0).to(DEVICE)
        im2 = preprocess(img2).unsqueeze(0).to(DEVICE)
        feat1 = model.encode_image(im1)
        feat2 = model.encode_image(im2)
        feat1 /= feat1.norm(dim=-1, keepdim=True)
        feat2 /= feat2.norm(dim=-1, keepdim=True)
        return (feat1 @ feat2.T).item()

def get_controlnet_checkpoints(root_dir: str) -> List[str]:
    import glob
    import os
    import re
    
    # Busca todas as pastas de checkpoint na raiz
    ckpt_dirs = glob.glob(os.path.join(root_dir, "checkpoint-*"))
    valid_ckpts = []
    
    for d in ckpt_dirs:
        # Força o caminho a apontar para a subpasta 'controlnet'
        controlnet_path = os.path.join(d, "controlnet")
        
        # Verifica se o diretório do controlnet existe e contém os arquivos do modelo
        if os.path.isdir(controlnet_path):
            if os.path.exists(os.path.join(controlnet_path, "diffusion_pytorch_model.safetensors")) or \
               os.path.exists(os.path.join(controlnet_path, "config.json")):
                valid_ckpts.append(controlnet_path)
                
    # O regex continua funcionando pois vai encontrar "checkpoint-4740" no meio do caminho
    valid_ckpts.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)) if re.search(r"checkpoint-(\d+)", x) else 10**18)
    
    return valid_ckpts

def attach_sd35_large_pos_embed_if_needed(controlnet, transformer):
    # Matches diffusers pipeline __init__ logic for SD3.5 Large controlnets
    if hasattr(controlnet.config, "use_pos_embed") and controlnet.config.use_pos_embed is False:
        pos_embed = controlnet._get_pos_embed_from_transformer(transformer)
        controlnet.pos_embed = pos_embed.to(controlnet.dtype).to(controlnet.device)
    return controlnet

# --- Amostras de Validação ---
base_dir = Path("../validation_samples")
example_pairs: List[Tuple[str, str]] = [
    (f"{base_dir}/34216_2_input.png", "High-fidelity mobile UI, Airbnb-style, bright and airy aesthetic, pastel teal and white color palette. A search bar at the top reads 'Anywhere · Anytime · 1 guest'. Below, a segmented control with labels 'FOR YOU', 'HOMES', 'EXPERIENCES', 'PLACES', with 'EXPERIENCES' highlighted. A grid layout displays experience cards. The first card shows a wine tasting image, with text '$28 1 hour · wine & wine tasting'. The second card shows an art image, with text '$51 2 hours · art & painting'. A 'Filters' button is present. A bottom navigation bar includes icons and labels: 'EXPLORE', 'SAVED', 'TRIPS', 'INBOX', 'PROFILE'. Each card has a heart icon for saving."),
    (f"{base_dir}/iphone_IMG_899828_input.png", "High-fidelity mobile UI, adventure travel theme, muted color palette with dark overlay. A full-screen hero image of a camper van in a mountain landscape with a dark gradient overlay. The top left corner has a white 'X' icon. Centered is a white outlined 'TURO' logo. Below, in white text, is the heading 'Start your adventure'. A primary CTA 'Sign up' in a light green color, and a secondary button 'Log in' with a white outline."),
    (f"{base_dir}/iphone_IMG_8773_input.png", "High-fidelity mobile UI, e-commerce product listing, clean and bright aesthetic, neutral color palette with white background. A search bar with a magnifying glass icon and the text 'Search' is at the top, with a 'Cancel' button to the right. Below is a product image with a heart icon for saving. The product details include the brand 'SUPPLY', the product name 'Boundless Pima Thread 50wt Cone 5000 Yards', and the price 'US$33.15 US$39.00'. Another product image is shown below."),
    (f"{base_dir}/Android_Android_15_input.png", "High-fidelity mobile UI, clean and bright food app, pastel color palette, with a focus on visual clarity. A modal screen with the title 'Personalize your feed!' and subtitle 'Add 3 or more Tastes to help us serve up better recommendations for you.'. The main content is a 3x3 grid of square image cards, each with a food photo and a text label: 'Appetizers & Snacks', 'Asian', 'BBQ & Grilling', 'Beef', 'Bread', 'Breakfast & Brunch', 'Budget Cooking', 'Cake', 'Casseroles'. An orange primary CTA at the bottom reads 'Done'. A close icon is in the top right corner."),
    (f"{base_dir}/iphone_IMG_906161_input.png", "High-fidelity mobile UI, travel discovery app, clean and modern, light mode with a muted gray background. The main feed features a search bar at the top with the placeholder text 'Search by destination or activity' and a shopping cart icon. Below, a section titled 'Trending Now' displays a full-width card with a lifestyle hero image and overlaid text 'Maybe it's Cold Outside' with a subtitle. Next is a section titled 'Klook Preferred' with a card featuring a nature image and a location pin with the text 'Sydney'. A bottom navigation bar contains icons and labels for 'Explore', 'Destinations', 'Categories', 'Bookings', and 'Account'."),
    (f"{base_dir}/4240_1_input.png", "High-fidelity mobile UI, e-commerce fashion retail, soft and airy aesthetic, pastel color accents on a clean white layout. The top app bar contains a hamburger menu, H&M logo, barcode icon, search icon, and bag icon. A promotional banner reads 'LAST DAY! FESTIVAL SHOP UP TO 50% OFF'. Below, a hero image showcases a woman in a white dress with the text 'TRY THE TREND NOVEL ROMANCE'. A list view follows, presenting black and white photos with text overlays like 'WOMEN'."),
    (f"{base_dir}/Android_Android_4_input.png", "High-fidelity mobile UI, e-commerce app, pastel color palette, clean and modern design. The screen is divided into two sections with product images. The top section shows rain boots and umbrellas, with text labels 'RAINCOATS, RAIN BOOTS, UMBRELLAS' and 'Get rain ready!'. The bottom section features hats, gloves, and scarves, with text labels 'HATS, GLOVES, SCARVES' and 'Discounts against cold weather!'. A bottom navigation bar contains icons for home, search, cart, and profile."),
    (f"{base_dir}/iphone_IMG_9359_input.png", "High-fidelity mobile UI, clean e-commerce app, neutral color palette with white background. The screen features a 'Shop' title and search icon in the top app bar. Below are category tabs: 'Men', 'Women', 'Boys', 'Girls'.  The main content area shows a full-width hero image with the text overlay 'New & Featured'. A second full-width image showcases 'Shoes'. A bottom navigation bar contains icons for Home, Search, Favorites, Bag, and Profile."),
    (f"{base_dir}/1904_1_input.png", "High-fidelity mobile UI, modern music streaming app, light mode, neutral color palette with red accents. The screen features a search bar at the top with the text 'Search and Discover new Music!'. Below is a hero image featuring the artist 'POUYA' and the text 'EXCLUSIVE INTERVIEW OUT NOW! CHARM LADONNA'. A 'Trending Singles' section follows with a 'SEE MORE' link. The list view displays song titles and artist names, such as 'Lil Durk - They Forgot [Prod. By Le...]', accompanied by small album art thumbnails. A bottom navigation bar includes icons for Discover, History, Trending, Video, and Library."),
]

negative_prompt = "ugly, noisy, chaotic hierarchy, heavy skeuomorphism, broken layout, deformed, blurry text, noisy text, phone frame, deformed body, disfigured, poorly drawn face, bad anatomy, extra limbs, missing limbs, floating limbs, grid, collage, tiny text, mutation, mutated, disgusting, amputation, tiling, low quality, unnatural, unprofessional, poorly composed, disconnected limbs"


def create_grid(images: List[Image.Image], row_labels: List[str], col_labels: List[str]) -> Image.Image:
    cols, rows = len(col_labels), len(row_labels)
    w, h = images[0].size
    top_pad, left_pad, cell_pad = 55, 180, 6
    grid_w = left_pad + cols * (w + cell_pad) + cell_pad
    grid_h = top_pad + rows * (h + cell_pad) + cell_pad
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    for c, lab in enumerate(col_labels):
        draw.text((left_pad + cell_pad + c * (w + cell_pad) + 10, 15), lab, fill="black")
    for r, rlab in enumerate(row_labels):
        y0 = top_pad + cell_pad + r * (h + cell_pad)
        draw.text((10, y0 + 10), rlab, fill="black")
        for c in range(cols):
            grid.paste(images[r * cols + c], (left_pad + cell_pad + c * (w + cell_pad), y0))
    return grid

def main():
    time_str = datetime.now().strftime("%d-%m_%H-%M")
    OUTPUT_DIR = f"./controlnet_eval_with_sim_{time_str}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Carregar Encoder e Checkpoints
    encoder, encoder_tf = load_semantic_encoder(ENCODER_PATH)
    ckpt_paths = get_controlnet_checkpoints(CONTROLNET_ROOT_DIR)
    
    prompts = [p[1] for p in example_pairs]
    control_images = [load_image(p[0]) for p in example_pairs]
    seed_labels = [f"seed {s}" for s in [42, 1234, 777, 2001, 1994]]
    per_prompt_images = [[] for _ in prompts]
    row_labels = []

    # 2. Setup do Pipeline SD3.5
    controlnet = SD3ControlNetModel.from_pretrained(ckpt_paths[0], torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    # Loop de Checkpoints
    for ckpt_path in tqdm(ckpt_paths, desc="ControlNet Checkpoints"):
        parent = Path(ckpt_path).parent.name  # e.g. "checkpoint-4740"
        m = re.search(r"checkpoint-(\d+)", parent)
        label = f"controlnet-{m.group(1)}" if m else "controlnet-final"
        row_labels.append(label)
        new_cn = SD3ControlNetModel.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).to(DEVICE)
        new_cn.eval()
        new_cn = attach_sd35_large_pos_embed_if_needed(new_cn, pipe.transformer)
        # pipe.unload_lora_weights()
        pipe.register_modules(controlnet=new_cn)
        pipe.enable_xformers_memory_efficient_attention()

        # pipe.load_lora_weights(
        #     FIXED_LORA_PATH, 
        #     weight_name="pytorch_lora_weights.safetensors", 
        #     adapter_name="ui_style"
        # )
        # pipe.set_adapters("ui_style", adapter_weights=[0.5])

        for start_idx, batch_prompts in chunked(prompts, PROMPT_BATCH_SIZE):
            batch_controls = control_images[start_idx:start_idx + len(batch_prompts)]
            seeds = [42, 1234, 777, 2001, 1994]
            generators = [torch.Generator(DEVICE).manual_seed(s) for _ in batch_prompts for s in seeds]
            print(batch_prompts, batch_controls)
            with torch.inference_mode():
                out = pipe(
                    prompt=batch_prompts,
                    negative_prompt=[negative_prompt] * len(batch_prompts),
                    control_image=batch_controls,
                    controlnet_conditioning_scale=0.85,
                    guidance_scale=4,
                    num_inference_steps=80,
                    num_images_per_prompt=len(seeds),
                    generator=generators,
                    height=TARGET_HEIGHT,
                    width=TARGET_WIDTH,
                ).images

            # --- Cálculo de Similaridade e Anotação ---
            for local_p_idx in range(len(batch_prompts)):
                global_p_idx = start_idx + local_p_idx
                original_sketch = control_images[global_p_idx]
                
                for s_idx in range(len(seeds)):
                    gen_img = out[local_p_idx * len(seeds) + s_idx]
                    
                    # Calcular similaridade usando o encoder do seu TCC
                    sim_score = get_similarity(encoder, encoder_tf, original_sketch, gen_img)
                    
                    # Escrever o score na imagem
                    draw = ImageDraw.Draw(gen_img)
                    # Retângulo de fundo para o texto
                    draw.rectangle([(10, 10), (140, 50)], fill="black")
                    draw.text((20, 15), f"Sim: {sim_score:.4f}", fill="white")
                    
                    per_prompt_images[global_p_idx].append(gen_img)

            torch.cuda.empty_cache()

    # Salvar Grids
    for i, imgs in enumerate(per_prompt_images):
        grid = create_grid(imgs, row_labels=row_labels, col_labels=seed_labels)
        grid.save(os.path.join(OUTPUT_DIR, f"prompt_{i:02d}_eval.png"))

if __name__ == "__main__":
    main()