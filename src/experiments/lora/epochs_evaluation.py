import os
from pathlib import Path
import re
import glob
import time
from datetime import datetime
from typing import List, Tuple
import argparse


import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image


CUSTOM_CONTROLNET_PATH = "/scratch/delineo_outputs/controlnet-v3/checkpoint-4074/controlnet"
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
PROMPT_BATCH_SIZE = 3 # 3 simultaneous is the max for A100 80GB without cpu offload

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA evaluation script")
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Adapter version number (e.g., 2, 3, 4...)"
    )
    return parser.parse_args()


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

test_seeds = [42, 1234, 777, 2001, 1994]
adapter_weight = 0.7
controlnet_conditioning_scale = 0.75
num_inference_steps = 80
guidance_scale = 5

KEEP_PIPE_ON_GPU = True 


def checkpoint_step(ckpt_path: str) -> int:
    parent = os.path.basename(os.path.dirname(ckpt_path))
    m = re.match(r"checkpoint-(\d+)", parent)
    if m:
        return int(m.group(1))
    return 10**18  # "final" goes last


def checkpoint_label(ckpt_path: str) -> str:
    parent = os.path.basename(os.path.dirname(ckpt_path))
    if parent.startswith("checkpoint-"):
        return parent
    return "final"


def get_checkpoints(lora_dir: str) -> List[str]:
    checkpoint_dirs = glob.glob(os.path.join(lora_dir, "checkpoint-*"))
    files = []
    for d in checkpoint_dirs:
        p = os.path.join(d, "pytorch_lora_weights.safetensors")
        if os.path.exists(p):
            files.append(p)

    root_final = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(root_final):
        files.append(root_final)

    files.sort(key=checkpoint_step)
    return files


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield i, lst[i:i + n]


def create_grid(images: List[Image.Image], row_labels: List[str], col_labels: List[str]) -> Image.Image:
    cols = len(col_labels)
    rows = len(row_labels)
    assert len(images) == rows * cols, f"Expected {rows*cols} images, got {len(images)}"

    w, h = images[0].size
    top_pad = 55
    left_pad = 180
    cell_pad = 6

    grid_w = left_pad + cols * (w + cell_pad) + cell_pad
    grid_h = top_pad + rows * (h + cell_pad) + cell_pad

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    # Column labels (seeds)
    for c, lab in enumerate(col_labels):
        x = left_pad + cell_pad + c * (w + cell_pad) + 10
        draw.text((x, 15), lab, fill="black")

    # Rows
    for r, rlab in enumerate(row_labels):
        y0 = top_pad + cell_pad + r * (h + cell_pad)
        draw.text((10, y0 + 10), rlab, fill="black")
        for c in range(cols):
            idx = r * cols + c
            x0 = left_pad + cell_pad + c * (w + cell_pad)
            grid.paste(images[idx], (x0, y0))

    return grid


def main():
    args = parse_args()
    version = args.version

    LORA_DIR = f"/scratch/delineo_outputs/lora/delineo_lora_v{version}"
    time_without_seconds = datetime.now().strftime("%d-%m_%H-%M")
    OUTPUT_DIR = f"./lora_eval_v{version}_{time_without_seconds}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint_files = get_checkpoints(LORA_DIR)
    print(f"Found {len(checkpoint_files)} checkpoints to test.")
    if not checkpoint_files:
        raise RuntimeError(f"No checkpoints found in {LORA_DIR}")

    prompts = [p[1] for p in example_pairs]
    control_images = [load_image(p[0]) for p in example_pairs]

    seed_labels = [f"seed {s}" for s in test_seeds]

    # one bucket per prompt: will accumulate rows*cols images (row-major across checkpoints)
    per_prompt_images: List[List[Image.Image]] = [[] for _ in prompts]
    row_labels: List[str] = []

    # Load models once
    controlnet = SD3ControlNetModel.from_pretrained(CUSTOM_CONTROLNET_PATH, torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_xformers_memory_efficient_attention()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # CPU offload slows down inference but allow to fit more in VRAM
    # pipe.enable_model_cpu_offload()

    # main loop: checkpoints -> prompt batches
    for ckpt_path in tqdm(checkpoint_files, desc="Checkpoints"):
        ckpt_name = checkpoint_label(ckpt_path)
        row_labels.append(ckpt_name)
        print(f"\nCheckpoint: {ckpt_name}")

        pipe.load_lora_weights(
            os.path.dirname(ckpt_path),
            weight_name=os.path.basename(ckpt_path),
            adapter_name="ui_style",
        )
        pipe.set_adapters("ui_style", adapter_weights=[adapter_weight])

        for start_idx, batch_prompts in chunked(prompts, PROMPT_BATCH_SIZE):
            batch_controls = control_images[start_idx:start_idx + len(batch_prompts)]

            # One generator per output image in this batch:
            # batch_size = len(batch_prompts) * num_images_per_prompt
            generators = []
            for _p in range(len(batch_prompts)):
                for s in test_seeds:
                    generators.append(torch.Generator(device="cuda").manual_seed(s))

            with torch.inference_mode():
                out = pipe(
                    prompt=batch_prompts,
                    negative_prompt=[negative_prompt] * len(batch_prompts),
                    control_image=batch_controls,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=len(test_seeds),
                    generator=generators,
                    height=TARGET_HEIGHT,
                    width=TARGET_WIDTH,
                ).images

            # out is ordered by prompt, then images_per_prompt
            cols = len(test_seeds)
            for local_p_idx in range(len(batch_prompts)):
                global_p_idx = start_idx + local_p_idx
                seg = out[local_p_idx * cols:(local_p_idx + 1) * cols]
                per_prompt_images[global_p_idx].extend(seg)

            # free temporary tensors quicker
            del out
            torch.cuda.empty_cache()

        pipe.unload_lora_weights()
        torch.cuda.empty_cache()

    # Save one file per prompt: rows=checkpoints, cols=seeds
    for i, imgs in enumerate(per_prompt_images):
        grid = create_grid(imgs, row_labels=row_labels, col_labels=seed_labels)
        out_path = os.path.join(OUTPUT_DIR, f"prompt_{i:02d}_across_checkpoints.png")
        grid.save(out_path)
        print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"\nTotal generation time: {int(elapsed//60)}m {elapsed%60:.2f}s")
