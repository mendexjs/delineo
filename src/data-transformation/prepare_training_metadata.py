import json
import os
from pathlib import Path
from tqdm import tqdm
from utils import load_ui_captions_map

# --- CONFIGURATION ---
DATA_ROOT = Path("/scratch/delineo_data/train")
PROMPT = "High-fidelity mobile UI design"
INVALID_UI = "NOISY UI"

# ---------------------

captions_map = load_ui_captions_map()


def get_file_id(input_name, ds_name):
    if ds_name == 'swire':
        parts = input_name.split('_')
        return parts[0]
    return input_name.replace("_input.png", "")


def process_dataset(dir, valid_pairs, dataset_name):
    if not dir.exists():
        print(f"Warning: '{dataset_name}' directory not found at {dir}")
        return
    
    invalid_samples = set()
    input_files = list(dir.glob("*_input.png"))
    print(f"Scanning '{dataset_name}': Found {len(input_files)} input candidates.")

    for input_path in tqdm(input_files):
        input_name = input_path.name
        input_filename = f"{dataset_name}/{input_name}"
        file_id = get_file_id(input_name, dataset_name)
        
        # Construct expected output path
        output_name = f"{file_id}_output.png"
        output_path = dir / output_name
        output_filename = f"{dataset_name}/{output_name}"
        caption = captions_map.get(output_filename)
        if not caption or caption == INVALID_UI:
            invalid_samples.update([input_filename, output_filename])
            continue
        
        if output_path.exists():
            valid_pairs.append({
                "input_file_name": input_filename,  # INPUT (Swire human sketch)
                "output_file_name": output_filename,     # TARGET (Rico UI)
                "text": caption
            })
    
    for invalid_swire in invalid_samples:
        try:
            os.remove(dir / invalid_swire)
        except:
            continue


def main():
    metadata_path = DATA_ROOT / "metadata.jsonl"
    mud_dir = DATA_ROOT / "mud"
    swire_dir = DATA_ROOT / "swire"
    vins_dir = DATA_ROOT / "vins"
    
    all_entries = []

    # 1. Process Folders
    process_dataset(mud_dir, all_entries, 'mud')
    process_dataset(swire_dir, all_entries, 'swire')
    process_dataset(vins_dir, all_entries, 'vins')

    # 2. Write to JSONL
    print(f"Writing {len(all_entries)} pairs to {metadata_path}...")
    
    with open(metadata_path, 'w') as f:
        for entry in tqdm(all_entries):
            json.dump(entry, f)
            f.write('\n')

    print("Done! Example entry:")
    if all_entries:
        print(json.dumps(all_entries[0], indent=2))

if __name__ == "__main__":
    main()