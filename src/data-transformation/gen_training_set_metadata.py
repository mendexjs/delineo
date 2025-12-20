import json
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = Path("/scratch/delineo_data/train")
PROMPT = "High-fidelity mobile UI design"

# ---------------------

def process_mud(mud_dir, valid_pairs):
    """
    Pattern: ID_input.png matches ID_output.png
    """
    if not mud_dir.exists():
        print(f"Warning: 'mud' directory not found at {mud_dir}")
        return

    # Find all input files
    input_files = list(mud_dir.glob("*_input.png"))
    print(f"Scanning 'mud': Found {len(input_files)} input candidates.")

    for input_path in tqdm(input_files):
        # Parsing: "123_input.png" -> ID "123"
        filename = input_path.name
        file_id = filename.replace("_input.png", "")
        
        # Construct expected output path
        output_filename = f"{file_id}_output.png"
        output_path = mud_dir / output_filename
        
        if output_path.exists():
            # Store relative paths (e.g., "mud/123_input.png")
            valid_pairs.append({
                "image": f"mud/{output_filename}", # TARGET (MUD UI)
                "sketch": f"mud/{filename}",       # INPUT (Programmatically generated sketch)
                "prompt": PROMPT
            })

def process_swire(swire_dir, valid_pairs):
    """
    Swire may have different sketches for the same UI, created by different designers
    Pattern: ID_N_input.png matches ID_output.png
    Example: 123_1_input.png -> 123_output.png
             123_2_input.png -> 123_output.png
    """
    if not swire_dir.exists():
        print(f"Warning: 'swire' directory not found at {swire_dir}")
        return

    input_files = list(swire_dir.glob("*_input.png"))
    print(f"Scanning 'swire': Found {len(input_files)} input candidates.")

    for input_path in tqdm(input_files):
        filename = input_path.name
        parts = filename.split('_')
        file_id = parts[0]
        
        # Construct expected output path
        output_filename = f"{file_id}_output.png"
        output_path = swire_dir / output_filename
        
        if output_path.exists():
            valid_pairs.append({
                "image": f"swire/{output_filename}", # TARGET (Rico UI)
                "sketch": f"swire/{filename}",       # INPUT (Swire human sketch)
                "prompt": PROMPT
            })

def main():
    metadata_path = DATA_ROOT / "metadata.jsonl"
    mud_dir = DATA_ROOT / "mud"
    swire_dir = DATA_ROOT / "swire"
    
    all_entries = []

    # 1. Process Folders
    process_mud(mud_dir, all_entries)
    process_swire(swire_dir, all_entries)

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