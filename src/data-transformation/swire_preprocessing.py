import cv2
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import crop_bars_opencv, resize_width_and_crop

# --- CONFIGURATION ---
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
N_JOBS = -1 # All CPUs available

RICO_STATUS_HEIGHT = 42
RICO_NAV_HEIGHT = 86


# These samples are corrupted
SWIRE_INVALID_SAMPLES = (
    '71922_2.jpg',
    '567986_2.jpg',
    '33181_1.jpg',
    '30635_1.jgp',
    '29822_1.jpg',
    '29799_1.jpg',
    '29770_1.jpg',
    '29766_1.jpg',
    '23424_1.jpg',
    '23426_1.jpg',
    '23427_1.jpg',
    '20613_1.jpg',
    '8946_3.jpg',
    '2961_3.jpg',
    '2959_1.jpg',
    '2220_2.jpg',
)

SWIRE_VALIDATION_SAMPLES = (
    '4240', 
    '12922',
    '33178',
    '56563',
)

def setup_paths():
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent.parent
    
    # Input Directories
    swire_dir = src_dir / "raw-data" / "swire"
    rico_dir = src_dir / "raw-data" / "rico"
    
    # Output to /scratch -- NVMe SSD
    out_train_dir = Path("/scratch/delineo_data/train/swire")
    out_validation_dir = Path("/scratch/delineo_data/validation")
    
    return swire_dir, rico_dir, out_train_dir, out_validation_dir

def process_single_pair(swire_path, rico_dir, out_train_dir, out_validation_dir):
    """
    Worker function to process a single image pair.
    """
    try:
        filename = swire_path.name
        rico_id = filename.split("_")[0]

        if filename in SWIRE_INVALID_SAMPLES:
            return f"SKIP: {rico_id} is listed in SWIRE_INVALID_SAMPLES"
        
        # 2. Find Correspondence in Rico
        rico_path = None
        potential_path = rico_dir / f"{rico_id}.jpg"
        if potential_path.exists():
            rico_path = potential_path
        
        if rico_path is None:
            return f"SKIP: No match for ID {rico_id}"

        # 3. Load Images
        img_swire = cv2.imread(str(swire_path))
        img_rico = cv2.imread(str(rico_path))

        if img_swire is None or img_rico is None:
            return f"ERROR: Failed to load images for {rico_id}"

        # 4. Resize
        # SWIRE (Wireframe) -> Use NEAREST to keep sharp edges (black/white)
        swire_resized = resize_width_and_crop(img_swire, TARGET_WIDTH, TARGET_HEIGHT, interpolation=cv2.INTER_NEAREST)
        
        # RICO (Screenshot) -> Use AREA for high-quality downsampling of UI text
        rico_resized = crop_bars_opencv(resize_width_and_crop(img_rico, TARGET_WIDTH, TARGET_HEIGHT, interpolation=cv2.INTER_AREA), RICO_STATUS_HEIGHT, RICO_NAV_HEIGHT)
        
        # Save
        export_dir = out_validation_dir if rico_id in SWIRE_VALIDATION_SAMPLES else out_train_dir
        compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        cv2.imwrite(str(export_dir / f"{swire_path.stem}_input.png"), swire_resized, compression_params)
        cv2.imwrite(str(export_dir / f"{rico_id}_output.png"), rico_resized, compression_params)

        return None

    except Exception as e:
        return f"EXCEPTION: {filename} - {str(e)}"

def main():
    # 1. Setup Directories
    swire_dir, rico_dir, out_train_dir, out_validation_dir = setup_paths()
    
    print(f"--- Configuration ---")
    print(f"Input Swire: {swire_dir}")
    print(f"Input Rico:  {rico_dir}")
    print(f"Output (train):      {out_train_dir}")
    print(f"Output (validation):      {out_validation_dir}")
    print(f"Target Res:  {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"---------------------")

    # Validate Inputs
    if not swire_dir.exists():
        print(f"❌ Error: Swire directory not found at {swire_dir}")
        return
    if not rico_dir.exists():
        print(f"❌ Error: Rico directory not found at {rico_dir}")
        return

    # Create Output Dirs
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_validation_dir.mkdir(parents=True, exist_ok=True)

    # 2. Collect Files
    # Get all .jpg files in swire directory
    swire_files = list(swire_dir.glob("*.jpg"))
    total_files = len(swire_files)
    print(f"Found {total_files} swire candidates. Processing...")

    # 3. Parallel Processing
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_single_pair)(
            p, rico_dir, out_train_dir, out_validation_dir
        ) for p in tqdm(swire_files, total=total_files, unit="img")
    )

    # 4. Reporting
    skipped = [r for r in results if r and r.startswith("SKIP")]
    errors = [r for r in results if r and (r.startswith("ERROR") or r.startswith("EXCEPTION"))]
    success_count = len(results) - len(skipped) - len(errors)

    print(f"\n--- Processing Complete ---")
    print(f"✅ Successfully processed: {success_count}")
    print(f"⏭️  Skipped (No match):    {len(skipped)}")
    print(f"❌ Errors:                 {len(errors)}")
    
    if errors:
        print("\nFirst 10 errors:")
        for e in errors[:10]:
            print(e)

if __name__ == "__main__":
    main()