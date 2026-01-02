import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
import time

TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
CHECKPOINT = 6250 # replace to compare different checkpoints
NUM_INFERENCE_STEPS = 100 # previously 28
CONTROL_STRENGTH = 0.9

# 1. SETUP
# Path to your latest checkpoint (check your folder for the highest number)
checkpoint_path = f"/scratch/delineo_data/.bkp/checkpoint-{CHECKPOINT}/controlnet"
original_canny_controlnet = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
base_model = "stabilityai/stable-diffusion-3.5-large"

# Load the model (switch to original_canny_controlnet to test results without our training)
controlnet = SD3ControlNetModel.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# 2. INPUT
control_image_path_1 = "./validation_samples/408_input.png"
control_image_1 = load_image(control_image_path_1)
prompt_1 = "High-fidelity mobile UI design of a registration screen. Features a prominent 'EasyBox' logo in blue and orange at the top. Below is a stacked form with input fields for Name, Email, Password, and Confirm Password. Includes a Terms of Use agreement checkbox and a solid blue primary submit button. Clean, trustworthy aesthetic with blue accents."
print("Generating IMG 1...")
image_1 = pipe(
    prompt_1, 
    control_image=control_image_1, 
    height=TARGET_HEIGHT,
    width=TARGET_WIDTH,
    num_inference_steps=NUM_INFERENCE_STEPS,
    controlnet_conditioning_scale=CONTROL_STRENGTH # How strong the lines are forced
).images[0]


control_image_path_2 = "./validation_samples/4240_1_input.png"
control_image_2 = load_image(control_image_path_2)
prompt_2 = "High-fidelity mobile UI design of the H&M app main page. Features a hero carousel with fashion trends, navigation for 'Women' and 'Men' categories, and a bright promotional banner displaying 'Last Day of Festival Shop - 50% OFF'. Minimalist, modern e-commerce aesthetic."
print("Generating IMG 2...")
image_2 = pipe(
    prompt_2, 
    control_image=control_image_2, 
    height=TARGET_HEIGHT,
    width=TARGET_WIDTH,
    num_inference_steps=NUM_INFERENCE_STEPS,
    controlnet_conditioning_scale=CONTROL_STRENGTH  # How strong the lines are forced
).images[0]

# 4. SAVE
image_1.save(f"checkpoint-{CHECKPOINT}-408_{NUM_INFERENCE_STEPS}_steps_{CONTROL_STRENGTH}_control_result-{time.time()}.png")
image_2.save(f"checkpoint-{CHECKPOINT}-4240_1_{NUM_INFERENCE_STEPS}_steps_{CONTROL_STRENGTH}_control_result-{time.time()}.png")
print("Saved results")