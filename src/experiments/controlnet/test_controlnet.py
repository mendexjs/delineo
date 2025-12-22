import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
import time

TARGET_WIDTH = 720
TARGET_HEIGHT = 1280

# 1. SETUP
# Path to your latest checkpoint (check your folder for the highest number)
checkpoint_path = "/home/matheus_mendes/projects/delineo/src/training/controlnet-training/sd35-delineo-finetuned-v1/checkpoint-2500/controlnet"
original_canny_controlnet = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
base_model = "stabilityai/stable-diffusion-3.5-large"

# Load the model (switch to original_canny_controlnet to test results without our training)
controlnet = SD3ControlNetModel.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# 2. INPUT
# Use your specific validation image
control_image_path = "/scratch/delineo_data/validation/408_input.png"
control_image = load_image(control_image_path)

prompt = """
High-fidelity mobile UI design of a registration screen. Features a prominent 'EasyBox' logo in 
blue and orange at the top. Below is a stacked form with input fields for Name, Email, Password, and Confirm Password. 
Includes a Terms of Use agreement checkbox and a solid blue primary submit button. 
Clean, trustworthy aesthetic with blue accents."
"""
print("Generating...")
image = pipe(
    prompt, 
    control_image=control_image, 
    height=TARGET_HEIGHT,
    width=TARGET_WIDTH,
    num_inference_steps=28,
    controlnet_conditioning_scale=1.0  # How strong the lines are forced
).images[0]

# 4. SAVE
image.save(f"vertical_test_result-{time.time()}.png")
print("Saved 'vertical_test_result.png'")