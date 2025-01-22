from diffusers import StableDiffusionPipeline
import torch

def download_sd_model():
    print("Downloading Stable Diffusion 2.0 base model...")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            safety_checker=None
        )
        print("Model downloaded successfully to the default cache directory!")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    download_sd_model()