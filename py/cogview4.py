import torch
import numpy as np
from PIL import Image
import os
import folder_paths

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def check_and_download_model(model_path, repo_id):
    model_path = os.path.join(folder_paths.models_dir, "CogView", model_path)

    if not os.path.exists(model_path):
        print(f"Downloading {repo_id} model to {model_path} ...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt", ".git", ".gitattributes"])
    return model_path

class CogView4_Wrapper:

    def __init__(self):
        self.NODE_NAME = 'CogView4 Wrapper'
        self.model_name = ""
        self.dtype = ""
        self.pipe = None


    @classmethod
    def INPUT_TYPES(self):
        model_list =['CogView4-6B']
        dtype_list = ['bf16', 'fp32']
        default_prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background.The license plate number of the car is 'CogView4'. The car sprinted along a coastal road, with the same sports car printed on the roadside billboard and large Chinese text '遥遥领先' written to it. The text was yellow, with thick strokes and heavy shadow lines."
        return {
            "required": {
                "model": (model_list,),
                "dtype": (dtype_list,),
                "prompt":("STRING", {"default":default_prompt, "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1e14}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.1, "max": 100, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "width": ("INT", {"default": 1536, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "cache_model": ("BOOLEAN", {"default": True,}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'cogview4'
    CATEGORY = '😺dzNodes/CogView4 Wrapper'

    def cogview4(self, model, prompt, seed, dtype, guidance_scale, batch_size, steps, width, height, cache_model):

        ret_images = []

        from diffusers import CogView4Pipeline

        if self.dtype != dtype or self.model_name != model:
            model_path = check_and_download_model(model, f"THUDM/{model}")
            if dtype == 'bf16':
                self.pipe = CogView4Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                self.model_name = model
                self.dtype = dtype
            else:
                self.pipe = CogView4Pipeline.from_pretrained(model_path, torch_dtype=torch.float32)
                self.model_name = model
                self.dtype = dtype
            # Open it for reduce GPU memory usage
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()

        image = self.pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images
        for i in image:
            ret_images.append(pil2tensor(i))

        if not cache_model:
            self.pipe = None
            self.model_name = ""
            self.dtype = ""
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "CogView4": CogView4_Wrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CogView4": "CogView4 Wrapper"
}