import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

import sys
import os

custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "custom_nodes", "ComfyUI-WanVideoWrapper")
sys.path.insert(0, custom_nodes_path)

from wanvideo.wan_video_vae import WanVideoVAE, WanVideoVAE38

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


class WanVideoVAELoaderFP8:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16", "fp8_e4m3fn", "fp8_e5m2"],
                    {"default": "bf16"}
                ),
                "compile_args": ("WANCOMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "FastVAE"
    DESCRIPTION = "Loads Wan VAE model with FP8 support"

    def loadmodel(self, model_name, precision, compile_args=None):
        dtype_map = {
            "bf16": torch.bfloat16, 
            "fp16": torch.float16, 
            "fp32": torch.float32,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2
        }
        
        is_fp8 = precision in ["fp8_e4m3fn", "fp8_e5m2"]
        base_dtype = torch.bfloat16 if is_fp8 else dtype_map[precision]
        target_dtype = dtype_map[precision]
        
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        has_model_prefix = any(k.startswith("model.") for k in vae_sd.keys())
        if not has_model_prefix:
            vae_sd = {f"model.{k}": v for k, v in vae_sd.items()}

        is_int4 = any(k.endswith(".int4") for k in vae_sd.keys())

        if is_int4:
            print(f"Detected INT4 quantized VAE model: {model_name}")
            print("INT4 format is not supported")
            print("Please use BF16, FP16, FP32, FP8, or FP8_scaled format instead")
            raise ValueError(f"INT4 quantized VAE models are not supported. Model: {model_name}")

        scale_weight_keys = {}
        if is_fp8:
            for key in list(vae_sd.keys()):
                if key.endswith(".scale_weight"):
                    scale_weight_keys[key] = vae_sd[key]

        if vae_sd["model.conv2.weight"].shape[0] == 16:
            vae = WanVideoVAE(dtype=base_dtype)
        elif vae_sd["model.conv2.weight"].shape[0] == 48:
            vae = WanVideoVAE38(dtype=base_dtype)

        if is_fp8:
            for key in list(vae_sd.keys()):
                if not key.endswith(".scale_weight"):
                    if vae_sd[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        vae_sd[key] = vae_sd[key].to(target_dtype)

        vae.load_state_dict(vae_sd, strict=False)
        
        if is_fp8 and len(scale_weight_keys) > 0:
            sys.path.insert(0, custom_nodes_path)
            from fp8_optimization import convert_fp8_linear
            convert_fp8_linear(vae.model, base_dtype, scale_weight_keys=scale_weight_keys)
            print(f"FP8 VAE loaded with {len(scale_weight_keys)} scale weights")
        
        del vae_sd
        vae.eval()
        vae.to(device=offload_device, dtype=base_dtype)
        
        if compile_args is not None:
            vae.model.decoder = torch.compile(vae.model.decoder, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        return (vae,)


NODE_CLASS_MAPPINGS = {
    "WanVideoVAELoaderFP8": WanVideoVAELoaderFP8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoVAELoaderFP8": "WanVideo VAE Loader (FP8)",
}

