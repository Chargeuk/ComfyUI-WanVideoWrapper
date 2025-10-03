import os
import sys
import random
import torch
import gc
import importlib.util
import comfy
import comfy.utils
import logging
import time
import traceback
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, NamedTuple

try:
    # Get the absolute path to the ComfyUI-vts-nodes folder
    comfyui_vts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ComfyUI-vts-nodes', 'py'))

    # Add the ComfyUI-vts-nodes folder to sys.path
    if comfyui_vts_path not in sys.path:
        sys.path.append(comfyui_vts_path)

    print(f"!!!!!!!!!!!!!ComfyUI-vts-nodes path: {comfyui_vts_path}")
    print(sys.path)

    # Dynamically load the nodes module from ComfyUI-vts-nodes
    vtstaevid_module_path = os.path.join(comfyui_vts_path, 'VTS_TaeVid.py')
    spec = importlib.util.spec_from_file_location("ComfyUI_vts_nodes.vtstaevid", vtstaevid_module_path)
    nodes_module = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_vts_nodes.vtstaevid"] = nodes_module
    spec.loader.exec_module(nodes_module)
    # Import the VTS_TAEVideoDecode class from the dynamically loaded module
    VTS_TAEVideoNodeBase = nodes_module.VTS_TAEVideoNodeBase
    print(f"Loaded VTS_TAEVideoNodeBase from: {comfyui_vts_path}")
    VTS_TAEVideoDecode = nodes_module.VTS_TAEVideoDecode
    print(f"Loaded VTS_TAEVideoDecode from: {comfyui_vts_path}")
    VTS_TAEVideoEncode = nodes_module.VTS_TAEVideoEncode
    print(f"Loaded VTS_TAEVideoEncode from: {comfyui_vts_path}")

    vtstaevid_module_path = os.path.join(comfyui_vts_path, 'VTS_ConvertLatents.py')
    spec = importlib.util.spec_from_file_location("ComfyUI_vts_nodes.vtsconvertlatents", vtstaevid_module_path)
    nodes_module = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_vts_nodes.vtsconvertlatents"] = nodes_module
    spec.loader.exec_module(nodes_module)
    # Import the VTS_TAEVideoDecode class from the dynamically loaded module
    VTS_ConvertLatents = nodes_module.VTS_ConvertLatents
    print(f"Loaded VTS_ConvertLatents from: {comfyui_vts_path}")
except Exception as e:
    print(f"Error loading ComfyUI-vts-nodes VTS_*: {e}")
    VTS_TAEVideoNodeBase = None
    VTS_TAEVideoDecode = None
    VTS_TAEVideoEncode = None
    VTS_ConvertLatents = None

class VTS_TAEWrapper:
    def __init__(self, kwargs):
        self.initialiseValues = kwargs

    def decode(self, latent: dict):
        if VTS_ConvertLatents is None:
            raise RuntimeError("VTS_ConvertLatents is not available.")
        
        if VTS_TAEVideoDecode is None:
            raise RuntimeError("VTS_TAEVideoDecode is not available.")
        
        converter = VTS_ConvertLatents()
        result = converter.convert_latents(source_latent=latent,
                                                     conversion_direction="WanWrapper->ComfyUI",
                                                     conversion_method="channel_wise")
        converted_latent = result[0]
        decoder = VTS_TAEVideoDecode()
        # Pass the latent first, then unpack the initialization values as keyword arguments
        result = decoder.go(latent=converted_latent, **self.initialiseValues)
        decoded_images = result[0]  # Assuming the first element is the decoded images
        return decoded_images
    
class VTS_TAEVLoader(VTS_TAEVideoNodeBase):
    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model from 'ComfyUI/models/vae_approx'"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        result = super().INPUT_TYPES()
        return result
    
    @classmethod
    def go(cls, **kwargs) -> tuple:

        vae = VTS_TAEWrapper(kwargs)
        return (vae,)
    
NODE_CLASS_MAPPINGS = {
    "VTS_TAEVLoader": VTS_TAEVLoader,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS_TAEVLoader": "VTS TAE Loader",
    }