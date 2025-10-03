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

try:
    # Get the absolute path to the ComfyUI-vts-nodes folder
    comfyui_vts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ComfyUI-vts-nodes', 'py'))

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
    VTS_TAEVideoDecode = nodes_module.VTS_TAEVideoDecode
    print(f"Loaded VTS_TAEVideoDecode from: {comfyui_vts_path}")
except Exception as e:
    print(f"Error loading ComfyUI-vts-nodes VTS_TAEVideoDecode: {e}")
    VTS_TAEVideoDecode = None

class VTS_TAEHV(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_decode = VTS_TAEVideoDecode()

    def forward(self, x):
        return self.video_decode(x)