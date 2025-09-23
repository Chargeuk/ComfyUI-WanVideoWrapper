import os, gc, math
import torch
import random
import torch.nn.functional as F
import gc
from .utils import log, print_memory, apply_lora, clip_encode_image_tiled, fourier_filter
import numpy as np
import math
from tqdm import tqdm
from typing import Literal, List
import glob
import inspect
import copy
from PIL import Image
import hashlib
import time
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .wanvideo.modules.model import rope_params
from .custom_linear import remove_lora_from_module, set_lora_params
from .wanvideo.schedulers import get_scheduler, get_sampling_sigmas, retrieve_timesteps, scheduler_list
from .gguf.gguf import set_lora_params_gguf
from .multitalk.multitalk import timestep_transform, add_noise
from .utils import(log, print_memory, apply_lora, clip_encode_image_tiled, fourier_filter, 
                   add_noise_to_reference_video, optimized_scale, setup_radial_attention, 
                   compile_model, dict_to_device, tangential_projection, set_module_tensor_to_device, get_raag_guidance)
from .cache_methods.cache_methods import cache_report
from .nodes_model_loading import load_weights
from .enhance_a_video.globals import set_enhance_weight, set_num_frames
from .taehv import TAEHV
from .nodes_utility import WanVideoVACEStartToEndFrame, colormatch
from contextlib import nullcontext
from einops import rearrange

from comfy import model_management as mm
from comfy.utils import ProgressBar, common_upscale, load_torch_file
from comfy.clip_vision import clip_preprocess, ClipVisionModel
from comfy.cli_args import args, LatentPreviewMethod
import folder_paths
import importlib.util
import sys

try:
    # Get the absolute path to the ComfyUI-ReActor folder
    comfyui_reactor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ComfyUI-ReActor'))

    # Add the ComfyUI-ReActor folder to sys.path
    if comfyui_reactor_path not in sys.path:
        sys.path.append(comfyui_reactor_path)

    print(f"!!!!!!!!!!!!!ComfyUI-ReActor path: {comfyui_reactor_path}")
    print(sys.path)

    # Dynamically load the nodes module from ComfyUI-ReActor
    nodes_module_path = os.path.join(comfyui_reactor_path, 'nodes.py')
    spec = importlib.util.spec_from_file_location("ComfyUI_ReActor.nodes", nodes_module_path)
    nodes_module = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_ReActor.nodes"] = nodes_module
    spec.loader.exec_module(nodes_module)

    # Import the reactor class from the dynamically loaded module
    reactor = nodes_module.reactor
    print(f"Loaded reactor from: {comfyui_reactor_path}")
except Exception as e:
    print(f"Error loading ComfyUI-ReActor nodes: {e}")
    reactor = None

script_directory = os.path.dirname(os.path.abspath(__file__))

def get_filename_list(folder_name: str) -> list[str]:
    result = folder_paths.get_filename_list(folder_name)
    # convert to linux paths rather than windows paths
    result = [f.replace("\\", "/") for f in result]
    return result

def get_full_path(folder_name: str, filename: str) -> str | None:
    # convert the folder_name, which is a path to a path using this operating system's path seperator as it may be wrong
    folder_name_for_this_os = folder_name.replace("\\", os.sep).replace("/", os.sep)
    filename_for_this_os = filename.replace("\\", os.sep).replace("/", os.sep)
    return folder_paths.get_full_path(folder_name_for_this_os, filename_for_this_os)

def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    result = get_full_path(folder_name, filename)
    if result is None:
        raise FileNotFoundError(f"Model in folder '{folder_name}' with filename '{filename}' not found.")
    return result

def parse_prompt_input(prompt):
    original_prompt = prompt
    if isinstance(prompt, list):
        return prompt
    elif isinstance(prompt, str):
        # Try to parse as a string representation of a list
        prompt = prompt.strip()
        if prompt.startswith('[') and prompt.endswith(']'):
            try:
                import ast
                return ast.literal_eval(prompt)
            except (ValueError, SyntaxError):
                # If parsing fails, treat as a single string
                return [original_prompt]
        else:
            return [original_prompt]
    else:
        return [str(original_prompt)]

models_dir = folder_paths.models_dir

def get_restorers():
    models_path = os.path.join(models_dir, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    # if len(models) == 0:
    #     fr_urls = [
    #         "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth",
    #         "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
    #         "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth",
    #         "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx",
    #     ]
    #     for model_url in fr_urls:
    #         model_name = os.path.basename(model_url)
    #         model_path = os.path.join(dir_facerestore_models, model_name)
    #         download(model_url, model_path, model_name)
    #     models = glob.glob(models_path)
    #     models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    return models

def get_model_names(get_models):
    models = get_models()
    names = []
    for x in models:
        names.append(os.path.basename(x))
    names.sort(key=str.lower)
    names.insert(0, "none")
    return names


def trim_memory():
        try:
            import ctypes
            import os
            if os.name == 'posix':  # For Linux/Unix systems
                libc = ctypes.CDLL("libc.so.6")
                return libc.malloc_trim(0)
            elif os.name == 'nt':  # For Windows systems
                kernel32 = ctypes.WinDLL("kernel32.dll")
                process = kernel32.GetCurrentProcess()
                result = kernel32.SetProcessWorkingSetSize(process, -1, -1)
                return result != 0  # Returns True if successful
        except Exception as e:
            # Fallback for systems that may not support these methods
            print(f"VTS_WAN_Clear_Ram: Trim memory failure: {e}")
            return False

def ClearMemory(
    gc_collect,
    empty_cache,
    unload_all_models
    ):
    if empty_cache:
        mm.soft_empty_cache()
        # torch.device('cpu').empty_cache()
    if unload_all_models:
        mm.unload_all_models()
        mm.free_memory(1e30, mm.unet_offload_device())
        mm.free_memory(1e30, torch.device('cpu'))
    if gc_collect:
        for _ in range(3):  # Run garbage collection multiple times
            trim_memory()
            gc.collect()
        
        trim_memory()

        # Handle CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # More aggressive memory cleanup
            # Force a synchronization point
            torch.cuda.synchronize()
            
            # Try to defragment memory
            if hasattr(torch.cuda, 'caching_allocator_delete_caches'):
                torch.cuda.caching_allocator_delete_caches()

class WanFaceRestoreArgs_VTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "model": (get_model_names(get_restorers),),
                "visibility": ("FLOAT", {"default": 1, "min": 0.0, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "max_batch_size": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "Maximum number of images to process in a single batch"}),
                "useFp16_Onnx": ("BOOLEAN", {"default": False, "tooltip": "When true, compile a provided onnx file as fp16, otherwise use fp32"}),
            },
        }
    RETURN_TYPES = ("RESTOREFACEARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Settings for reactor while looping"

    def setargs(self, **kwargs):
        return (kwargs, )
    
class WanColorMatchArgs_VTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "used_source": (
                [   
                    'disable-match',
                    'provided',
                    'first-frame',
                    'provided-frame', 
                ], {
                    "default": 'first-frame'
                }),
            },
            "optional": {
                "source": ("IMAGE",),
                "provided_frame":  ("INT", {"default": 0, "min": 0, "step": 1}),
                "color_match_method": (
                [   
                    'mkl',
                    'hm', 
                    'reinhard', 
                    'mvgd', 
                    'hm-mvgd-hm', 
                    'hm-mkl-hm',
                ], {
                "default": 'hm-mvgd-hm'
                }),  
                "color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),          
            },
        }
    RETURN_TYPES = ("COLOURMATCHARGS",)
    RETURN_NAMES = ("color_match_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Settings for color matching while looping"

    def setargs(self, **kwargs):
        return (kwargs, )

#region VideoDecode
class WanVideoDecodeSettings_VTS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": (
                        "Drastically reduces memory use but will introduce seams at tile stride boundaries. "
                        "The location and number of seams is dictated by the tile stride size. "
                        "The visibility of seams can be controlled by increasing the tile size. "
                        "Seams become less obvious at 1.5x stride and are barely noticeable at 2x stride size. "
                        "Which is to say if you use a stride width of 160, the seams are barely noticeable with a tile width of 320."
                    )}),
                    "tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile width in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile height in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride width in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride height in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    },
                }

    RETURN_TYPES = ("WANDECODEARGS",)
    RETURN_NAMES = ("decode_latent_args",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper"

    def decode(self, **kwargs):
        return (kwargs, )

#region VideoEncode
class WanVideoEncodeSettings_VTS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for leapfusion I2V where some noise can add motion and give sharper results"}),
                    "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for leapfusion I2V where lower values allow for more motion"}),
                    }
                }

    RETURN_TYPES = ("WANENCODEARGS",)
    RETURN_NAMES = ("encode_latent_args",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, **kwargs):
        return (kwargs, )

class WanVideoUpscaleWithModelSettings_VTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "device_preference": (["auto", "cuda", "cpu"],),
            },
        }
    RETURN_TYPES = ("WANMODELUPSCALEARGS",)
    RETURN_NAMES = ("model_upscale_args",)

    FUNCTION = "upscale"

    CATEGORY = "WanVideoWrapper"

    def upscale(self, **kwargs):
        return (kwargs, )

class WanVideoSimpleScaleSettings_VTS:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_method": (s.upscale_methods,),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("WANSIMPLESCALEARGS",)
    FUNCTION = "scale"

    CATEGORY = "WanVideoWrapper"

    def scale(self, **kwargs):
        return (kwargs, )
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

try:
    from .gguf.gguf import GGUFParameter
except:
    pass

class MetaParameter(torch.nn.Parameter):
    def __new__(cls, dtype, quant_type=None):
        data = torch.empty(0, dtype=dtype)
        self = torch.nn.Parameter(data, requires_grad=False)
        self.quant_type = quant_type
        return self

def offload_transformer(transformer):
    for block in transformer.blocks:
        block.kv_cache = None
    transformer.teacache_state.clear_all()
    transformer.magcache_state.clear_all()
    transformer.easycache_state.clear_all()
    
    if transformer.patched_linear:
        for name, param in transformer.named_parameters():
            if "loras" in name or "controlnet" in name:
                continue
            module = transformer
            subnames = name.split('.')
            for subname in subnames[:-1]:
                module = getattr(module, subname)
            attr_name = subnames[-1]
            if param.data.is_floating_point():
                meta_param = torch.nn.Parameter(torch.empty_like(param.data, device='meta'), requires_grad=False)
                setattr(module, attr_name, meta_param)
            elif isinstance(param.data, GGUFParameter):
                quant_type = getattr(param, 'quant_type', None)
                setattr(module, attr_name, MetaParameter(param.data.dtype, quant_type))
            else:
                pass
    else:
        transformer.to(offload_device)

    mm.soft_empty_cache()
    gc.collect()

class WanVideoEnhanceAVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 2.0, "min": 0, "max": 100, "step": 0.01, "tooltip": "The feta Weight of the Enhance-A-Video"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply Enhance-A-Video"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply Enhance-A-Video"}),
            },
            "optional": {
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "The node will do nothing and return a None value"}),
            }
        }
    RETURN_TYPES = ("FETAARGS",)
    RETURN_NAMES = ("feta_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"

    def setargs(self, **kwargs):
        # check if passthrough is provided and is true
        if kwargs.get("passthrough", False):
            print("WanVideoEnhanceAVideo: Passthrough enabled, returning None")
            return (None, )
        print("WanVideoEnhanceAVideo: Passthrough disabled, returning kwargs")
        return (kwargs, )

class WanVideoSetBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
               },
            "optional": {
                "block_swap_args": ("BLOCKSWAPARGS", ),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, block_swap_args=None):
        if block_swap_args is None:
            return (model,)
        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args     

        return (patcher,)

class WanVideoSetRadialAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
                "dense_attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "sparse_sage_attention",
                    ], {"default": "sageattn", "tooltip": "The attention mode for dense attention"}),
                "dense_blocks": ("INT",  {"default": 1, "min": 0, "max": 40, "step": 1, "tooltip": "Number of blocks to apply normal attention to"}),
                "dense_vace_blocks": ("INT",  {"default": 1, "min": 0, "max": 15, "step": 1, "tooltip": "Number of vace blocks to apply normal attention to"}),
                "dense_timesteps": ("INT",  {"default": 2, "min": 0, "max": 100, "step": 1, "tooltip": "The step to start applying sparse attention"}),
                "decay_factor": ("FLOAT",  {"default": 0.2, "min": 0, "max": 1, "step": 0.01, "tooltip": "Controls how quickly the attention window shrinks as the distance between frames increases in the sparse attention mask."}),
                "block_size":([128, 64], {"default": 128, "tooltip": "Radial attention block size, larger blocks are faster but restricts usable dimensions more."}),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Sets radial attention parameters, dense attention refers to normal attention"

    def loadmodel(self, model, dense_attention_mode, dense_blocks, dense_vace_blocks, dense_timesteps, decay_factor, block_size):
        if "radial" not in model.model.diffusion_model.attention_mode:
            raise Exception("Enable radial attention first in the model loader.")
            
        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}

        patcher.model_options["transformer_options"]["dense_attention_mode"] = dense_attention_mode
        patcher.model_options["transformer_options"]["dense_blocks"] = dense_blocks
        patcher.model_options["transformer_options"]["dense_vace_blocks"] = dense_vace_blocks
        patcher.model_options["transformer_options"]["dense_timesteps"] = dense_timesteps
        patcher.model_options["transformer_options"]["decay_factor"] = decay_factor
        patcher.model_options["transformer_options"]["block_size"] = block_size

        return (patcher,)

class WanVideoBlockList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blocks": ("STRING",  {"default": "1", "multiline":True}),
               }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("block_list", )
    FUNCTION = "create_list"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Comma separated list of blocks to apply block swap to, can also use ranges like '0-5' or '0,2,3-5' etc., can be connected to the dense_blocks input of 'WanVideoSetRadialAttention' node"

    def create_list(self, blocks):
        block_list = []
        for line in blocks.splitlines():
            for part in line.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    try:
                        start, end = map(int, part.split("-", 1))
                        block_list.extend(range(start, end + 1))
                    except Exception:
                        raise ValueError(f"Invalid range: '{part}'")
                else:
                    try:
                        block_list.append(int(part))
                    except Exception:
                        raise ValueError(f"Invalid integer: '{part}'")
        return (block_list,)



# In-memory cache for prompt extender output
_extender_cache = {}

cache_dir = os.path.join(script_directory, 'text_embed_cache')

def get_cache_path(prompt):
    cache_key = prompt.strip()
    cache_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{cache_hash}.pt")

def get_cached_text_embeds(positive_prompt, negative_prompt):
    
    os.makedirs(cache_dir, exist_ok=True)

    context = None
    context_null = None

    pos_cache_path = get_cache_path(positive_prompt)
    neg_cache_path = get_cache_path(negative_prompt)

    # Try to load positive prompt embeds
    if os.path.exists(pos_cache_path):
        try:
            log.info(f"Loading prompt embeds from cache: {pos_cache_path}")
            context = torch.load(pos_cache_path)
        except Exception as e:
            log.warning(f"Failed to load cache: {e}, will re-encode.")

    # Try to load negative prompt embeds
    if os.path.exists(neg_cache_path):
        try:
            log.info(f"Loading prompt embeds from cache: {neg_cache_path}")
            context_null = torch.load(neg_cache_path)
        except Exception as e:
            log.warning(f"Failed to load cache: {e}, will re-encode.")

    return context, context_null

class WanVideoTextEncodeCached:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
            "precision": (["fp32", "bf16"],
                    {"default": "bf16"}
                ),
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "use_disk_cache": ("BOOLEAN", {"default": True, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
            "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            },
            "optional": {
                "extender_args": ("WANVIDEOPROMPTEXTENDER_ARGS", {"tooltip": "Use this node to extend the prompt with additional text."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("text_embeds", "negative_text_embeds", "positive_prompt")
    OUTPUT_TOOLTIPS = ("The text embeddings for both prompts", "The text embeddings for the negative prompt only (for NAG)", "Positive prompt to display prompt extender results")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = """Encodes text prompts into text embeddings. This node loads and completely unloads the T5 after done,  
leaving no VRAM or RAM imprint. If prompts have been cached before T5 is not loaded at all.  
negative output is meant to be used with NAG, it contains only negative prompt embeddings.  

Additionally you can provide a Qwen LLM model to extend the positive prompt with either one  
of the original Wan templates or a custom system prompt.  
"""


    def process(self, model_name, precision, positive_prompt, negative_prompt, quantization='disabled', use_disk_cache=True, device="gpu", extender_args=None):
        from .nodes_model_loading import LoadWanVideoT5TextEncoder

        pbar = ProgressBar(3)

        echoshot = True if "[1]" in positive_prompt else False

        # Handle prompt extension with in-memory cache
        orig_prompt = positive_prompt
        if extender_args is not None:
            extender_key = (orig_prompt, str(extender_args))
            if extender_key in _extender_cache:
                positive_prompt = _extender_cache[extender_key]
                log.info(f"Loaded extended prompt from in-memory cache: {positive_prompt}")
            else:
                from .qwen.qwen import QwenLoader, WanVideoPromptExtender
                log.info("Using WanVideoPromptExtender to process prompts")
                qwen, = QwenLoader().load(
                    extender_args["model"], 
                    load_device="main_device" if device == "gpu" else "cpu", 
                    precision=precision)
                positive_prompt, = WanVideoPromptExtender().generate(
                    qwen=qwen,
                    max_new_tokens=extender_args["max_new_tokens"],
                    prompt=orig_prompt,
                    device=device,
                    force_offload=False,
                    custom_system_prompt=extender_args["system_prompt"],
                    seed=extender_args["seed"]
                )
                log.info(f"Extended positive prompt: {positive_prompt}")
                _extender_cache[extender_key] = positive_prompt
                del qwen
            pbar.update(1)

        # Now check disk cache using the (possibly extended) prompt
        if use_disk_cache:
            context, context_null = get_cached_text_embeds(positive_prompt, negative_prompt)
            if context is not None and context_null is not None:
                return{
                    "prompt_embeds": context,
                    "negative_prompt_embeds": context_null,
                    "echoshot": echoshot,
                },{"prompt_embeds": context_null}, positive_prompt

        t5, = LoadWanVideoT5TextEncoder().loadmodel(model_name, precision, "main_device", quantization)
        pbar.update(1)

        prompt_embeds_dict, = WanVideoTextEncode().process(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            t5=t5,
            force_offload=False,
            model_to_offload=None,
            use_disk_cache=use_disk_cache,
            device=device
        )
        pbar.update(1)
        del t5
        mm.soft_empty_cache()
        gc.collect() 
        return (prompt_embeds_dict, {"prompt_embeds": prompt_embeds_dict["negative_prompt_embeds"]}, positive_prompt)



class VTS_WanVideoMultiTextEncodeCached:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
            "precision": (["fp32", "bf16"],
                    {"default": "bf16"}
                ),
            "positive_prompt_in": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt_in": ("STRING", {"default": "", "multiline": True} ),
            "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "use_disk_cache": ("BOOLEAN", {"default": True, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
            "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            },
            "optional": {
                "extender_args": ("WANVIDEOPROMPTEXTENDER_ARGS", {"tooltip": "Use this node to extend the prompt with additional text."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("text_embeds", "negative_text_embeds", "positive_prompt")
    OUTPUT_IS_LIST = (
        True,
        True,
        True,
    )
    OUTPUT_TOOLTIPS = ("The text embeddings for both prompts", "The text embeddings for the negative prompt only (for NAG)", "Positive prompt to display prompt extender results")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = """Encodes text prompts into text embeddings. This node loads and completely unloads the T5 after done,  
leaving no VRAM or RAM imprint. If prompts have been cached before T5 is not loaded at all.  
negative output is meant to be used with NAG, it contains only negative prompt embeddings.  

Additionally you can provide a Qwen LLM model to extend the positive prompt with either one  
of the original Wan templates or a custom system prompt.  
"""


    def process(self, model_name, precision, positive_prompt_in, negative_prompt_in, quantization='disabled', use_disk_cache=True, device="gpu", extender_args=None):
        from .nodes_model_loading import LoadWanVideoT5TextEncoder
        # both the positive and negative prompts could be text representations of arrays
        # where each item in the array is a seperate prompt to be processed
        # so we need to handle them accordingly
        positive_prompts = parse_prompt_input(positive_prompt_in)
        negative_prompts = parse_prompt_input(negative_prompt_in)
        number_of_positive_prompts = len(positive_prompts)
        number_of_negative_prompts = len(negative_prompts)
        total_number_of_prompts = max(number_of_positive_prompts, number_of_negative_prompts)
        pbar = ProgressBar(total_number_of_prompts * 3)
        text_embeds_list = []
        negative_text_embeds_list = []
        positive_prompt_list = []
        t5 = None

        # loop through the total_number_of_prompts
        for i in range(total_number_of_prompts):
            if i < number_of_positive_prompts:
                positive_prompt = positive_prompts[i]
            else:
                positive_prompt = positive_prompts[-1]

            if i < number_of_negative_prompts:
                negative_prompt = negative_prompts[i]
            else:
                negative_prompt = negative_prompts[-1]

            echoshot = True if "[1]" in positive_prompt else False

            # Handle prompt extension with in-memory cache
            orig_prompt = positive_prompt
            if extender_args is not None:
                extender_key = (orig_prompt, str(extender_args))
                if extender_key in _extender_cache:
                    positive_prompt = _extender_cache[extender_key]
                    log.info(f"Loaded extended prompt from in-memory cache: {positive_prompt}")
                else:
                    from .qwen.qwen import QwenLoader, WanVideoPromptExtender
                    log.info("Using WanVideoPromptExtender to process prompts")
                    qwen, = QwenLoader().load(
                        extender_args["model"], 
                        load_device="main_device" if device == "gpu" else "cpu", 
                        precision=precision)
                    positive_prompt, = WanVideoPromptExtender().generate(
                        qwen=qwen,
                        max_new_tokens=extender_args["max_new_tokens"],
                        prompt=orig_prompt,
                        device=device,
                        force_offload=False,
                        custom_system_prompt=extender_args["system_prompt"],
                        seed=extender_args["seed"]
                    )
                    log.info(f"Extended positive prompt: {positive_prompt}")
                    _extender_cache[extender_key] = positive_prompt
                    del qwen
                    mm.soft_empty_cache()
                    gc.collect() 
            pbar.update(1)

            # Now check disk cache using the (possibly extended) prompt
            if use_disk_cache:
                context, context_null = get_cached_text_embeds(positive_prompt, negative_prompt)
                if context is not None and context_null is not None:
                    text_embeds_list.append({
                        "prompt_embeds": context,
                        "negative_prompt_embeds": context_null,
                        "echoshot": echoshot,
                    })
                    negative_text_embeds_list.append({
                        "prompt_embeds": context_null,
                    })
                    positive_prompt_list.append(positive_prompt)
                    pbar.update(2)
                    continue

            if t5 is None:
                t5, = LoadWanVideoT5TextEncoder().loadmodel(model_name, precision, "main_device", quantization)
            pbar.update(1)

            prompt_embeds_dict, = WanVideoTextEncode().process(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                t5=t5,
                force_offload=False,
                model_to_offload=None,
                use_disk_cache=use_disk_cache,
                device=device
            )
            text_embeds_list.append(prompt_embeds_dict)
            negative_text_embeds_list.append({"prompt_embeds": prompt_embeds_dict["negative_prompt_embeds"]})
            positive_prompt_list.append(positive_prompt)
            pbar.update(1)

        if t5 is not None:
            del t5
            t5 = None
        mm.soft_empty_cache()
        gc.collect()
        print(f"VTS_WanVideoMultiTextEncodeCached Text embeddings collected: {len(text_embeds_list)}, Negative: {len(negative_text_embeds_list)}, Positive: {len(positive_prompt_list)}")
        return (text_embeds_list, negative_text_embeds_list, positive_prompt_list)

#region TextEncode
class WanVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
                "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts into text embeddings. For rudimentary prompt travel you can input multiple prompts separated by '|', they will be equally spread over the video length"


    def process(self, positive_prompt, negative_prompt, t5=None, force_offload=True, model_to_offload=None, use_disk_cache=False, device="gpu"):
        if t5 is None and not use_disk_cache:
            raise ValueError("T5 encoder is required for text encoding. Please provide a valid T5 encoder or enable disk cache.")

        echoshot = True if "[1]" in positive_prompt else False

        if use_disk_cache:
            context, context_null = get_cached_text_embeds(positive_prompt, negative_prompt)
            if context is not None and context_null is not None:
                return{
                    "prompt_embeds": context,
                    "negative_prompt_embeds": context_null,
                    "echoshot": echoshot,
                },
            
        if t5 is None:
            raise ValueError("No cached text embeds found for prompts, please provide a T5 encoder.")

        if model_to_offload is not None and device == "gpu":
            try:
                log.info(f"Moving video model to {offload_device}")
                model_to_offload.model.to(offload_device)
            except:
                pass

        encoder = t5["model"]
        dtype = t5["dtype"]
        
        positive_prompts = []
        all_weights = []

        # Split positive prompts and process each with weights
        if "|" in positive_prompt:
            log.info("Multiple positive prompts detected, splitting by '|'")
            positive_prompts_raw = [p.strip() for p in positive_prompt.split('|')]
        elif "[1]" in positive_prompt:
            log.info("Multiple positive prompts detected, splitting by [#] and enabling EchoShot")
            import re
            segments = re.split(r'\[\d+\]', positive_prompt)
            positive_prompts_raw = [segment.strip() for segment in segments if segment.strip()]
            assert len(positive_prompts_raw) > 1 and len(positive_prompts_raw) < 7, 'Input shot num must between 2~6 !'
        else:
            positive_prompts_raw = [positive_prompt.strip()]
            
        for p in positive_prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)

        mm.soft_empty_cache()

        if device == "gpu":
            device_to = mm.get_torch_device()
        else:
            device_to = torch.device("cpu")

        if encoder.quantization == "fp8_e4m3fn":
            cast_dtype = torch.float8_e4m3fn
        else:
            cast_dtype = encoder.dtype

        params_to_keep = {'norm', 'pos_embedding', 'token_embedding'}
        for name, param in encoder.model.named_parameters():
            dtype_to_use = dtype if any(keyword in name for keyword in params_to_keep) else cast_dtype
            value = encoder.state_dict[name] if hasattr(encoder, 'state_dict') else encoder.model.state_dict()[name]
            set_module_tensor_to_device(encoder.model, name, device=device_to, dtype=dtype_to_use, value=value)
        if hasattr(encoder, 'state_dict'):
            del encoder.state_dict
            mm.soft_empty_cache()
            gc.collect()

        with torch.autocast(device_type=mm.get_autocast_device(device_to), dtype=encoder.dtype, enabled=encoder.quantization != 'disabled'):
            # Encode positive if not loaded from cache
            if use_disk_cache and context is not None:
                pass
            else:
                context = encoder(positive_prompts, device_to)
                # Apply weights to embeddings if any were extracted
                for i, weights in enumerate(all_weights):
                    for text, weight in weights.items():
                        log.info(f"Applying weight {weight} to prompt: {text}")
                        if len(weights) > 0:
                            context[i] = context[i] * weight

            # Encode negative if not loaded from cache
            if use_disk_cache and context_null is not None:
                pass
            else:
                context_null = encoder([negative_prompt], device_to)

        if force_offload:
            encoder.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        prompt_embeds_dict = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
            "echoshot": echoshot,
        }

        # Save each part to its own cache file if needed
        if use_disk_cache:
            pos_cache_path = get_cache_path(positive_prompt)
            neg_cache_path = get_cache_path(negative_prompt)
            try:
                if not os.path.exists(pos_cache_path):
                    torch.save(context, pos_cache_path)
                    log.info(f"Saved prompt embeds to cache: {pos_cache_path}")
            except Exception as e:
                log.warning(f"Failed to save cache: {e}")
            try:
                if not os.path.exists(neg_cache_path):
                    torch.save(context_null, neg_cache_path)
                    log.info(f"Saved prompt embeds to cache: {neg_cache_path}")
            except Exception as e:
                log.warning(f"Failed to save cache: {e}")

        return (prompt_embeds_dict,)
    
    def parse_prompt_weights(self, prompt):
        """Extract text and weights from prompts with (text:weight) format"""
        import re
        
        # Parse all instances of (text:weight) in the prompt
        pattern = r'\((.*?):([\d\.]+)\)'
        matches = re.findall(pattern, prompt)
        
        # Replace each match with just the text part
        cleaned_prompt = prompt
        weights = {}
        
        for match in matches:
            text, weight = match
            orig_text = f"({text}:{weight})"
            cleaned_prompt = cleaned_prompt.replace(orig_text, text)
            weights[text] = float(weight)
            
        return cleaned_prompt, weights

class WanVideoTextEncodeSingle:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
                "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompt into text embedding."

    def process(self, prompt, t5=None, force_offload=True, model_to_offload=None, use_disk_cache=False, device="gpu"):
        # Unified cache logic: use a single cache file per unique prompt
        encoded = None
        echoshot = True if "[1]" in prompt else False
        if use_disk_cache:
            cache_dir = os.path.join(script_directory, 'text_embed_cache')
            os.makedirs(cache_dir, exist_ok=True)
            def get_cache_path(prompt):
                cache_key = prompt.strip()
                cache_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
                return os.path.join(cache_dir, f"{cache_hash}.pt")
            cache_path = get_cache_path(prompt)
            if os.path.exists(cache_path):
                try:
                    log.info(f"Loading prompt embeds from cache: {cache_path}")
                    encoded = torch.load(cache_path)
                except Exception as e:
                    log.warning(f"Failed to load cache: {e}, will re-encode.")

        if t5 is None and encoded is None:
            raise ValueError("No cached text embeds found for prompts, please provide a T5 encoder.")

        if encoded is None:
            try:
                if model_to_offload is not None and device == "gpu":
                    log.info(f"Moving video model to {offload_device}")
                    model_to_offload.model.to(offload_device)
                    mm.soft_empty_cache()
            except:
                pass

            encoder = t5["model"]
            dtype = t5["dtype"]

            if device == "gpu":
                device_to = mm.get_torch_device()
            else:
                device_to = torch.device("cpu")

            if encoder.quantization == "fp8_e4m3fn":
                cast_dtype = torch.float8_e4m3fn
            else:
                cast_dtype = encoder.dtype
            params_to_keep = {'norm', 'pos_embedding', 'token_embedding'}
            for name, param in encoder.model.named_parameters():
                dtype_to_use = dtype if any(keyword in name for keyword in params_to_keep) else cast_dtype
                value = encoder.state_dict[name] if hasattr(encoder, 'state_dict') else encoder.model.state_dict()[name]
                set_module_tensor_to_device(encoder.model, name, device=device_to, dtype=dtype_to_use, value=value)
            if hasattr(encoder, 'state_dict'):
                del encoder.state_dict
                mm.soft_empty_cache()
                gc.collect()
            with torch.autocast(device_type=mm.get_autocast_device(device_to), dtype=encoder.dtype, enabled=encoder.quantization != 'disabled'):
                encoded = encoder([prompt], device_to)

            if force_offload:
                encoder.model.to(offload_device)
                mm.soft_empty_cache()

            # Save to cache if enabled
            if use_disk_cache:
                try:
                    if not os.path.exists(cache_path):
                        torch.save(encoded, cache_path)
                        log.info(f"Saved prompt embeds to cache: {cache_path}")
                except Exception as e:
                    log.warning(f"Failed to save cache: {e}")

        prompt_embeds_dict = {
            "prompt_embeds": encoded,
            "negative_prompt_embeds": None,
            "echoshot": echoshot
        }
        return (prompt_embeds_dict,)
    
class WanVideoApplyNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "original_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "nag_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Adds NAG prompt embeds to original prompt embeds: 'https://github.com/ChenDarYen/Normalized-Attention-Guidance'"

    def process(self, original_text_embeds, nag_text_embeds, nag_scale, nag_tau, nag_alpha):
        prompt_embeds_dict_copy = original_text_embeds.copy()
        prompt_embeds_dict_copy.update({
                "nag_prompt_embeds": nag_text_embeds["prompt_embeds"],
                "nag_params": {
                    "nag_scale": nag_scale,
                    "nag_tau": nag_tau,
                    "nag_alpha": nag_alpha,
                }
            })
        return (prompt_embeds_dict_copy,)

class VTS_WanVideoApplyMultiNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "original_text_embeds_in": ("WANVIDEOTEXTEMBEDS",),
            "nag_text_embeds_in": ("WANVIDEOTEXTEMBEDS",),
            "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    # OUTPUT_IS_LIST = (
    #     True, 
    # )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Adds NAG prompt embeds to original prompt embeds: 'https://github.com/ChenDarYen/Normalized-Attention-Guidance'"

    def process(self, original_text_embeds_in, nag_text_embeds_in, nag_scale, nag_tau, nag_alpha):
        # only the embeds should be processed as lists
        # for everyting else we use the 1st item
        nag_scale = nag_scale[0]
        nag_tau = nag_tau[0]
        nag_alpha = nag_alpha[0]
        text_embeds_list = []
        number_of_original_text_embeds_in = len(original_text_embeds_in)
        number_of_nag_text_embeds_in = len(nag_text_embeds_in)
        maximum_number_of_embeds = max(number_of_original_text_embeds_in, number_of_nag_text_embeds_in)

        for i in range(maximum_number_of_embeds):
            original_text_embeds = original_text_embeds_in[i] if i < number_of_original_text_embeds_in else original_text_embeds_in[-1]
            nag_text_embeds = nag_text_embeds_in[i] if i < number_of_nag_text_embeds_in else nag_text_embeds_in[-1]

            prompt_embeds_dict_copy = original_text_embeds.copy()
            prompt_embeds_dict_copy.update({
                    "nag_prompt_embeds": nag_text_embeds["prompt_embeds"],
                    "nag_params": {
                        "nag_scale": nag_scale,
                        "nag_tau": nag_tau,
                        "nag_alpha": nag_alpha,
                    }
                })
            text_embeds_list.append(prompt_embeds_dict_copy)

        text_embeds_result = {
            "charge_uk_embeds" : text_embeds_list
        }

        return (text_embeds_result,)

class WanVideoTextEmbedReverseBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "wan_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            },
            "optional": {
                "conditioning_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength multiplier for the conditioning"}),
                "output_negative": ("BOOLEAN", {"default": True, "tooltip": "Whether to output negative conditioning as well"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive_conditioning", "negative_conditioning")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Reverse bridge from WanVideoWrapper text embeddings back to ComfyUI native CONDITIONING format"

    def process(self, wan_text_embeds, conditioning_strength=1.0, output_negative=True):
        print("=== WanVideoTextEmbedReverseBridge DEBUG ===")
        print(f"Input wan_text_embeds type: {type(wan_text_embeds)}")
        print(f"Input wan_text_embeds keys: {wan_text_embeds.keys() if isinstance(wan_text_embeds, dict) else 'N/A'}")
        
        # Extract embeddings from WanVideoWrapper format
        positive_embeds_raw = wan_text_embeds["prompt_embeds"]
        print(f"positive_embeds_raw type: {type(positive_embeds_raw)}")
        
        if isinstance(positive_embeds_raw, list):
            print(f"positive_embeds_raw is a list with {len(positive_embeds_raw)} items")
            for i, item in enumerate(positive_embeds_raw):
                print(f"  Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'no shape')}")
            positive_embeds = positive_embeds_raw[0]  # Take first item if it's a list
        else:
            print(f"positive_embeds_raw shape: {getattr(positive_embeds_raw, 'shape', 'no shape')}")
            positive_embeds = positive_embeds_raw
        
        print(f"Final positive_embeds type: {type(positive_embeds)}")
        print(f"Final positive_embeds shape: {getattr(positive_embeds, 'shape', 'no shape')}")
        print(f"Final positive_embeds dtype: {getattr(positive_embeds, 'dtype', 'no dtype')}")
        
        # Check if it's a torch tensor
        if not isinstance(positive_embeds, torch.Tensor):
            print(f"WARNING: positive_embeds is not a torch.Tensor, it's {type(positive_embeds)}")
            if hasattr(positive_embeds, '__iter__') and not isinstance(positive_embeds, (str, bytes)):
                try:
                    positive_embeds = torch.stack(list(positive_embeds))
                    print(f"Converted to tensor with shape: {positive_embeds.shape}")
                except Exception as e:
                    print(f"Failed to convert to tensor: {e}")
                    return None
        
        # Ensure positive_embeds has batch dimension (3D tensor)
        if positive_embeds.dim() == 2:
            positive_embeds = positive_embeds.unsqueeze(0)  # Add batch dimension [seq_len, dim] -> [1, seq_len, dim]
            print(f"Added batch dimension to positive_embeds: {positive_embeds.shape}")
        
        negative_embeds_raw = wan_text_embeds.get("negative_prompt_embeds", None)
        print(f"negative_embeds_raw type: {type(negative_embeds_raw)}")
        
        if negative_embeds_raw is not None:
            if isinstance(negative_embeds_raw, list):
                print(f"negative_embeds_raw is a list with {len(negative_embeds_raw)} items")
                negative_embeds = negative_embeds_raw[0]  # Take first item if it's a list
            else:
                print(f"negative_embeds_raw shape: {getattr(negative_embeds_raw, 'shape', 'no shape')}")
                negative_embeds = negative_embeds_raw
                
            print(f"Final negative_embeds type: {type(negative_embeds)}")
            print(f"Final negative_embeds shape: {getattr(negative_embeds, 'shape', 'no shape')}")
            
            # Check if it's a torch tensor
            if not isinstance(negative_embeds, torch.Tensor):
                print(f"WARNING: negative_embeds is not a torch.Tensor, it's {type(negative_embeds)}")
                if hasattr(negative_embeds, '__iter__') and not isinstance(negative_embeds, (str, bytes)):
                    try:
                        negative_embeds = torch.stack(list(negative_embeds))
                        print(f"Converted negative to tensor with shape: {negative_embeds.shape}")
                    except Exception as e:
                        print(f"Failed to convert negative to tensor: {e}")
                        negative_embeds = None
            
            # Ensure negative_embeds has batch dimension (3D tensor)
            if negative_embeds is not None and negative_embeds.dim() == 2:
                negative_embeds = negative_embeds.unsqueeze(0)  # Add batch dimension
                print(f"Added batch dimension to negative_embeds: {negative_embeds.shape}")
        else:
            negative_embeds = None
        
        # Apply strength multiplier if specified
        if conditioning_strength != 1.0:
            print(f"Applying conditioning strength: {conditioning_strength}")
            positive_embeds = positive_embeds * conditioning_strength
            if negative_embeds is not None:
                negative_embeds = negative_embeds * conditioning_strength
        
        # Convert to ComfyUI CONDITIONING format
        # ComfyUI CONDITIONING format is: [[embedding_tensor, pooled_output_dict]]
        print("Creating ComfyUI CONDITIONING format...")
        positive_conditioning = [[positive_embeds, {}]]
        print(f"positive_conditioning structure: list with {len(positive_conditioning)} items")
        print(f"positive_conditioning[0][0] type: {type(positive_conditioning[0][0])}")
        print(f"positive_conditioning[0][0] shape: {getattr(positive_conditioning[0][0], 'shape', 'no shape')}")
        
        if output_negative and negative_embeds is not None:
            negative_conditioning = [[negative_embeds, {}]]
            print(f"negative_conditioning[0][0] type: {type(negative_conditioning[0][0])}")
            print(f"negative_conditioning[0][0] shape: {getattr(negative_conditioning[0][0], 'shape', 'no shape')}")
        else:
            # Create empty negative conditioning if not available
            negative_conditioning = [[torch.zeros_like(positive_embeds), {}]]
            print("Created empty negative conditioning")
        
        print("=== END WanVideoTextEmbedReverseBridge DEBUG ===")
        return (positive_conditioning, negative_conditioning)
class WanVideoTextEmbedBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("CONDITIONING",),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Bridge between ComfyUI native text embedding and WanVideoWrapper text embedding"

    def process(self, positive, negative=None):
        print("=== WanVideoTextEmbedBridge DEBUG ===")
        print(f"Input positive type: {type(positive)}")
        print(f"Input positive length: {len(positive) if hasattr(positive, '__len__') else 'no length'}")
        
        if hasattr(positive, '__len__') and len(positive) > 0:
            print(f"positive[0] type: {type(positive[0])}")
            print(f"positive[0] length: {len(positive[0]) if hasattr(positive[0], '__len__') else 'no length'}")
            
            if hasattr(positive[0], '__len__') and len(positive[0]) > 0:
                print(f"positive[0][0] type: {type(positive[0][0])}")
                print(f"positive[0][0] shape: {getattr(positive[0][0], 'shape', 'no shape')}")
                print(f"positive[0][0] dtype: {getattr(positive[0][0], 'dtype', 'no dtype')}")
        
        if negative is not None:
            print(f"Input negative type: {type(negative)}")
            print(f"Input negative length: {len(negative) if hasattr(negative, '__len__') else 'no length'}")
            
            if hasattr(negative, '__len__') and len(negative) > 0:
                print(f"negative[0] type: {type(negative[0])}")
                print(f"negative[0] length: {len(negative[0]) if hasattr(negative[0], '__len__') else 'no length'}")
                
                if hasattr(negative[0], '__len__') and len(negative[0]) > 0:
                    print(f"negative[0][0] type: {type(negative[0][0])}")
                    print(f"negative[0][0] shape: {getattr(negative[0][0], 'shape', 'no shape')}")
                    print(f"negative[0][0] dtype: {getattr(negative[0][0], 'dtype', 'no dtype')}")
        else:
            print("negative is None")
        
        prompt_embeds_dict = {
                "prompt_embeds": positive[0][0].to(device),
                "negative_prompt_embeds": negative[0][0].to(device) if negative is not None else None,
            }
        
        print(f"Output prompt_embeds_dict keys: {prompt_embeds_dict.keys()}")
        print(f"Output prompt_embeds type: {type(prompt_embeds_dict['prompt_embeds'])}")
        print(f"Output prompt_embeds shape: {getattr(prompt_embeds_dict['prompt_embeds'], 'shape', 'no shape')}")
        
        if prompt_embeds_dict['negative_prompt_embeds'] is not None:
            print(f"Output negative_prompt_embeds type: {type(prompt_embeds_dict['negative_prompt_embeds'])}")
            print(f"Output negative_prompt_embeds shape: {getattr(prompt_embeds_dict['negative_prompt_embeds'], 'shape', 'no shape')}")
        else:
            print("Output negative_prompt_embeds is None")
        
        print("=== END WanVideoTextEmbedBridge DEBUG ===")
        return (prompt_embeds_dict,)


#region clip vision
class WanVideoClipVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "image_1": ("IMAGE", {"tooltip": "Image to encode"}),
            "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}), 
            "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}),
            "crop": (["center", "disabled"], {"default": "center", "tooltip": "Crop image to 224x224 before encoding"}),
            "combine_embeds": (["average", "sum", "concat", "batch"], {"default": "average", "tooltip": "Method to combine multiple clip embeds"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_2": ("IMAGE", ),
                "negative_image": ("IMAGE", {"tooltip": "image to use for uncond"}),
                "tiles": ("INT", {"default": 0, "min": 0, "max": 16, "step": 2, "tooltip": "Use matteo's tiled image encoding for improved accuracy"}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ratio of the tile average"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_CLIPEMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, clip_vision, image_1, strength_1, strength_2, force_offload, crop, combine_embeds, image_2=None, negative_image=None, tiles=0, ratio=1.0):
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

        if image_2 is not None:
            image = torch.cat([image_1, image_2], dim=0)
        else:
            image = image_1

        clip_vision.model.to(device)
        
        negative_clip_embeds = None

        if tiles > 0:
            log.info("Using tiled image encoding")
            clip_embeds = clip_encode_image_tiled(clip_vision, image.to(device), tiles=tiles, ratio=ratio)
            if negative_image is not None:
                negative_clip_embeds = clip_encode_image_tiled(clip_vision, negative_image.to(device), tiles=tiles, ratio=ratio)
        else:
            if isinstance(clip_vision, ClipVisionModel):
                clip_embeds = clip_vision.encode_image(image).penultimate_hidden_states.to(device)
                if negative_image is not None:
                    negative_clip_embeds = clip_vision.encode_image(negative_image).penultimate_hidden_states.to(device)
            else:
                pixel_values = clip_preprocess(image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                clip_embeds = clip_vision.visual(pixel_values)
                if negative_image is not None:
                    pixel_values = clip_preprocess(negative_image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                    negative_clip_embeds = clip_vision.visual(pixel_values)
    
        log.info(f"Clip embeds shape: {clip_embeds.shape}, dtype: {clip_embeds.dtype}")

        weighted_embeds = []
        weighted_embeds.append(clip_embeds[0:1] * strength_1)

        # Handle all additional embeddings
        if clip_embeds.shape[0] > 1:
            weighted_embeds.append(clip_embeds[1:2] * strength_2)
            
            if clip_embeds.shape[0] > 2:
                for i in range(2, clip_embeds.shape[0]):
                    weighted_embeds.append(clip_embeds[i:i+1])  # Add as-is without strength modifier
            
            # Combine all weighted embeddings
            if combine_embeds == "average":
                clip_embeds = torch.mean(torch.stack(weighted_embeds), dim=0)
            elif combine_embeds == "sum":
                clip_embeds = torch.sum(torch.stack(weighted_embeds), dim=0)
            elif combine_embeds == "concat":
                clip_embeds = torch.cat(weighted_embeds, dim=1)
            elif combine_embeds == "batch":
                clip_embeds = torch.cat(weighted_embeds, dim=0)
        else:
            clip_embeds = weighted_embeds[0]
                

        log.info(f"Combined clip embeds shape: {clip_embeds.shape}")
        
        if force_offload:
            clip_vision.model.to(offload_device)
            mm.soft_empty_cache()

        clip_embeds_dict = {
            "clip_embeds": clip_embeds,
            "negative_clip_embeds": negative_clip_embeds
        }

        return (clip_embeds_dict,)
        
class WanVideoRealisDanceLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ref_latent": ("LATENT", {"tooltip": "Reference image to encode"}),
            "pose_cond_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the SMPL model"}),
            "pose_cond_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the SMPL model"}),
            },
            "optional": {
                "smpl_latent": ("LATENT", {"tooltip": "SMPL pose image to encode"}),
                "hamer_latent": ("LATENT", {"tooltip": "Hamer hand pose image to encode"}),
            },
        }

    RETURN_TYPES = ("ADD_COND_LATENTS",)
    RETURN_NAMES = ("add_cond_latents",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, ref_latent, pose_cond_start_percent, pose_cond_end_percent, hamer_latent=None, smpl_latent=None):
        if smpl_latent is None and hamer_latent is None:
            raise Exception("At least one of smpl_latent or hamer_latent must be provided")
        if smpl_latent is None:
            smpl = torch.zeros_like(hamer_latent["samples"])
        else:
            smpl = smpl_latent["samples"]
        if hamer_latent is None:
            hamer = torch.zeros_like(smpl_latent["samples"])
        else:
            hamer = hamer_latent["samples"]

        pose_latent = torch.cat((smpl, hamer), dim=1)
        
        add_cond_latents = {
            "ref_latent": ref_latent["samples"],
            "pose_latent": pose_latent,
            "pose_cond_start_percent": pose_cond_start_percent,
            "pose_cond_end_percent": pose_cond_end_percent,
        }

        return (add_cond_latents,)

    
class WanVideoAddStandInLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "ip_image_latent": ("LATENT", {"tooltip": "Reference image to encode"}),
                    "freq_offset": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "EXPERIMENTAL: RoPE frequency offset between the reference and rest of the sequence"}),
                    #"start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent to apply the ref "}),
                    #"end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent to apply the ref "}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, ip_image_latent, freq_offset):
        # Prepare the new extra latent entry
        new_entry = {
            "ip_image_latent": ip_image_latent["samples"],
            "freq_offset": freq_offset,
            #"ip_start_percent": start_percent,
            #"ip_end_percent": end_percent,
        }    

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["standin_input"] = new_entry
        return (updated,)

class WanVideoAddMTVMotion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "mtv_crafter_motion": ("MTVCRAFTERMOTION",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the MTV motion"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent to apply the ref "}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent to apply the ref "}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, mtv_crafter_motion, strength, start_percent, end_percent):
        # Prepare the new extra latent entry
        new_entry = {
            "mtv_motion_tokens": mtv_crafter_motion["mtv_motion_tokens"],
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "global_mean": mtv_crafter_motion["global_mean"],
            "global_std": mtv_crafter_motion["global_std"]
        }

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["mtv_crafter_motion"] = new_entry
        return (updated,)

#region I2V encode
class WanVideoImageToVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("WANVAE",),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "start_image": ("IMAGE", {"tooltip": "Image to encode"}),
                "end_image": ("IMAGE", {"tooltip": "end frame"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Control signal for the Fun -model"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "Enable when using official FLF2V or Fun model"}),
                "temporal_mask": ("MASK", {"tooltip": "mask"}),
                "extra_latents": ("LATENT", {"tooltip": "Extra latents to add to the input front, used for Skyreels A2 reference images"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "add_cond_latents": ("ADD_COND_LATENTS", {"advanced": True, "tooltip": "Additional cond latents WIP"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, width, height, num_frames, force_offload, noise_aug_strength, 
                start_latent_strength, end_latent_strength, start_image=None, end_image=None, control_embeds=None, fun_or_fl2v_model=False, 
                temporal_mask=None, extra_latents=None, clip_embeds=None, tiled_vae=False, add_cond_latents=None, vae=None):
        
        if start_image is None and end_image is None and add_cond_latents is None:
            return WanVideoEmptyEmbeds().process(
                num_frames, width, height, control_embeds=control_embeds, extra_latents=extra_latents,
            )
        if vae is None:
            raise ValueError("VAE is required for image encoding.")
        H = height
        W = width
           
        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_frames = ((num_frames - 1) // 4) * 4 + 1
        two_ref_images = start_image is not None and end_image is not None

        if start_image is None and end_image is not None:
            fun_or_fl2v_model = True # end image alone only works with this option

        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device, dtype=vae.dtype)
            if start_image is not None:
                mask[:, 0:start_image.shape[0]] = 1  # First frame
            if end_image is not None:
                mask[:, -end_image.shape[0]:] = 1  # End frame if exists
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device, vae.dtype)

        # Repeat first frame and optionally end frame
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
        if end_image is not None and not fun_or_fl2v_model:
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1) # T, C, H, W
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

        # Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
        mask = mask.movedim(1, 2)[0]# C, T, H, W

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            start_image = start_image[..., :3]
            if start_image.shape[1] != H or start_image.shape[2] != W:
                resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_start_image = start_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_start_image = resized_start_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_start_image = add_noise_to_reference_video(resized_start_image, ratio=noise_aug_strength)
        
        if end_image is not None:
            end_image = end_image[..., :3]
            if end_image.shape[1] != H or end_image.shape[2] != W:
                resized_end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_end_image = end_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_end_image = resized_end_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_end_image = add_noise_to_reference_video(resized_end_image, ratio=noise_aug_strength)
            
        # Concatenate image with zero frames and encode
        if temporal_mask is None:
            if start_image is not None and end_image is None:
                zero_frames = torch.zeros(3, num_frames-start_image.shape[0], H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames], dim=1)
                del resized_start_image, zero_frames
            elif start_image is None and end_image is not None:
                zero_frames = torch.zeros(3, num_frames-end_image.shape[0], H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
                del zero_frames
            elif start_image is None and end_image is None:
                concatenated = torch.zeros(3, num_frames, H, W, device=device, dtype=vae.dtype)
            else:
                if fun_or_fl2v_model:
                    zero_frames = torch.zeros(3, num_frames-(start_image.shape[0]+end_image.shape[0]), H, W, device=device, dtype=vae.dtype)
                else:
                    zero_frames = torch.zeros(3, num_frames-1, H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
                del resized_start_image, zero_frames
        else:
            temporal_mask = common_upscale(temporal_mask.unsqueeze(1), W, H, "nearest", "disabled").squeeze(1)
            concatenated = resized_start_image[:,:num_frames].to(vae.dtype) * temporal_mask[:num_frames].unsqueeze(0).to(vae.dtype)
            del resized_start_image, temporal_mask

        mm.soft_empty_cache()
        gc.collect()

        vae.to(device)
        y = vae.encode([concatenated], device, end_=(end_image is not None and not fun_or_fl2v_model),tiled=tiled_vae)[0]
        vae.model.clear_cache()
        del concatenated

        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        y[:, :1] *= start_latent_strength
        y[:, -1:] *= end_latent_strength

        # Calculate maximum sequence length
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame

        if add_cond_latents is not None:
            add_cond_latents["ref_latent_neg"] = vae.encode(torch.zeros(1, 3, 1, H, W, device=device, dtype=vae.dtype), device)
        
        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "image_embeds": y,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": resized_end_image if end_image is not None else None,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents,
            "mask": mask
        }

        return (image_embeds,)
    
class WanVideoEmptyEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
            "optional": {
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "control signal for the Fun -model"}),
                "extra_latents": ("LATENT", {"tooltip": "First latent to use for the Pusa -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, control_embeds=None, extra_latents=None):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
        }
        if extra_latents is not None:
            embeds["extra_latents"] = [{
                "samples": extra_latents["samples"],
                "index": 0,
            }]

        return (embeds,)
    
class WanVideoAddExtraLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "extra_latents": ("LATENT",),
                    "latent_index": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1, "tooltip": "Index to insert the extra latents at in latent space"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, extra_latents, latent_index):
        # Prepare the new extra latent entry
        new_entry = {
            "samples": extra_latents["samples"],
            "index": latent_index,
        }
        # Get previous extra_latents list, or start a new one
        prev_extra_latents = embeds.get("extra_latents", None)
        if prev_extra_latents is None:
            extra_latents_list = [new_entry]
        elif isinstance(prev_extra_latents, list):
            extra_latents_list = prev_extra_latents + [new_entry]
        else:
            extra_latents_list = [prev_extra_latents, new_entry]

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["extra_latents"] = extra_latents_list
        return (updated,)

class WanVideoMiniMaxRemoverEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
            "mask_latents": ("LATENT", {"tooltip": "Encoded latents to use as mask"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, latents, mask_latents):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "minimax_latents": latents["samples"].squeeze(0),
            "minimax_mask_latents": mask_latents["samples"].squeeze(0),
        }
    
        return (embeds,)
    
# region phantom
class WanVideoPhantomEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "phantom_latent_1": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
            
            "phantom_cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "CFG scale for the extra phantom cond pass"}),
            "phantom_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the phantom model"}),
            "phantom_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the phantom model"}),
            },
            "optional": {
                "phantom_latent_2": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "phantom_latent_3": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "phantom_latent_4": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "vace_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "VACE embeds"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, phantom_cfg_scale, phantom_start_percent, phantom_end_percent, phantom_latent_1, phantom_latent_2=None, phantom_latent_3=None, phantom_latent_4=None, vace_embeds=None):
        samples = phantom_latent_1["samples"].squeeze(0)
        if phantom_latent_2 is not None:
            samples = torch.cat([samples, phantom_latent_2["samples"].squeeze(0)], dim=1)
        if phantom_latent_3 is not None:
            samples = torch.cat([samples, phantom_latent_3["samples"].squeeze(0)], dim=1)
        if phantom_latent_4 is not None:
            samples = torch.cat([samples, phantom_latent_4["samples"].squeeze(0)], dim=1)
        C, T, H, W = samples.shape

        log.info(f"Phantom latents shape: {samples.shape}")

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        H * 8 // VAE_STRIDE[1],
                        W * 8 // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "phantom_latents": samples,
            "phantom_cfg_scale": phantom_cfg_scale,
            "phantom_start_percent": phantom_start_percent,
            "phantom_end_percent": phantom_end_percent,
        }
        if vace_embeds is not None:
            vace_input = {
                "vace_context": vace_embeds["vace_context"],
                "vace_scale": vace_embeds["vace_scale"],
                "has_ref": vace_embeds["has_ref"],
                "vace_start_percent": vace_embeds["vace_start_percent"],
                "vace_end_percent": vace_embeds["vace_end_percent"],
                "vace_seq_len": vace_embeds["vace_seq_len"],
                "additional_vace_inputs": vace_embeds["additional_vace_inputs"],
                }
            embeds.update(vace_input)
    
        return (embeds,)
    
class WanVideoControlEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
            },
            "optional": {
                "fun_ref_image": ("LATENT", {"tooltip": "Reference latent for the Fun 1.1 -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, latents, start_percent, end_percent, fun_ref_image=None):
        samples = latents["samples"].squeeze(0)
        C, T, H, W = samples.shape

        num_frames = (T - 1) * 4 + 1
        seq_len = math.ceil((H * W) / 4 * ((num_frames - 1) // 4 + 1))
      
        embeds = {
            "max_seq_len": seq_len,
            "target_shape": samples.shape,
            "num_frames": num_frames,
            "control_embeds": {
                "control_images": samples,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "fun_ref_image": fun_ref_image["samples"][:,:, 0] if fun_ref_image is not None else None,
            }
        }
    
        return (embeds,)
    
class WanVideoAddControlEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("WANVIDIMAGE_EMBEDS",),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            },
            "optional": {
                "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
                "fun_ref_image": ("LATENT", {"tooltip": "Reference latent for the Fun 1.1 -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, embeds, start_percent, end_percent, fun_ref_image=None, latents=None):      
        new_entry = {
            "control_images": latents["samples"].squeeze(0) if latents is not None else None,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "fun_ref_image": fun_ref_image["samples"][:,:, 0] if fun_ref_image is not None else None,
        }

        updated = dict(embeds)
        updated["control_embeds"] = new_entry

        return (updated,)
    
class WanVideoAddPusaNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("WANVIDIMAGE_EMBEDS",),
            "noise_multipliers": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Noise multipliers for Pusa, can be a list of floats"}),
            "noisy_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Number steps to apply the extra noise"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Adds latent and timestep noise multipliers when using flowmatch_pusa"

    def add(self, embeds, noise_multipliers, noisy_steps):
        updated = dict(embeds)
        updated["pusa_noise_multipliers"] = noise_multipliers
        updated["pusa_noisy_steps"] = noisy_steps

        return (updated,)
    
class WanVideoSLG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "blocks": ("STRING", {"default": "10", "tooltip": "Blocks to skip uncond on, separated by comma, index starts from 0"}),
            "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            },
            "optional": {
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "The node will do nothing and return a None value"}),
            },
        }

    RETURN_TYPES = ("SLGARGS", )
    RETURN_NAMES = ("slg_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Skips uncond on the selected blocks"

    def process(self, blocks, start_percent, end_percent,
                passthrough=False):
        if passthrough:
            print("WanVideoSLG passthrough enabled, returning None")
            return (None,)
        slg_block_list = [int(x.strip()) for x in blocks.split(",")]

        slg_args = {
            "blocks": slg_block_list,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }
        return (slg_args,)

#region VACE
class WanVideoVACEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply VACE"}),
            "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply VACE"}),
            },
            "optional": {
                "input_frames": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "input_masks": ("MASK",),
                "prev_vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "Pass the prev_vace_embeds through without modification"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, strength, vace_start_percent, vace_end_percent, input_frames=None, ref_images=None, input_masks=None, prev_vace_embeds=None, tiled_vae=False,
                passthrough=False):
        if passthrough or strength <= 0.0:
            return (prev_vace_embeds,)
        
        width = (width // 16) * 16
        height = (height // 16) * 16

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        # vace context encode
        if input_frames is None:
            input_frames = torch.zeros((1, 3, num_frames, height, width), device=device, dtype=vae.dtype)
        else:
            input_frames = input_frames.clone()[:num_frames, :, :, :3]
            input_frames = common_upscale(input_frames.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            input_frames = input_frames.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
            input_frames = input_frames * 2 - 1
        if input_masks is None:
            input_masks = torch.ones_like(input_frames, device=device)
        else:
            log.info(f"input_masks shape: {input_masks.shape}")
            input_masks = input_masks[:num_frames]
            input_masks = common_upscale(input_masks.clone().unsqueeze(1), width, height, "nearest-exact", "disabled").squeeze(1)
            input_masks = input_masks.to(vae.dtype).to(device)
            input_masks = input_masks.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1) # B, C, T, H, W

        if ref_images is not None:
            ref_images = ref_images.clone()[..., :3]
            # Create padded image
            if ref_images.shape[0] > 1:
                ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], dim=1).unsqueeze(0)
            
            B, H, W, C = ref_images.shape
            current_aspect = W / H
            target_aspect = width / height
            if current_aspect > target_aspect:
                # Image is wider than target, pad height
                new_h = int(W / target_aspect)
                pad_h = (new_h - H) // 2
                padded = torch.ones(ref_images.shape[0], new_h, W, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, pad_h:pad_h+H, :, :] = ref_images
                ref_images = padded
            elif current_aspect < target_aspect:
                # Image is taller than target, pad width
                new_w = int(H * target_aspect)
                pad_w = (new_w - W) // 2
                padded = torch.ones(ref_images.shape[0], H, new_w, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, :, pad_w:pad_w+W, :] = ref_images
                ref_images = padded
            ref_images = common_upscale(ref_images.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
            
            ref_images = ref_images.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3).unsqueeze(0)
            ref_images = ref_images * 2 - 1

        vae = vae.to(device)
        z0 = self.vace_encode_frames(vae, input_frames, ref_images, masks=input_masks, tiled_vae=tiled_vae)
        vae.model.clear_cache()
        m0 = self.vace_encode_masks(input_masks, ref_images)
        z = self.vace_latent(z0, m0)
        vae.to(offload_device)

        vace_input = {
            "vace_context": z,
            "vace_scale": strength,
            "has_ref": ref_images is not None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "vace_start_percent": vace_start_percent,
            "vace_end_percent": vace_end_percent,
            "vace_seq_len": math.ceil((z[0].shape[2] * z[0].shape[3]) / 4 * z[0].shape[1]),
            "additional_vace_inputs": [],
        }

        if prev_vace_embeds is not None:
            if "additional_vace_inputs" in prev_vace_embeds and prev_vace_embeds["additional_vace_inputs"]:
                vace_input["additional_vace_inputs"] = prev_vace_embeds["additional_vace_inputs"].copy()
            vace_input["additional_vace_inputs"].append(prev_vace_embeds)
    
        return (vace_input,)
    
    def vace_encode_frames(self, vae, frames, ref_images, masks=None, tiled_vae=False):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        pbar = ProgressBar(len(frames))
        if masks is None:
            latents = vae.encode(frames, device=device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            del frames
            inactive = vae.encode(inactive, device=device, tiled=tiled_vae)
            reactive = vae.encode(reactive, device=device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
            del inactive, reactive
        vae.model.clear_cache()
        
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                else:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
            pbar.update(1)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        pbar = ProgressBar(len(masks))
        for mask, refs in zip(masks, ref_images):
            _c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // VAE_STRIDE[0])
            height = 2 * (int(height) // (VAE_STRIDE[1] * 2))
            width = 2 * (int(width) // (VAE_STRIDE[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                VAE_STRIDE[1] * VAE_STRIDE[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
            pbar.update(1)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]


#region context options
class WanVideoContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context_schedule": (["uniform_standard", "uniform_looped", "static_standard"],),
            "context_frames": ("INT", {"default": 81, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of pixel frames in the context, NOTE: the latent space has 4 frames in 1"} ),
            "context_stride": ("INT", {"default": 4, "min": 4, "max": 100, "step": 1, "tooltip": "Context stride as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "context_overlap": ("INT", {"default": 16, "min": 4, "max": 100, "step": 1, "tooltip": "Context overlap as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "freenoise": ("BOOLEAN", {"default": True, "tooltip": "Shuffle the noise"}),
            "verbose": ("BOOLEAN", {"default": False, "tooltip": "Print debug output"}),
            },
            "optional": {
                "fuse_method": (["linear", "pyramid"], {"default": "linear", "tooltip": "Window weight function: linear=ramps at edges only, pyramid=triangular weights peaking in middle"}),
                "reference_latent": ("LATENT", {"tooltip": "Image to be used as init for I2V models for windows where first frame is not the actual first frame. Mostly useful with MAGREF model"}),
                "rolling_extra": ("BOOLEAN", {"default": False, "tooltip": "If enabled (5B models), feed the previous window's last clean latent (x0) as the first frame of the next window."}),
            }
        }

    RETURN_TYPES = ("WANVIDCONTEXT", )
    RETURN_NAMES = ("context_options",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Context options for WanVideo, allows splitting the video into context windows and attemps blending them for longer generations than the model and memory otherwise would allow."

    def process(self, context_schedule, context_frames, context_stride, context_overlap, freenoise, verbose, image_cond_start_step=6, image_cond_window_count=2, vae=None, fuse_method="linear", reference_latent=None, rolling_extra=False):
        context_options = {
            "context_schedule":context_schedule,
            "context_frames":context_frames,
            "context_stride":context_stride,
            "context_overlap":context_overlap,
            "freenoise":freenoise,
            "verbose":verbose,
            "fuse_method":fuse_method,
            "reference_latent":reference_latent["samples"] if reference_latent is not None else None,
            "rolling_extra": rolling_extra,
        }

        return (context_options,)

class WanVideoLoopingControlImagesOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "control_images": ("IMAGE", {"tooltip": "Control images for the context"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the control signal on the 1st pass"}),
                "strength_later": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the control signal on the later passes"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE for control images"}),
                "use_provided_starting_images": ("BOOLEAN", {"default": True, "tooltip": "Use provided starting images as the first frames"}),
                "use_generated_starting_images": ("BOOLEAN", {"default": True, "tooltip": "Use generated starting images as the first frames"}),
                "custom_name": ("STRING", {"default": "NoName", "multiline": False})
            }
        }

    RETURN_TYPES = ("WANVIDEOCONTROLIMAGESOPTIONS", )
    RETURN_NAMES = ("control_images_options",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Control images for a full video created by a looping Wan Video."

    def process(self, control_images=None, strength=1.0, strength_later=None, start_percent=0.0, end_percent=1.0, tiled_vae=False, use_provided_starting_images=True, use_generated_starting_images=True, custom_name="NoName"):
        if strength_later is None:
            strength_later = strength
        control_images_options = {
            "control_images": control_images,
            "strength": strength,
            "strength_later": strength_later,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "tiled_vae": tiled_vae,
            "use_provided_starting_images": use_provided_starting_images,
            "use_generated_starting_images": use_generated_starting_images,
            "custom_name": custom_name,
        }
        return (control_images_options,)


class WanVideoFlowEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "source_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "skip_steps": ("INT", {"default": 4, "min": 0}),
                "drift_steps": ("INT", {"default": 0, "min": 0}),
                "drift_flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.01}),
                "source_cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "drift_cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
            },
            "optional": {
                "source_image_embeds": ("WANVIDIMAGE_EMBEDS", ),
            }
        }

    RETURN_TYPES = ("FLOWEDITARGS", )
    RETURN_NAMES = ("flowedit_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Flowedit options for WanVideo"

    def process(self, **kwargs):
        return (kwargs,)
    
class WanVideoLoopArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "shift_skip": ("INT", {"default": 6, "min": 0, "tooltip": "Skip step of latent shift"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the looping effect"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the looping effect"}),
            },
        }

    RETURN_TYPES = ("LOOPARGS", )
    RETURN_NAMES = ("loop_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Looping through latent shift as shown in https://github.com/YisuiTT/Mobius/"

    def process(self, **kwargs):
        return (kwargs,)

class WanVideoExperimentalArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "video_attention_split_steps": ("STRING", {"default": "", "tooltip": "Steps to split self attention when using multiple prompts"}),
                "cfg_zero_star": ("BOOLEAN", {"default": False, "tooltip": "https://github.com/WeichenFan/CFG-Zero-star"}),
                "use_zero_init": ("BOOLEAN", {"default": False}),
                "zero_star_steps": ("INT", {"default": 0, "min": 0, "tooltip": "Steps to split self attention when using multiple prompts"}),
                "use_fresca": ("BOOLEAN", {"default": False, "tooltip": "https://github.com/WikiChao/FreSca"}),
                "fresca_scale_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fresca_scale_high": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fresca_freq_cutoff": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
                "use_tcfg": ("BOOLEAN", {"default": False, "tooltip": "https://arxiv.org/abs/2503.18137 TCFG: Tangential Damping Classifier-free Guidance. CFG artifacts reduction."}),
                "raag_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Alpha value for RAAG, 1.0 is default, 0.0 is disabled."}),
                "bidirectional_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable bidirectional sampling, based on https://github.com/ff2416/WanFM"})
            },
            "optional": {
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "The node will do nothing and return a None value"}),
            }
        }

    RETURN_TYPES = ("EXPERIMENTALARGS", )
    RETURN_NAMES = ("exp_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Experimental stuff"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        # check if passthrough is provided and is true
        if kwargs.get("passthrough", False):
            print("WanVideoExperimentalArgs: Passthrough enabled, returning None")
            return (None, )
        print("WanVideoExperimentalArgs: Passthrough disabled, returning kwargs")
        return (kwargs,)
    
class WanVideoFreeInitArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "freeinit_num_iters": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Number of FreeInit iterations"}),
                "freeinit_method": (["butterworth", "ideal", "gaussian", "none"], {"default": "ideal", "tooltip": "Frequency filter type"}),
                "freeinit_n": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "Butterworth filter order (only for butterworth)"}),
                "freeinit_d_s": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Spatial filter cutoff"}),
                "freeinit_d_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Temporal filter cutoff"}),
            },
        }

    RETURN_TYPES = ("FREEINITARGS", )
    RETURN_NAMES = ("freeinit_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/TianxingWu/FreeInit; FreeInit, a concise yet effective method to improve temporal consistency of videos generated by diffusion models"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        return (kwargs,)
    
class WanVideoScheduler: #WIP
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "scheduler": (scheduler_list, {"default": "unipc"}),
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "Number of steps for the scheduler"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0, "tooltip": "Starting step for the scheduler"}),
                "end_step": ("INT", {"default": -1, "min": -1, "tooltip": "Ending step for the scheduler"})
            },
            "optional": {
                "sigmas": ("SIGMAS", ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("SIGMAS", "INT", "FLOAT", scheduler_list, "INT", "INT",)
    RETURN_NAMES = ("sigmas", "steps", "shift", "scheduler", "start_step", "end_step")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, scheduler, steps, start_step, end_step, shift, unique_id, sigmas=None):
        sample_scheduler, timesteps, start_idx, end_idx = get_scheduler(
            scheduler, 
            steps, 
            start_step, end_step, shift, 
            device, 
            sigmas=sigmas,
            log_timesteps=True)
        
        scheduler_dict = {
            "sample_scheduler": sample_scheduler,
            "timesteps": timesteps,
        }

        try:
            from server import PromptServer
            import io
            import base64
            import matplotlib.pyplot as plt
        except:
            PromptServer = None
        if unique_id and PromptServer is not None:
            try:
                # Plot sigmas and save to a buffer
                sigmas_np = sample_scheduler.full_sigmas.cpu().numpy()
                if not np.isclose(sigmas_np[-1], 0.0, atol=1e-6):
                    sigmas_np = np.append(sigmas_np, 0.0)
                buf = io.BytesIO()
                fig = plt.figure(facecolor='#353535')
                ax = fig.add_subplot(111)
                ax.set_facecolor('#353535')  # Set axes background color
                x_values = range(0, len(sigmas_np))
                ax.plot(x_values, sigmas_np)
                # Annotate each sigma value
                ax.scatter(x_values, sigmas_np, color='white', s=20, zorder=3)  # Small dots at each sigma
                for x, y in zip(x_values, sigmas_np):
                    if len(sigmas_np) <= 10:  # Only annotate if few steps
                        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(10, 1), ha='center', color='orange', fontsize=12)
                ax.set_xticks(x_values)
                ax.set_title("Sigmas", color='white')           # Title font color
                ax.set_xlabel("Step", color='white')            # X label font color
                ax.set_ylabel("Sigma Value", color='white')     # Y label font color
                ax.tick_params(axis='x', colors='white', labelsize=10)        # X tick color
                ax.tick_params(axis='y', colors='white', labelsize=10)        # Y tick color
                # Add split point if end_step is defined
                end_idx += 1
                if end_idx != -1 and 0 <= end_idx < len(sigmas_np) - 1:
                    ax.axvline(end_idx, color='red', linestyle='--', linewidth=2, label='end_step split')
                # Add split point if start_step is defined
                if start_idx > 0 and 0 <= start_idx < len(sigmas_np):
                    ax.axvline(start_idx, color='green', linestyle='--', linewidth=2, label='start_step split')
                if (end_idx != -1 and 0 <= end_idx < len(sigmas_np)) or (start_idx > 0 and 0 <= start_idx < len(sigmas_np)):
                    ax.legend()
                if start_idx < end_idx and 0 <= start_idx < len(sigmas_np) and 0 < end_idx < len(sigmas_np):
                    ax.axvspan(start_idx, end_idx, color='lightblue', alpha=0.1, label='Sampled Range')
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # Send as HTML img tag with base64 data
                html_img = f"<img src='data:image/png;base64,{img_base64}' alt='Sigmas Plot' style='max-width:100%; height:100%; overflow:hidden; display:block;'>"
                PromptServer.instance.send_progress_text(html_img, unique_id)
            except Exception as e:
                print("Failed to send sigmas plot:", e)
                pass

        return (sigmas, steps, shift, scheduler_dict, start_step, end_step)

rope_functions = ["default", "comfy", "comfy_chunked"]
class WanVideoRoPEFunction:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "rope_function": (rope_functions, {"default": "comfy"}),
                "ntk_scale_f": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "ntk_scale_h": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "ntk_scale_w": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (rope_functions, )
    RETURN_NAMES = ("rope_function",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, rope_function, ntk_scale_f, ntk_scale_h, ntk_scale_w):
        if ntk_scale_f != 1.0 or ntk_scale_h != 1.0 or ntk_scale_w != 1.0:
            rope_func_dict = {
                "rope_function": rope_function,
                "ntk_scale_f": ntk_scale_f,
                "ntk_scale_h": ntk_scale_h,
                "ntk_scale_w": ntk_scale_w,
            }
            return (rope_func_dict,)
        return (rope_function,)


#region Sampler
class WanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (scheduler_list, {"default": "unipc",}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
            },
            "optional": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "cache_args": ("CACHEARGS", ),
                "flowedit_args": ("FLOWEDITARGS", ),
                "batched_cfg": ("BOOLEAN", {"default": False, "tooltip": "Batch cond and uncond for faster sampling, possibly faster on some hardware, uses more memory"}),
                "slg_args": ("SLGARGS", ),
                "rope_function": (rope_functions, {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile. Chunked version has reduced peak VRAM usage when not using torch.compile"}),
                "loop_args": ("LOOPARGS", ),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "sigmas": ("SIGMAS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS", ),
                "uni3c_embeds": ("UNI3C_EMBEDS", ),
                "multitalk_embeds": ("MULTITALK_EMBEDS", ),
                "freeinit_args": ("FREEINITARGS", ),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Start step for the sampling, 0 means full sampling, otherwise samples only from this step"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1, "tooltip": "End step for the sampling, -1 means full sampling, otherwise samples only until this step"}),
                "add_noise_to_samples": ("BOOLEAN", {"default": False, "tooltip": "Add noise to the samples before sampling, needed for video2video sampling when starting from clean video"}),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "Pass the samples through without modification"}),
                "denoise_latents": ("BOOLEAN", {"default": False, "tooltip": "Denoise the latents before VAE decoding in multitalk mode for improved quality"}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT",)
    RETURN_NAMES = ("samples", "denoised_samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, model, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index, text_embeds=None,
        force_offload=True, samples=None, feta_args=None, denoise_strength=1.0, context_options=None, 
        cache_args=None, teacache_args=None, flowedit_args=None, batched_cfg=False, slg_args=None, rope_function="default", loop_args=None, 
        experimental_args=None, sigmas=None, unianimate_poses=None, fantasytalking_embeds=None, uni3c_embeds=None, multitalk_embeds=None, freeinit_args=None, start_step=0, end_step=-1, add_noise_to_samples=False,
        passthrough=False, denoise_latents=False):

        if passthrough or denoise_strength <= 0.0:
            return (samples, samples,)
        input_samples = None
        patcher = model
        model = model.model
        transformer = model.diffusion_model

        dtype = model["base_dtype"]
        weight_dtype = model["weight_dtype"]
        fp8_matmul = model["fp8_matmul"]
        gguf_reader = model["gguf_reader"]
        control_lora = model["control_lora"]

        transformer_options = patcher.model_options.get("transformer_options", None)
        merge_loras = transformer_options["merge_loras"]

        block_swap_args = transformer_options.get("block_swap_args", None)
        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", False)
            transformer.blocks_to_swap = block_swap_args.get("blocks_to_swap", 0)
            transformer.vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", 0)
            transformer.prefetch_blocks = block_swap_args.get("prefetch_blocks", 0)
            transformer.block_swap_debug = block_swap_args.get("block_swap_debug", False)
            transformer.offload_img_emb = block_swap_args.get("offload_img_emb", False)
            transformer.offload_txt_emb = block_swap_args.get("offload_txt_emb", False)

        is_5b = transformer.out_dim == 48
        vae_upscale_factor = 16 if is_5b else 8

        # Load weights
        if transformer.patched_linear and gguf_reader is None:
            load_weights(patcher.model.diffusion_model, patcher.model["sd"], weight_dtype, base_dtype=dtype, transformer_load_device=device, block_swap_args=block_swap_args)

        if gguf_reader is not None: #handle GGUF
            load_weights(transformer, patcher.model["sd"], base_dtype=dtype, transformer_load_device=device, patcher=patcher, gguf=True, reader=gguf_reader, block_swap_args=block_swap_args)
            set_lora_params_gguf(transformer, patcher.patches)
            transformer.patched_linear = True
        elif len(patcher.patches) != 0 and transformer.patched_linear: #handle patched linear layers (unmerged loras, fp8 scaled)
            log.info(f"Using {len(patcher.patches)} LoRA weight patches for WanVideo model")
            if not merge_loras and fp8_matmul:
                raise NotImplementedError("FP8 matmul with unmerged LoRAs is not supported")
            set_lora_params(transformer, patcher.patches)
        else:
            remove_lora_from_module(transformer) #clear possible unmerged lora weights

        transformer.lora_scheduling_enabled = transformer_options.get("lora_scheduling_enabled", False)

        #torch.compile
        if model["auto_cpu_offload"] is False:
            transformer = compile_model(transformer, model["compile_args"])

        multitalk_sampling = image_embeds.get("multitalk_sampling", False)

        if multitalk_sampling and context_options is not None:
            raise Exception("context_options are not compatible or necessary with 'WanVideoImageToVideoMultiTalk' node, since it's already an alternative method that creates the video in a loop.")

        if not multitalk_sampling and scheduler == "multitalk":
            raise Exception("multitalk scheduler is only for multitalk sampling when using ImagetoVideoMultiTalk -node")
        
        # check if text_embeds is a list
        # text_embeds_list = [text_embeds]
        # if isinstance(text_embeds, list):
        #     print(f"!!!!!!Received {len(text_embeds)} text_embeds as a list for {steps} steps.")
        #     text_embeds_list = []
        #     for text_embed in text_embeds:
        #         text_embed_correct_device = dict_to_device(text_embed, device)
        #         text_embeds_list.append(text_embed_correct_device)
        #     text_embeds = text_embeds[0]
        if text_embeds == None:
            print(f"!!!!!!Received No text_embeds at all for {steps} steps.")
            text_embeds = {
                "prompt_embeds": [],
                "negative_prompt_embeds": [],
            }
            text_embeds_list = [text_embeds]
        else:
            
            charge_uk_embeds = text_embeds.get("charge_uk_embeds", None)
            if charge_uk_embeds is not None:
                print(f"!!!!!!Received {len(charge_uk_embeds)} text_embeds as a list for {steps} steps.")
                text_embeds_list = []
                for text_embed in charge_uk_embeds:
                    text_embed_correct_device = dict_to_device(text_embed, device)
                    text_embeds_list.append(text_embed_correct_device)
                text_embeds = text_embeds_list[0]
            else:
                print(f"!!!!!!Received 1 text_embeds as a single item for {steps} steps.")
                text_embeds = dict_to_device(text_embeds, device)
                text_embeds_list = [text_embeds]

        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)

        #region Scheduler
        sample_scheduler = None
        if isinstance(scheduler, dict):
            sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
            timesteps = scheduler["timesteps"]
        elif scheduler != "multitalk":
            sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas, log_timesteps=True, keep_number_of_steps=True)
            log.info(f"sigmas: {sample_scheduler.sigmas}")
        else:
            timesteps = torch.tensor([1000, 750, 500, 250], device=device)

        total_steps = steps
        steps = len(timesteps)

        is_pusa = "pusa" in sample_scheduler.__class__.__name__.lower()

        if end_step != -1 and start_step >= end_step:
            raise ValueError("start_step must be less than end_step")

        scheduler_step_args = {"generator": seed_g}
        step_sig = inspect.signature(sample_scheduler.step)
        for arg in list(scheduler_step_args.keys()):
            if arg not in step_sig.parameters:
                scheduler_step_args.pop(arg)

        if isinstance(cfg, list):
            if steps < len(cfg):
                log.info(f"Received {len(cfg)} cfg values, but only {steps} steps. Slicing cfg list to match steps.")
                cfg = cfg[:steps]
            elif steps > len(cfg):
                log.info(f"Received only {len(cfg)} cfg values, but {steps} steps. Extending cfg list to match steps.")
                cfg.extend([cfg[-1]] * (steps - len(cfg)))
            log.info(f"Using per-step cfg list: {cfg}")
        else:
            cfg = [cfg] * (steps + 1)
       
        control_latents = control_camera_latents = clip_fea = clip_fea_neg = end_image = recammaster = camera_embed = unianim_data = None
        vace_data = vace_context = vace_scale = None
        fun_or_fl2v_model = has_ref = drop_last = False
        phantom_latents = fun_ref_image = ATI_tracks = None
        add_cond = attn_cond = attn_cond_neg = noise_pred_flipped = None
        humo_audio = humo_audio_neg = None

        #I2V
        image_cond = image_embeds.get("image_embeds", None)
        if image_cond is not None:
            if transformer.in_dim == 16:
                raise ValueError("T2V (text to video) model detected, encoded images only work with I2V (Image to video) models")
            elif transformer.in_dim not in [48, 32]: # fun 2.1 models don't use the mask
                image_cond_mask = image_embeds.get("mask", None)
                if image_cond_mask is not None:
                    image_cond = torch.cat([image_cond_mask, image_cond])
            else:
                image_cond[:, 1:] = 0

            log.info(f"image_cond shape: {image_cond.shape}")

            #ATI tracks
            if transformer_options is not None:
                ATI_tracks = transformer_options.get("ati_tracks", None)
                if ATI_tracks is not None:
                    from .ATI.motion_patch import patch_motion
                    topk = transformer_options.get("ati_topk", 2)
                    temperature = transformer_options.get("ati_temperature", 220.0)
                    ati_start_percent = transformer_options.get("ati_start_percent", 0.0)
                    ati_end_percent = transformer_options.get("ati_end_percent", 1.0)
                    image_cond_ati = patch_motion(ATI_tracks.to(image_cond.device, image_cond.dtype), image_cond, topk=topk, temperature=temperature)
                    log.info(f"ATI tracks shape: {ATI_tracks.shape}")
            
            add_cond_latents = image_embeds.get("add_cond_latents", None)
            if add_cond_latents is not None:
                add_cond = add_cond_latents["pose_latent"]
                attn_cond = add_cond_latents["ref_latent"]
                attn_cond_neg = add_cond_latents["ref_latent_neg"]
                add_cond_start_percent = add_cond_latents["pose_cond_start_percent"]
                add_cond_end_percent = add_cond_latents["pose_cond_end_percent"]

            end_image = image_embeds.get("end_image", None)
            fun_or_fl2v_model = image_embeds.get("fun_or_fl2v_model", False)

            noise = torch.randn( #C, T, H, W
                48 if is_5b else 16,
                (image_embeds["num_frames"] - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1),
                image_embeds["lat_h"],
                image_embeds["lat_w"],
                dtype=torch.float32,
                generator=seed_g,
                device=torch.device("cpu"))
            seq_len = image_embeds["max_seq_len"]

            clip_fea = image_embeds.get("clip_context", None)
            if clip_fea is not None:
                clip_fea = clip_fea.to(dtype)
            clip_fea_neg = image_embeds.get("negative_clip_context", None)
            if clip_fea_neg is not None:
                clip_fea_neg = clip_fea_neg.to(dtype)

            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                if transformer.in_dim not in [148, 52, 48, 36, 32]:
                    raise ValueError("Control signal only works with Fun-Control model")

                control_latents = control_embeds.get("control_images", None)
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
                control_camera_latents = control_embeds.get("control_camera_latents", None)
                if control_camera_latents is not None:
                    if transformer.control_adapter is None:
                        raise ValueError("Control camera latents are only supported with Fun-Control-Camera model")
                    control_camera_start_percent = control_embeds.get("control_camera_start_percent", 0.0)
                    control_camera_end_percent = control_embeds.get("control_camera_end_percent", 1.0)
                
            drop_last = image_embeds.get("drop_last", False)
            has_ref = image_embeds.get("has_ref", False)
        else: #t2v
            target_shape = image_embeds.get("target_shape", None)
            if target_shape is None:
                raise ValueError("Empty image embeds must be provided for T2V models")
            
            has_ref = image_embeds.get("has_ref", False)

            # VACE
            vace_context = image_embeds.get("vace_context", None)
            vace_scale = image_embeds.get("vace_scale", None)
            if not isinstance(vace_scale, list):
                vace_scale = [vace_scale] * (steps+1)
            vace_start_percent = image_embeds.get("vace_start_percent", 0.0)
            vace_end_percent = image_embeds.get("vace_end_percent", 1.0)
            vace_seqlen = image_embeds.get("vace_seq_len", None)

            vace_additional_embeds = image_embeds.get("additional_vace_inputs", [])
            if vace_context is not None:
                vace_data = [
                    {"context": vace_context, 
                     "scale": vace_scale, 
                     "start": vace_start_percent, 
                     "end": vace_end_percent,
                     "seq_len": vace_seqlen
                     }
                ]
                if len(vace_additional_embeds) > 0:
                    for i in range(len(vace_additional_embeds)):
                        if vace_additional_embeds[i].get("has_ref", False):
                            has_ref = True
                        vace_scale = vace_additional_embeds[i]["vace_scale"]
                        if not isinstance(vace_scale, list):
                            vace_scale = [vace_scale] * (steps+1)
                        vace_data.append({
                            "context": vace_additional_embeds[i]["vace_context"],
                            "scale": vace_scale,
                            "start": vace_additional_embeds[i]["vace_start_percent"],
                            "end": vace_additional_embeds[i]["vace_end_percent"],
                            "seq_len": vace_additional_embeds[i]["vace_seq_len"]
                        })

            noise = torch.randn(
                    48 if is_5b else 16,
                    target_shape[1] + 1 if has_ref else target_shape[1],
                    target_shape[2] // 2 if is_5b else target_shape[2], #todo make this smarter
                    target_shape[3] // 2 if is_5b else target_shape[3], #todo make this smarter
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    generator=seed_g)
            
            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

            recammaster = image_embeds.get("recammaster", None)
            if recammaster is not None:
                camera_embed = recammaster.get("camera_embed", None)
                recam_latents = recammaster.get("source_latents", None)
                orig_noise_len = noise.shape[1]
                log.info(f"RecamMaster camera embed shape: {camera_embed.shape}")
                log.info(f"RecamMaster source video shape: {recam_latents.shape}")
                seq_len *= 2
            
            # Fun control and control lora
            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                control_latents = control_embeds.get("control_images", None)
                if control_latents is not None:
                    control_latents = control_latents.to(device)

                control_camera_latents = control_embeds.get("control_camera_latents", None)
                if control_camera_latents is not None:
                    if transformer.control_adapter is None:
                        raise ValueError("Control camera latents are only supported with Fun-Control-Camera model")
                    control_camera_start_percent = control_embeds.get("control_camera_start_percent", 0.0)
                    control_camera_end_percent = control_embeds.get("control_camera_end_percent", 1.0)

                if control_lora:
                    image_cond = control_latents.to(device)
                    if not patcher.model.is_patched:
                        log.info("Re-loading control LoRA...")
                        patcher = apply_lora(patcher, device, device, low_mem_load=False, control_lora=True)
                        patcher.model.is_patched = True
                else:
                    if transformer.in_dim not in [148, 48, 36, 32, 52]:
                        raise ValueError("Control signal only works with Fun-Control model")
                    image_cond = torch.zeros_like(noise).to(device) #fun control
                    if transformer.in_dim in [148, 52] or transformer.control_adapter is not None: #fun 2.2 control
                        mask_latents = torch.tile(
                            torch.zeros_like(noise[:1]), [4, 1, 1, 1]
                        )
                        masked_video_latents_input = torch.zeros_like(noise)
                        image_cond = torch.cat([mask_latents, masked_video_latents_input], dim=0).to(device)
                    clip_fea = None
                    fun_ref_image = control_embeds.get("fun_ref_image", None)
                    if fun_ref_image is not None:
                        if transformer.ref_conv.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                            raise ValueError("Fun-Control reference image won't work with this specific fp8_scaled model, it's been fixed in latest version of the model")
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
            else:
                if transformer.in_dim in [148, 52]: #fun inp
                    mask_latents = torch.tile(
                        torch.zeros_like(noise[:1]), [4, 1, 1, 1]
                    )
                    masked_video_latents_input = torch.zeros_like(noise)
                    image_cond = torch.cat([mask_latents, masked_video_latents_input], dim=0).to(device)

            # Phantom inputs
            phantom_latents = image_embeds.get("phantom_latents", None)
            phantom_cfg_scale = image_embeds.get("phantom_cfg_scale", None)
            if not isinstance(phantom_cfg_scale, list):
                phantom_cfg_scale = [phantom_cfg_scale] * (steps +1)
            phantom_start_percent = image_embeds.get("phantom_start_percent", 0.0)
            phantom_end_percent = image_embeds.get("phantom_end_percent", 1.0)


        #HuMo inputs
        humo_audio = image_embeds.get("humo_audio_emb", None)
        if humo_audio is not None:
            humo_audio = humo_audio.to(device, dtype)
        humo_audio_neg = image_embeds.get("humo_audio_emb_neg", None)
        if humo_audio_neg is not None:
            humo_audio_neg = humo_audio_neg.to(device, dtype)
        humo_audio_scale = image_embeds.get("humo_audio_scale", 1.0)
        humo_image_cond = image_embeds.get("humo_image_cond", None)
        humo_image_cond_neg = image_embeds.get("humo_image_cond_neg", None)
        humo_reference_count = image_embeds.get("humo_reference_count", 0)
        humo_audio_cfg_scale = image_embeds.get("humo_audio_cfg_scale", 1.0)
        humo_start_percent = image_embeds.get("humo_start_percent", 0.0)
        humo_end_percent = image_embeds.get("humo_end_percent", 1.0)
        if not isinstance(humo_audio_cfg_scale, list):
            humo_audio_cfg_scale = [humo_audio_cfg_scale] * (steps + 1)

        latent_video_length = noise.shape[1]

        # Initialize FreeInit filter if enabled
        freq_filter = None
        if freeinit_args is not None:
            from .freeinit.freeinit_utils import get_freq_filter, freq_mix_3d
            filter_shape = list(noise.shape)  # [batch, C, T, H, W]
            freq_filter = get_freq_filter(
                filter_shape,
                device=device,
                filter_type=freeinit_args.get("freeinit_method", "butterworth"),
                n=freeinit_args.get("freeinit_n", 4) if freeinit_args.get("freeinit_method", "butterworth") == "butterworth" else None,
                d_s=freeinit_args.get("freeinit_s", 1.0),
                d_t=freeinit_args.get("freeinit_t", 1.0)
            )
            if samples is not None:
                saved_generator_state = samples.get("generator_state", None)
                if saved_generator_state is not None:
                    seed_g.set_state(saved_generator_state)
        
        # UniAnimate
        if unianimate_poses is not None:
            if not multitalk_sampling:
                # Process poses normally for non-multitalk sampling
                transformer.dwpose_embedding.to(device, dtype)
                dwpose_data = unianimate_poses["pose"].to(device, dtype)
                dwpose_data = torch.cat([dwpose_data[:,:,:1].repeat(1,1,3,1,1), dwpose_data], dim=2)
                dwpose_data = transformer.dwpose_embedding(dwpose_data)
                log.info(f"UniAnimate pose embed shape: {dwpose_data.shape}")
                if dwpose_data.shape[2] > latent_video_length:
                    log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is longer than the video length {latent_video_length}, truncating")
                    dwpose_data = dwpose_data[:,:, :latent_video_length]
                elif dwpose_data.shape[2] < latent_video_length:
                    log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is shorter than the video length {latent_video_length}, padding with last pose")
                    pad_len = latent_video_length - dwpose_data.shape[2]
                    pad = dwpose_data[:,:,:1].repeat(1,1,pad_len,1,1)
                    dwpose_data = torch.cat([dwpose_data, pad], dim=2)
                
                random_ref_dwpose_data = None
                if image_cond is not None:
                    transformer.randomref_embedding_pose.to(device, dtype)
                    random_ref_dwpose = unianimate_poses.get("ref", None)
                    if random_ref_dwpose is not None:
                        random_ref_dwpose_data = transformer.randomref_embedding_pose(
                            random_ref_dwpose.to(device, dtype)
                        ).unsqueeze(2).to(dtype) # [1, 20, 104, 60]
                    del random_ref_dwpose
                    
                unianim_data = {
                    "dwpose": dwpose_data,
                    "random_ref": random_ref_dwpose_data.squeeze(0) if random_ref_dwpose_data is not None else None,
                    "strength": unianimate_poses["strength"],
                    "start_percent": unianimate_poses["start_percent"],
                    "end_percent": unianimate_poses["end_percent"]
                }
            else:
                # For multitalk sampling, we'll process poses per chunk in the loop
                unianim_data = None
                log.info(f"UniAnimate poses will be processed per chunk for multitalk sampling")
                print(f"[UniAnimate Debug] Original pose tensor shape: {unianimate_poses['pose'].shape}")
                print(f"[UniAnimate Debug] Using multitalk sampling - poses will be processed dynamically")

        # FantasyTalking
        audio_proj = multitalk_audio_embedding = None
        audio_scale = 1.0
        max_multitalk_frames = None
        if fantasytalking_embeds is not None:
            audio_proj = fantasytalking_embeds["audio_proj"].to(device)
            audio_scale = fantasytalking_embeds["audio_scale"]
            audio_cfg_scale = fantasytalking_embeds["audio_cfg_scale"]
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps +1)
            log.info(f"Audio proj shape: {audio_proj.shape}")
        elif multitalk_embeds is not None:
            # Handle single or multiple speaker embeddings
            audio_features_in = multitalk_embeds.get("audio_features", None)
            max_multitalk_frames = multitalk_embeds.get("max_frames", None)
            if audio_features_in is None:
                multitalk_audio_embedding = None
            else:
                if isinstance(audio_features_in, list):
                    multitalk_audio_embedding = [emb.to(device, dtype) for emb in audio_features_in]
                else:
                    # keep backward-compatibility with single tensor input
                    multitalk_audio_embedding = [audio_features_in.to(device, dtype)]

            audio_scale = multitalk_embeds.get("audio_scale", 1.0)
            audio_cfg_scale = multitalk_embeds.get("audio_cfg_scale", 1.0)
            ref_target_masks = multitalk_embeds.get("ref_target_masks", None)
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps + 1)

            shapes = [tuple(e.shape) for e in multitalk_audio_embedding]
            log.info(f"Multitalk audio features shapes (per speaker): {shapes}")

        # FantasyPortrait
        fantasy_portrait_input = None
        fantasy_portrait_embeds = image_embeds.get("portrait_embeds", None)
        if fantasy_portrait_embeds is not None:
            log.info("Using FantasyPortrait embeddings")
            fantasy_portrait_input = {
                "adapter_proj": fantasy_portrait_embeds.get("adapter_proj", None),
                "strength": fantasy_portrait_embeds.get("strength", 1.0),
                "start_percent": fantasy_portrait_embeds.get("start_percent", 0.0),
                "end_percent": fantasy_portrait_embeds.get("end_percent", 1.0),
            }

        # MiniMax Remover
        minimax_latents = minimax_mask_latents = None
        minimax_latents = image_embeds.get("minimax_latents", None)
        minimax_mask_latents = image_embeds.get("minimax_mask_latents", None)
        if minimax_latents is not None:
            log.info(f"minimax_latents: {minimax_latents.shape}")
            log.info(f"minimax_mask_latents: {minimax_mask_latents.shape}")
            minimax_latents = minimax_latents.to(device, dtype)
            minimax_mask_latents = minimax_mask_latents.to(device, dtype)

        # Context windows
        is_looped = False
        context_reference_latent = None
        if context_options is not None:
            context_schedule = context_options["context_schedule"]
            context_frames =  (context_options["context_frames"] - 1) // 4 + 1
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4
            context_reference_latent = context_options.get("reference_latent", None)

            # Get total number of prompts
            num_prompts = len(text_embeds["prompt_embeds"])
            log.info(f"Number of prompts: {num_prompts}")
            # Calculate which section this context window belongs to
            section_size = (latent_video_length / num_prompts) if num_prompts != 0 else 1
            log.info(f"Section size: {section_size}")
            is_looped = context_schedule == "uniform_looped"

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * context_frames)

            if context_options["freenoise"]:
                log.info("Applying FreeNoise")
                # code from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
                delta = context_frames - context_overlap
                for start_idx in range(0, latent_video_length-context_frames, delta):
                    place_idx = start_idx + context_frames
                    if place_idx >= latent_video_length:
                        break
                    end_idx = place_idx - 1

                    if end_idx + delta >= latent_video_length:
                        final_delta = latent_video_length - place_idx
                        list_idx = torch.tensor(list(range(start_idx,start_idx+final_delta)), device=torch.device("cpu"), dtype=torch.long)
                        list_idx = list_idx[torch.randperm(final_delta, generator=seed_g)]
                        noise[:, place_idx:place_idx + final_delta, :, :] = noise[:, list_idx, :, :]
                        break
                    list_idx = torch.tensor(list(range(start_idx,start_idx+delta)), device=torch.device("cpu"), dtype=torch.long)
                    list_idx = list_idx[torch.randperm(delta, generator=seed_g)]
                    noise[:, place_idx:place_idx + delta, :, :] = noise[:, list_idx, :, :]
            
            log.info(f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
            from .context_windows.context import get_context_scheduler, create_window_mask, WindowTracker
            self.window_tracker = WindowTracker(verbose=context_options["verbose"])
            context = get_context_scheduler(context_schedule)

        #MTV Crafter
        mtv_input = image_embeds.get("mtv_crafter_motion", None)
        mtv_motion_tokens = None
        if mtv_input is not None:
            from .MTV.mtv import prepare_motion_embeddings
            log.info("Using MTV Crafter embeddings")
            mtv_start_percent = mtv_input.get("start_percent", 0.0)
            mtv_end_percent = mtv_input.get("end_percent", 1.0)
            mtv_strength = mtv_input.get("strength", 1.0)
            mtv_motion_tokens = mtv_input.get("mtv_motion_tokens", None)
            if not isinstance(mtv_strength, list):
                mtv_strength = [mtv_strength] * (steps + 1)
            d = transformer.dim // transformer.num_heads
            mtv_freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)
            motion_rotary_emb = prepare_motion_embeddings(
                latent_video_length if context_options is None else context_frames, 
                24, mtv_input["global_mean"], [mtv_input["global_std"]], device=device)
            log.info(f"mtv_motion_rotary_emb: {motion_rotary_emb[0].shape}")
            mtv_freqs = mtv_freqs.to(device, dtype)

        #region S2V
        s2v_audio_input = s2v_ref_latent = s2v_pose = s2v_ref_motion = None
        framepack = False
        s2v_audio_embeds = image_embeds.get("audio_embeds", None)
        if s2v_audio_embeds is not None:
            log.info(f"Using S2V audio embeddings")
            framepack = s2v_audio_embeds.get("enable_framepack", False)
            if framepack and context_options is not None:
                raise ValueError("S2V framepack and context windows cannot be used at the same time")

            s2v_audio_input = s2v_audio_embeds.get("audio_embed_bucket", None)
            if s2v_audio_input is not None:
                #s2v_audio_input = s2v_audio_input[..., 0:image_embeds["num_frames"]]
                s2v_audio_input = s2v_audio_input.to(device, dtype)
            s2v_audio_scale = s2v_audio_embeds["audio_scale"]
            s2v_ref_latent = s2v_audio_embeds.get("ref_latent", None)
            if s2v_ref_latent is not None:
                s2v_ref_latent = s2v_ref_latent.to(device, dtype)
            s2v_ref_motion = s2v_audio_embeds.get("ref_motion", None)
            if s2v_ref_motion is not None:
                s2v_ref_motion = s2v_ref_motion.to(device, dtype)
            s2v_pose = s2v_audio_embeds.get("pose_latent", None)
            if s2v_pose is not None:
                s2v_pose = s2v_pose.to(device, dtype)
            s2v_pose_start_percent = s2v_audio_embeds.get("pose_start_percent", 0.0)
            s2v_pose_end_percent = s2v_audio_embeds.get("pose_end_percent", 1.0)
            s2v_num_repeat = s2v_audio_embeds.get("num_repeat", 1)
            vae = s2v_audio_embeds.get("vae", None)

        # vid2vid
        noise_mask=original_image=None
        if samples is not None and not multitalk_sampling:
            saved_generator_state = samples.get("generator_state", None)
            if saved_generator_state is not None:
                seed_g.set_state(saved_generator_state)
            input_samples = samples["samples"].squeeze(0).to(noise)
            if input_samples.shape[1] != noise.shape[1]:
               input_samples = torch.cat([input_samples[:, :1].repeat(1, noise.shape[1] - input_samples.shape[1], 1, 1), input_samples], dim=1)
            
            if add_noise_to_samples:
                latent_timestep = timesteps[:1].to(noise)
                noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples
            else:
                noise = input_samples

            noise_mask = samples.get("noise_mask", None)
            if noise_mask is not None:
                log.info(f"Latent noise_mask shape: {noise_mask.shape}")
                original_image = samples.get("original_image", None)
                if original_image is None:
                    original_image = input_samples
                if len(noise_mask.shape) == 4:
                    noise_mask = noise_mask.squeeze(1)
                if noise_mask.shape[0] < noise.shape[1]:
                    noise_mask = noise_mask.repeat(noise.shape[1] // noise_mask.shape[0], 1, 1)

                noise_mask = torch.nn.functional.interpolate(
                    noise_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                    size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                    mode='trilinear',
                    align_corners=False
                ).repeat(1, noise.shape[0], 1, 1, 1)
        
        # extra latents (Pusa) and 5b
        latents_to_insert = add_index = noise_multipliers = None
        extra_latents = image_embeds.get("extra_latents", None)
        all_indices = []
        noise_multiplier_list = image_embeds.get("pusa_noise_multipliers", None)
        if noise_multiplier_list is not None:
            if len(noise_multiplier_list) != latent_video_length:
                noise_multipliers = torch.zeros(latent_video_length)
            else:
                noise_multipliers = torch.tensor(noise_multiplier_list)
                log.info(f"Using Pusa noise multipliers: {noise_multipliers}")
        if extra_latents is not None and transformer.multitalk_model_type.lower() != "infinitetalk":
            if noise_multiplier_list is not None:
                noise_multiplier_list = list(noise_multiplier_list) + [1.0] * (len(all_indices) - len(noise_multiplier_list))
            for i, entry in enumerate(extra_latents):
                add_index = entry["index"]
                num_extra_frames = entry["samples"].shape[2]
                # Handle negative indices
                if add_index < 0:
                    add_index = noise.shape[1] + add_index
                add_index = max(0, min(add_index, noise.shape[1] - num_extra_frames))
                if start_step == 0:
                    noise[:, add_index:add_index+num_extra_frames] = entry["samples"].to(noise)
                    log.info(f"Adding extra samples to latent indices {add_index} to {add_index+num_extra_frames-1}")
                all_indices.extend(range(add_index, add_index+num_extra_frames))
            if noise_multipliers is not None and len(noise_multiplier_list) != latent_video_length:
                for i, idx in enumerate(all_indices):
                    noise_multipliers[idx] = noise_multiplier_list[i]
                log.info(f"Using Pusa noise multipliers: {noise_multipliers}")

        latent = noise.to(device)

        #controlnet
        controlnet_latents = controlnet = None
        if transformer_options is not None:
            controlnet = transformer_options.get("controlnet", None)
            if controlnet is not None:
                self.controlnet = controlnet["controlnet"]
                controlnet_start = controlnet["controlnet_start"]
                controlnet_end = controlnet["controlnet_end"]
                controlnet_latents = controlnet["control_latents"]
                controlnet["controlnet_weight"] = controlnet["controlnet_strength"]
                controlnet["controlnet_stride"] = controlnet["control_stride"]

        #uni3c
        pcd_data = pcd_data_input = None
        if uni3c_embeds is not None:
            transformer.controlnet = uni3c_embeds["controlnet"]
            render_latent = uni3c_embeds["render_latent"].to(device)
            if render_latent.shape != noise.shape:
                render_latent = torch.nn.functional.interpolate(render_latent, size=(noise.shape[1], noise.shape[2], noise.shape[3]), mode='trilinear', align_corners=False)
            pcd_data = {
                "render_latent": render_latent,
                "render_mask": uni3c_embeds["render_mask"],
                "camera_embedding": uni3c_embeds["camera_embedding"],
                "controlnet_weight": uni3c_embeds["controlnet_weight"],
                "start": uni3c_embeds["start"],
                "end": uni3c_embeds["end"],
            }

        # Enhance-a-video (feta)
        if feta_args is not None and latent_video_length > 1:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            set_num_frames(latent_video_length) if context_options is None else set_num_frames(context_frames)
            enhance_enabled = True
        else:
            feta_args = None
            enhance_enabled = False

        # EchoShot https://github.com/D2I-ai/EchoShot
        echoshot = False
        shot_len = None
        if text_embeds is not None:
            echoshot = text_embeds.get("echoshot", False)
        if echoshot:
            shot_num = len(text_embeds["prompt_embeds"])
            shot_len = [latent_video_length//shot_num] * (shot_num-1)
            shot_len.append(latent_video_length-sum(shot_len))
            rope_function = "default" #echoshot does not support comfy rope function
            log.info(f"Number of shots in prompt: {shot_num}, Shot token lengths: {shot_len}")

        
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()

        #blockswap init
        if not transformer.patched_linear:
            if block_swap_args is not None:
                transformer.use_non_blocking = block_swap_args.get("use_non_blocking", False)
                for name, param in transformer.named_parameters():
                    if "block" not in name:
                        param.data = param.data.to(device)
                    if "control_adapter" in name:
                        param.data = param.data.to(device)
                    elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                        param.data = param.data.to(offload_device)
                    elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                        param.data = param.data.to(offload_device)

                transformer.block_swap(
                    block_swap_args["blocks_to_swap"] - 1 ,
                    block_swap_args["offload_txt_emb"],
                    block_swap_args["offload_img_emb"],
                    vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
                    prefetch_blocks = block_swap_args.get("prefetch_blocks", 0),
                    block_swap_debug = block_swap_args.get("block_swap_debug", False),
                )
            elif model["auto_cpu_offload"]:
                for module in transformer.modules():
                    if hasattr(module, "offload"):
                        module.offload()
                    if hasattr(module, "onload"):
                        module.onload()
                for block in transformer.blocks:
                    block.modulation = torch.nn.Parameter(block.modulation.to(device))
                transformer.head.modulation = torch.nn.Parameter(transformer.head.modulation.to(device))
            else:
                transformer.to(device)

        # Initialize Cache if enabled
        previous_cache_states = None
        transformer.enable_teacache = transformer.enable_magcache = transformer.enable_easycache = False
        cache_args = teacache_args if teacache_args is not None else cache_args #for backward compatibility on old workflows
        if cache_args is not None:            
            from .cache_methods.cache_methods import set_transformer_cache_method
            transformer = set_transformer_cache_method(transformer, timesteps, cache_args)

            # Initialize cache state
            if samples is not None:
                previous_cache_states = samples.get("cache_states", None)
                print("Using previous cache states", previous_cache_states)
                if previous_cache_states is not None:
                    log.info("Using cache states from previous sampler")
                    
                    self.cache_state = previous_cache_states["cache_state"]
                    transformer.easycache_state = previous_cache_states["easycache_state"]
                    transformer.magcache_state = previous_cache_states["magcache_state"]
                    transformer.teacache_state = previous_cache_states["teacache_state"]

        if previous_cache_states is None:
            self.cache_state = [None, None]
            if phantom_latents is not None:
                log.info(f"Phantom latents shape: {phantom_latents.shape}")
                self.cache_state = [None, None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []

        # Skip layer guidance (SLG)
        if slg_args is not None:
            assert batched_cfg is not None, "Batched cfg is not supported with SLG"
            transformer.slg_blocks = slg_args["blocks"]
            transformer.slg_start_percent = slg_args["start_percent"]
            transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None

        # Setup radial attention
        if transformer.attention_mode == "radial_sage_attention":
            setup_radial_attention(transformer, transformer_options, latent, seq_len, latent_video_length, context_options=context_options)

        # FlowEdit setup
        if flowedit_args is not None:
            source_embeds = flowedit_args["source_embeds"]
            source_embeds = dict_to_device(source_embeds, device)
            source_image_embeds = flowedit_args.get("source_image_embeds", image_embeds)
            source_image_cond = source_image_embeds.get("image_embeds", None)
            source_clip_fea = source_image_embeds.get("clip_fea", clip_fea)
            if source_image_cond is not None:
                source_image_cond = source_image_cond.to(dtype)
            skip_steps = flowedit_args["skip_steps"]
            drift_steps = flowedit_args["drift_steps"]
            source_cfg = flowedit_args["source_cfg"]
            if not isinstance(source_cfg, list):
                source_cfg = [source_cfg] * (steps +1)
            drift_cfg = flowedit_args["drift_cfg"]
            if not isinstance(drift_cfg, list):
                drift_cfg = [drift_cfg] * (steps +1)

            x_init = samples["samples"].clone().squeeze(0).to(device)
            x_tgt = samples["samples"].squeeze(0).to(device)

            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=flowedit_args["drift_flow_shift"],
                use_dynamic_shifting=False)

            sampling_sigmas = get_sampling_sigmas(steps, flowedit_args["drift_flow_shift"])
           
            drift_timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)

            if drift_steps > 0:
                drift_timesteps = torch.cat([drift_timesteps, torch.tensor([0]).to(drift_timesteps.device)]).to(drift_timesteps.device)
                timesteps[-drift_steps:] = drift_timesteps[-drift_steps:]

        # Experimental args
        use_cfg_zero_star = use_tangential = use_fresca = bidirectional_sampling =False
        raag_alpha = 0.0
        if experimental_args is not None:
            video_attention_split_steps = experimental_args.get("video_attention_split_steps", [])
            if video_attention_split_steps:
                transformer.video_attention_split_steps = [int(x.strip()) for x in video_attention_split_steps.split(",")]
            else:
                transformer.video_attention_split_steps = []

            use_zero_init = experimental_args.get("use_zero_init", True)
            use_cfg_zero_star = experimental_args.get("cfg_zero_star", False)
            use_tangential = experimental_args.get("use_tcfg", False)
            zero_star_steps = experimental_args.get("zero_star_steps", 0)
            raag_alpha = experimental_args.get("raag_alpha", 0.0)

            use_fresca = experimental_args.get("use_fresca", False)
            if use_fresca:
                fresca_scale_low = experimental_args.get("fresca_scale_low", 1.0)
                fresca_scale_high = experimental_args.get("fresca_scale_high", 1.25)
                fresca_freq_cutoff = experimental_args.get("fresca_freq_cutoff", 20)

            bidirectional_sampling = experimental_args.get("bidirectional_sampling", False)
            if bidirectional_sampling:
                sample_scheduler_flipped = copy.deepcopy(sample_scheduler)

        # Rotary positional embeddings (RoPE)

        # RoPE base freq scaling as used with CineScale
        ntk_alphas = [1.0, 1.0, 1.0]
        if isinstance(rope_function, dict):
            ntk_alphas = rope_function["ntk_scale_f"], rope_function["ntk_scale_h"], rope_function["ntk_scale_w"]
            rope_function = rope_function["rope_function"]

        # Stand-In
        standin_input = image_embeds.get("standin_input", None)
        if standin_input is not None:
            rope_function = "comfy" # only works with this currently

        freqs = None
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        if "default" in rope_function or bidirectional_sampling: # original RoPE
            d = transformer.dim // transformer.num_heads
            freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)
        elif "comfy" in rope_function: # comfy's rope
            transformer.rope_embedder.k = riflex_freq_index
            transformer.rope_embedder.num_frames = latent_video_length
           
        transformer.rope_func = rope_function
        for block in transformer.blocks:
            block.rope_func = rope_function
        if transformer.vace_layers is not None:
            for block in transformer.vace_blocks:
                block.rope_func = rope_function

        #region model pred
        def predict_with_cfg(z, cfg_scale, positive_embeds, negative_embeds, timestep, idx, image_cond=None, clip_fea=None, 
                             control_latents=None, vace_data=None, unianim_data=None, audio_proj=None, control_camera_latents=None, 
                             add_cond=None, cache_state=None, context_window=None, multitalk_audio_embeds=None, fantasy_portrait_input=None, reverse_time=False,
                             mtv_motion_tokens=None, s2v_audio_input=None, s2v_ref_motion=None, s2v_motion_frames=[1, 0], s2v_pose=None, text_embeds_override=None):
            nonlocal transformer
            if text_embeds_override is None:
                text_embeds_override = text_embeds
            z = z.to(dtype)
            autocast_enabled = ("fp8" in model["quantization"] and not transformer.patched_linear)
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype) if autocast_enabled else nullcontext():

                if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                    return z*0, None

                nonlocal patcher
                current_step_percentage = idx / len(timesteps)
                control_lora_enabled = False
                image_cond_input = None
                if control_embeds is not None and control_camera_latents is None:
                    if control_lora:
                        control_lora_enabled = True
                    else:
                        if ((control_start_percent <= current_step_percentage <= control_end_percent) or \
                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_start_percent)) and \
                            (control_latents is not None):
                            image_cond_input = torch.cat([control_latents.to(z), image_cond.to(z)])
                        else:
                            image_cond_input = torch.cat([torch.zeros_like(noise, device=device, dtype=dtype), image_cond.to(z)])
                        if fun_ref_image is not None:
                            fun_ref_input = fun_ref_image.to(z)
                        else:
                            fun_ref_input = torch.zeros_like(z, dtype=z.dtype)[:, 0].unsqueeze(1)

                    if control_lora:
                        if not control_start_percent <= current_step_percentage <= control_end_percent:
                            control_lora_enabled = False
                            if patcher.model.is_patched:
                                log.info("Unloading LoRA...")
                                patcher.unpatch_model(device)
                                patcher.model.is_patched = False
                        else:
                            image_cond_input = control_latents.to(z)
                            if not patcher.model.is_patched:
                                log.info("Loading LoRA...")
                                patcher = apply_lora(patcher, device, device, low_mem_load=False, control_lora=True)
                                patcher.model.is_patched = True
                                
                elif ATI_tracks is not None and ((ati_start_percent <= current_step_percentage <= ati_end_percent) or 
                              (ati_end_percent > 0 and idx == 0 and current_step_percentage >= ati_start_percent)):
                    image_cond_input = image_cond_ati.to(z)
                elif humo_image_cond is not None:
                    if context_window is not None:
                        image_cond_input = humo_image_cond[:, context_window].to(z)
                        humo_image_cond_neg_input = humo_image_cond_neg[:, context_window].to(z)
                        if humo_reference_count > 0:
                            image_cond_input[:, -humo_reference_count:] = humo_image_cond[:, -humo_reference_count:]
                            humo_image_cond_neg_input[:, -humo_reference_count:] = humo_image_cond_neg[:, -humo_reference_count:]
                    else:
                        image_cond_input = humo_image_cond.to(z)
                        humo_image_cond_neg_input = humo_image_cond_neg.to(z)
                elif image_cond is not None:
                    if reverse_time: # Flip the image condition
                        image_cond_input = torch.cat([
                            torch.flip(image_cond[:4], dims=[1]), 
                            torch.flip(image_cond[4:], dims=[1])
                        ]).to(z)
                    else:
                        image_cond_input = image_cond.to(z)

                if control_camera_latents is not None:
                    if (control_camera_start_percent <= current_step_percentage <= control_camera_end_percent) or \
                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_camera_start_percent):
                        control_camera_input = control_camera_latents.to(z)
                    else:
                        control_camera_input = None

                if recammaster is not None:
                    z = torch.cat([z, recam_latents.to(z)], dim=1)

                if mtv_input is not None:
                    if ((mtv_start_percent <= current_step_percentage <= mtv_end_percent) or \
                            (mtv_end_percent > 0 and idx == 0 and current_step_percentage >= mtv_start_percent)):
                        mtv_motion_tokens = mtv_motion_tokens.to(z)
                        mtv_motion_rotary_emb = motion_rotary_emb

                use_phantom = False
                phantom_ref = None
                if phantom_latents is not None:
                    if (phantom_start_percent <= current_step_percentage <= phantom_end_percent) or \
                        (phantom_end_percent > 0 and idx == 0 and current_step_percentage >= phantom_start_percent):
                        phantom_ref = phantom_latents.to(z)
                        use_phantom = True
                        if cache_state is not None and len(cache_state) != 3:
                            cache_state.append(None)

                if controlnet_latents is not None:
                    if (controlnet_start <= current_step_percentage < controlnet_end):
                        self.controlnet.to(device)
                        controlnet_states = self.controlnet(
                            hidden_states=z.unsqueeze(0).to(device, self.controlnet.dtype),
                            timestep=timestep,
                            encoder_hidden_states=positive_embeds[0].unsqueeze(0).to(device, self.controlnet.dtype),
                            attention_kwargs=None,
                            controlnet_states=controlnet_latents.to(device, self.controlnet.dtype),
                            return_dict=False,
                        )[0]
                        if isinstance(controlnet_states, (tuple, list)):
                            controlnet["controlnet_states"] = [x.to(z) for x in controlnet_states]
                        else:
                            controlnet["controlnet_states"] = controlnet_states.to(z)

                add_cond_input = None
                if add_cond is not None:
                    if (add_cond_start_percent <= current_step_percentage <= add_cond_end_percent) or \
                        (add_cond_end_percent > 0 and idx == 0 and current_step_percentage >= add_cond_start_percent):
                        add_cond_input = add_cond

                if minimax_latents is not None:
                    if context_window is not None:
                        z = torch.cat([z, minimax_latents[:, context_window], minimax_mask_latents[:, context_window]], dim=0)
                    else:
                        z = torch.cat([z, minimax_latents, minimax_mask_latents], dim=0)
                
                if not multitalk_sampling and multitalk_audio_embedding is not None:
                    audio_embedding = multitalk_audio_embedding
                    audio_embs = []
                    indices = (torch.arange(4 + 1) - 2) * 1
                    human_num = len(audio_embedding)
                    # split audio with window size
                    if context_window is None:
                        for human_idx in range(human_num):   
                            center_indices = torch.arange(
                                0,
                                latent_video_length * 4 + 1 if add_cond is not None else (latent_video_length-1) * 4 + 1,
                                1).unsqueeze(1) + indices.unsqueeze(0)
                            center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0] - 1)
                            audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                            audio_embs.append(audio_emb)
                    else:
                        for human_idx in range(human_num):
                            audio_start = context_window[0] * 4
                            audio_end = context_window[-1] * 4 + 1
                            #print("audio_start: ", audio_start, "audio_end: ", audio_end)
                            center_indices = torch.arange(audio_start, audio_end, 1).unsqueeze(1) + indices.unsqueeze(0)
                            center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0] - 1)
                            audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                            audio_embs.append(audio_emb)
                    multitalk_audio_input = torch.concat(audio_embs, dim=0).to(dtype)
                    
                elif multitalk_sampling and multitalk_audio_embeds is not None:
                    multitalk_audio_input = multitalk_audio_embeds
                
                if context_window is not None and pcd_data is not None and pcd_data["render_latent"].shape[2] != context_frames:
                    pcd_data_input = {"render_latent": pcd_data["render_latent"][:, :, context_window]}
                    for k in pcd_data:
                        if k != "render_latent":
                            pcd_data_input[k] = pcd_data[k]
                else:
                    pcd_data_input = pcd_data

                if s2v_pose is not None:
                    if not ((s2v_pose_start_percent <= current_step_percentage <= s2v_pose_end_percent) or \
                            (s2v_pose_end_percent > 0 and idx == 0 and current_step_percentage >= s2v_pose_start_percent)):
                        s2v_pose = None

                
                if humo_audio is not None and ((humo_start_percent <= current_step_percentage <= humo_end_percent) or \
                            (humo_end_percent > 0 and idx == 0 and current_step_percentage >= humo_start_percent)):
                    if context_window is None:
                        humo_audio_input = humo_audio
                        humo_audio_input_neg = humo_audio_neg if humo_audio_neg is not None else None
                    else:
                        humo_audio_input = humo_audio[context_window].to(z)
                        if humo_audio_neg is not None:
                            humo_audio_input_neg = humo_audio_neg[context_window].to(z)
                        else:
                            humo_audio_input_neg = None
                else:
                    humo_audio_input = humo_audio_input_neg = None

                base_params = {
                    'x': [z], # latent
                    'y': [image_cond_input] if image_cond_input is not None else None, # image cond
                    'clip_fea': clip_fea, # clip features
                    'seq_len': seq_len, # sequence length
                    'device': device, # main device
                    'freqs': freqs, # rope freqs
                    't': timestep, # current timestep
                    'is_uncond': False, # is unconditional
                    'current_step': idx, # current step
                    'current_step_percentage': current_step_percentage, # current step percentage
                    'last_step': len(timesteps) - 1 == idx, # is last step
                    'control_lora_enabled': control_lora_enabled, # control lora toggle for patch embed selection
                    'enhance_enabled': enhance_enabled, # enhance-a-video toggle
                    'camera_embed': camera_embed, # recammaster embedding
                    'unianim_data': unianim_data, # unianimate input
                    'fun_ref': fun_ref_input if fun_ref_image is not None else None, # Fun model reference latent
                    'fun_camera': control_camera_input if control_camera_latents is not None else None, # Fun model camera embed
                    'audio_proj': audio_proj if fantasytalking_embeds is not None else None, # FantasyTalking audio projection
                    'audio_scale': audio_scale, # FantasyTalking audio scale
                    "pcd_data": pcd_data_input, # Uni3C input
                    "controlnet": controlnet, # TheDenk's controlnet input
                    "add_cond": add_cond_input, # additional conditioning input
                    "nag_params": text_embeds_override.get("nag_params", {}), # normalized attention guidance
                    "nag_context": text_embeds_override.get("nag_prompt_embeds", None), # normalized attention guidance context
                    "multitalk_audio": multitalk_audio_input if multitalk_audio_embedding is not None else None, # Multi/InfiniteTalk audio input
                    "ref_target_masks": ref_target_masks if multitalk_audio_embedding is not None else None, # Multi/InfiniteTalk reference target masks
                    "inner_t": [shot_len] if shot_len else None, # inner timestep for EchoShot
                    "standin_input": standin_input, # Stand-in reference input
                    "fantasy_portrait_input": fantasy_portrait_input, # Fantasy portrait input
                    "phantom_ref": phantom_ref, # Phantom reference input
                    "reverse_time": reverse_time, # Reverse RoPE toggle
                    "ntk_alphas": ntk_alphas, # RoPE freq scaling values
                    "mtv_motion_tokens": mtv_motion_tokens if mtv_input is not None else None, # MTV-Crafter motion tokens
                    "mtv_motion_rotary_emb": mtv_motion_rotary_emb if mtv_input is not None else None, # MTV-Crafter RoPE
                    "mtv_strength": mtv_strength[idx] if mtv_input is not None else 1.0, # MTV-Crafter scaling
                    "mtv_freqs": mtv_freqs if mtv_input is not None else None, # MTV-Crafter extra RoPE freqs
                    "s2v_audio_input": s2v_audio_input, # official speech-to-video audio input
                    "s2v_ref_latent": s2v_ref_latent, # speech-to-video reference latent
                    "s2v_ref_motion": s2v_ref_motion, # speech-to-video reference motion latent
                    "s2v_audio_scale": s2v_audio_scale if s2v_audio_input is not None else 1.0, # speech-to-video audio scale
                    "s2v_pose": s2v_pose if s2v_pose is not None else None, # speech-to-video pose control
                    "s2v_motion_frames": s2v_motion_frames, # speech-to-video motion frames,
                    "humo_audio": humo_audio_input, # humo audio input
                    "humo_audio_scale": humo_audio_scale if humo_audio is not None else 1.0, # humo audio scale
                }

                batch_size = 1

                if not math.isclose(cfg_scale, 1.0):
                    if negative_embeds is None:
                        raise ValueError("Negative embeddings must be provided for CFG scale > 1.0")
                    if len(positive_embeds) > 1:
                        negative_embeds = negative_embeds * len(positive_embeds)

                try:
                    if not batched_cfg:
                        #conditional (positive) pass
                        noise_pred_cond, cache_state_cond = transformer(
                            context=positive_embeds,
                            pred_id=cache_state[0] if cache_state else None,
                            vace_data=vace_data, attn_cond=attn_cond,
                            **base_params
                        )
                        noise_pred_cond = noise_pred_cond[0]
                        if math.isclose(cfg_scale, 1.0):
                            if use_fresca:
                                noise_pred_cond = fourier_filter(noise_pred_cond, fresca_scale_low, fresca_scale_high, fresca_freq_cutoff)
                            return noise_pred_cond, [cache_state_cond]
                        
                        #unconditional (negative) pass
                        base_params['is_uncond'] = True
                        base_params['clip_fea'] = clip_fea_neg if clip_fea_neg is not None else clip_fea
                        if humo_audio_input_neg is not None:
                            base_params['humo_audio'] = humo_audio_input_neg
                        
                        noise_pred_uncond, cache_state_uncond = transformer(
                            context=negative_embeds if humo_audio_input_neg is None else positive_embeds, #ti
                            pred_id=cache_state[1] if cache_state else None,
                            vace_data=vace_data, attn_cond=attn_cond_neg,
                            **base_params)
                        noise_pred_uncond = noise_pred_uncond[0]

                        # HuMo
                        if humo_audio_input_neg is not None and not math.isclose(humo_audio_cfg_scale[idx], 1.0):
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            if t > 980 and humo_image_cond_neg_input is not None: # use image cond for first timesteps
                                base_params['y'] = [humo_image_cond_neg_input]
                            
                            noise_pred_humo_audio_uncond, cache_state_humo = transformer(
                            context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                            **base_params)

                            noise_pred = (noise_pred_uncond + humo_audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_humo_audio_uncond[0]) 
                                          + (cfg_scale - 2.0) * (noise_pred_humo_audio_uncond[0] - noise_pred_uncond))
                            return noise_pred, [cache_state_cond, cache_state_uncond, cache_state_humo]
                        #phantom
                        if use_phantom and not math.isclose(phantom_cfg_scale[idx], 1.0):
                            noise_pred_phantom, cache_state_phantom = transformer(
                            context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                            **base_params)

                            noise_pred = (noise_pred_uncond + phantom_cfg_scale[idx] * (noise_pred_phantom[0] - noise_pred_uncond)
                                          + cfg_scale * (noise_pred_cond - noise_pred_phantom[0]))
                            return noise_pred, [cache_state_cond, cache_state_uncond, cache_state_phantom]
                        #audio cfg (fantasytalking and multitalk)
                        if (fantasytalking_embeds is not None or multitalk_audio_embedding is not None):
                            if not math.isclose(audio_cfg_scale[idx], 1.0):
                                if cache_state is not None and len(cache_state) != 3:
                                    cache_state.append(None)
                                
                                # Set audio parameters to None/zeros based on type
                                if fantasytalking_embeds is not None:
                                    base_params['audio_proj'] = None
                                    audio_context = positive_embeds
                                else:  # multitalk
                                    base_params['multitalk_audio'] = torch.zeros_like(multitalk_audio_input)[-1:]
                                    audio_context = negative_embeds
                                
                                noise_pred_no_audio, cache_state_audio = transformer(
                                    context=audio_context, is_uncond=False,
                                    pred_id=cache_state[2] if cache_state else None,
                                    vace_data=vace_data,
                                    **base_params)

                                noise_pred = (noise_pred_uncond
                                    + cfg_scale * (noise_pred_no_audio[0] - noise_pred_uncond)
                                    + audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_no_audio[0]))
                                return noise_pred, [cache_state_cond, cache_state_uncond, cache_state_audio]

                    #batched
                    else:
                        base_params['z'] = [z] * 2
                        base_params['y'] = [image_cond_input] * 2 if image_cond_input is not None else None
                        base_params['clip_fea'] = torch.cat([clip_fea, clip_fea], dim=0)
                        cache_state_uncond = None
                        [noise_pred_cond, noise_pred_uncond], cache_state_cond = transformer(
                            context=positive_embeds + negative_embeds, is_uncond=False,
                            pred_id=cache_state[0] if cache_state else None,
                            **base_params
                        )
                except Exception as e:
                    log.error(f"Error during model prediction: {e}")
                    if force_offload:
                        if not model["auto_cpu_offload"]:
                            offload_transformer(transformer)
                    raise e

                #https://github.com/WeichenFan/CFG-Zero-star/
                alpha = 1.0
                if use_cfg_zero_star:
                    alpha = optimized_scale(
                        noise_pred_cond.view(batch_size, -1),
                        noise_pred_uncond.view(batch_size, -1)
                    ).view(batch_size, 1, 1, 1)
                    
                noise_pred_uncond_scaled = noise_pred_uncond * alpha

                if use_tangential:
                    noise_pred_uncond_scaled = tangential_projection(noise_pred_cond, noise_pred_uncond_scaled)

                # RAAG (RATIO-aware Adaptive Guidance)
                if raag_alpha > 0.0:
                    cfg_scale = get_raag_guidance(noise_pred_cond, noise_pred_uncond_scaled, cfg_scale, raag_alpha)
                    log.info(f"RAAG modified cfg: {cfg_scale}")

                #https://github.com/WikiChao/FreSca
                if use_fresca:
                    filtered_cond = fourier_filter(noise_pred_cond - noise_pred_uncond, fresca_scale_low, fresca_scale_high, fresca_freq_cutoff)
                    noise_pred = noise_pred_uncond_scaled + cfg_scale * filtered_cond * alpha
                else:
                    noise_pred = noise_pred_uncond_scaled + cfg_scale * (noise_pred_cond - noise_pred_uncond_scaled)
                del noise_pred_uncond_scaled, noise_pred_cond, noise_pred_uncond
                
                return noise_pred, [cache_state_cond, cache_state_uncond]

        if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]: #default for latent2rgb
            from latent_preview import prepare_callback
        else:
            from .latent_preview import prepare_callback #custom for tiny VAE previews
        callback = prepare_callback(patcher, len(timesteps))

        if not multitalk_sampling and not framepack:
            log.info(f"Input sequence length: {seq_len}")
            log.info(f"Sampling {(latent_video_length-1) * 4 + 1} frames at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")

        intermediate_device = device

        # Differential diffusion prep
        masks = None
        if not multitalk_sampling and samples is not None and noise_mask is not None:
            thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
            thresholds = thresholds.reshape(-1, 1, 1, 1, 1).to(device)
            masks = (1-noise_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device)) > thresholds

        latent_shift_loop = False
        if loop_args is not None:
            latent_shift_loop = is_looped = True
            latent_skip = loop_args["shift_skip"]
            latent_shift_start_percent = loop_args["start_percent"]
            latent_shift_end_percent = loop_args["end_percent"]
            shift_idx = 0

        #clear memory before sampling
        mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
            #torch.cuda.memory._record_memory_history(max_entries=100000)
        except:
            pass

        # Main sampling loop with FreeInit iterations
        iterations = freeinit_args.get("freeinit_num_iters", 3) if freeinit_args is not None else 1
        current_latent = latent
        
        for iter_idx in range(iterations): 

            # FreeInit noise reinitialization (after first iteration)
            if freeinit_args is not None and iter_idx > 0:
                # restart scheduler for each iteration
                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas, keep_number_of_steps=True)

                # Re-apply start_step and end_step logic to timesteps and sigmas
                if end_step != -1:
                    timesteps = timesteps[:end_step]
                    sample_scheduler.sigmas = sample_scheduler.sigmas[:end_step+1]
                if start_step > 0:
                    timesteps = timesteps[start_step:]
                    sample_scheduler.sigmas = sample_scheduler.sigmas[start_step:]
                if hasattr(sample_scheduler, 'timesteps'):
                    sample_scheduler.timesteps = timesteps

                # Diffuse current latent to t=999
                diffuse_timesteps = torch.full((noise.shape[0],), 999, device=device, dtype=torch.long)
                z_T = add_noise(
                    current_latent.to(device),
                    initial_noise_saved.to(device),
                    diffuse_timesteps
                )

                # Generate new random noise
                z_rand = torch.randn(z_T.shape, dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))
                # Apply frequency mixing
                current_latent = (freq_mix_3d(z_T.to(torch.float32), z_rand.to(device), LPF=freq_filter)).to(dtype)

            # Store initial noise for first iteration
            if freeinit_args is not None and iter_idx == 0:
                initial_noise_saved = current_latent.detach().clone()
                if samples is not None and input_samples is not None:
                    current_latent = input_samples.to(device)
                    continue
                else:
                    current_latent = latent
            
            # Reset per-iteration states
            self.cache_state = [None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []
            if context_options is not None:
                self.window_tracker = WindowTracker(verbose=context_options["verbose"])
            
            # Set latent for denoising
            latent = current_latent

            if is_pusa and all_indices:
                pusa_noisy_steps = image_embeds.get("pusa_noisy_steps", -1)
                if pusa_noisy_steps == -1:
                    pusa_noisy_steps = len(timesteps)
            try:
                pbar = ProgressBar(len(timesteps))
                #region main loop start
                for idx, t in enumerate(tqdm(timesteps, disable=multitalk_sampling)):
                    if flowedit_args is not None:
                        if idx < skip_steps:
                            continue

                    if bidirectional_sampling:
                        latent_flipped = torch.flip(latent, dims=[1])
                        latent_model_input_flipped = latent_flipped.to(device)

                    #InfiniteTalk first frame handling
                    if (extra_latents is not None
                        and not multitalk_sampling
                        and transformer.multitalk_model_type=="InfiniteTalk"):
                        for entry in extra_latents:
                            add_index = entry["index"]
                            num_extra_frames = entry["samples"].shape[2]
                            latent[:, add_index:add_index+num_extra_frames] = entry["samples"].to(latent)

                    latent_model_input = latent.to(device)

                    current_step_percentage = idx / len(timesteps)

                    timestep = torch.tensor([t]).to(device)
                    if is_pusa or (is_5b and all_indices):
                        orig_timestep = timestep
                        timestep = timestep.unsqueeze(1).repeat(1, latent_video_length)
                        if extra_latents is not None:
                            if all_indices and noise_multipliers is not None:
                                if is_pusa:
                                    scheduler_step_args["cond_frame_latent_indices"] = all_indices
                                    scheduler_step_args["noise_multipliers"] = noise_multipliers
                                for latent_idx in all_indices:
                                    timestep[:, latent_idx] = timestep[:, latent_idx] * noise_multipliers[latent_idx]
                                    # add noise for conditioning frames if multiplier > 0
                                    if idx < pusa_noisy_steps and noise_multipliers[latent_idx] > 0:
                                        latent_size = (1, latent.shape[0], latent.shape[1], latent.shape[2], latent.shape[3])
                                        noise_for_cond = torch.randn(latent_size, generator=seed_g, device=torch.device("cpu"))
                                        timestep_cond = torch.ones_like(timestep) * timestep.max()
                                        if is_pusa:
                                            latent[:, latent_idx:latent_idx+1] = sample_scheduler.add_noise_for_conditioning_frames(
                                                latent[:, latent_idx:latent_idx+1].to(device),
                                                noise_for_cond[:, :, latent_idx:latent_idx+1].to(device),
                                                timestep_cond[:, latent_idx:latent_idx+1].to(device),
                                                noise_multiplier=noise_multipliers[latent_idx])
                            else:
                                timestep[:, all_indices] = 0
                            #print("timestep: ", timestep)

                    ### latent shift
                    if latent_shift_loop:
                        if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                            latent_model_input = torch.cat([latent_model_input[:, shift_idx:]] + [latent_model_input[:, :shift_idx]], dim=1)

                    #enhance-a-video
                    enhance_enabled = False
                    if feta_args is not None and feta_start_percent <= current_step_percentage <= feta_end_percent:
                        enhance_enabled = True                    

                    #flow-edit
                    if flowedit_args is not None:
                        sigma = t / 1000.0
                        sigma_prev = (timesteps[idx + 1] if idx < len(timesteps) - 1 else timesteps[-1]) / 1000.0
                        noise = torch.randn(x_init.shape, generator=seed_g, device=torch.device("cpu"))
                        if idx < len(timesteps) - drift_steps:
                            cfg = drift_cfg
                        
                        zt_src = (1-sigma) * x_init + sigma * noise.to(t)
                        zt_tgt = x_tgt + zt_src - x_init

                        #source
                        if idx < len(timesteps) - drift_steps:
                            if context_options is not None:
                                counter = torch.zeros_like(zt_src, device=intermediate_device)
                                vt_src = torch.zeros_like(zt_src, device=intermediate_device)
                                context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                                for c in context_queue:
                                    window_id = self.window_tracker.get_window_id(c)

                                    if cache_args is not None:
                                        current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                                    else:
                                        current_teacache = None

                                    prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                                    if context_options["verbose"]:
                                        log.info(f"Prompt index: {prompt_index}")

                                    if len(source_embeds["prompt_embeds"]) > 1:
                                        positive = source_embeds["prompt_embeds"][prompt_index]
                                    else:
                                        positive = source_embeds["prompt_embeds"]

                                    partial_img_emb = None
                                    if source_image_cond is not None:
                                        partial_img_emb = source_image_cond[:, c, :, :]
                                        partial_img_emb[:, 0, :, :] = source_image_cond[:, 0, :, :].to(intermediate_device)

                                    partial_zt_src = zt_src[:, c, :, :]
                                    vt_src_context, new_teacache = predict_with_cfg(
                                        partial_zt_src, cfg[idx], 
                                        positive, source_embeds["negative_prompt_embeds"],
                                        timestep, idx, partial_img_emb, control_latents,
                                        source_clip_fea, current_teacache)
                                    
                                    if cache_args is not None:
                                        self.window_tracker.cache_states[window_id] = new_teacache

                                    window_mask = create_window_mask(vt_src_context, c, latent_video_length, context_overlap)
                                    vt_src[:, c, :, :] += vt_src_context * window_mask
                                    counter[:, c, :, :] += window_mask
                                vt_src /= counter
                            else:
                                vt_src, self.cache_state_source = predict_with_cfg(
                                    zt_src, cfg[idx], 
                                    source_embeds["prompt_embeds"], 
                                    source_embeds["negative_prompt_embeds"],
                                    timestep, idx, source_image_cond, 
                                    source_clip_fea, control_latents,
                                    cache_state=self.cache_state_source)
                        else:
                            if idx == len(timesteps) - drift_steps:
                                x_tgt = zt_tgt
                            zt_tgt = x_tgt
                            vt_src = 0
                        #target
                        if context_options is not None:
                            counter = torch.zeros_like(zt_tgt, device=intermediate_device)
                            vt_tgt = torch.zeros_like(zt_tgt, device=intermediate_device)
                            context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                            for c in context_queue:
                                window_id = self.window_tracker.get_window_id(c)

                                if cache_args is not None:
                                    current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                                else:
                                    current_teacache = None

                                prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                                if context_options["verbose"]:
                                    log.info(f"Prompt index: {prompt_index}")
                            
                                if len(text_embeds["prompt_embeds"]) > 1:
                                    positive = text_embeds["prompt_embeds"][prompt_index]
                                else:
                                    positive = text_embeds["prompt_embeds"]
                                
                                partial_img_emb = None
                                partial_control_latents = None
                                if image_cond is not None:
                                    partial_img_emb = image_cond[:, c, :, :]
                                    partial_img_emb[:, 0, :, :] = image_cond[:, 0, :, :].to(intermediate_device)
                                if control_latents is not None:
                                    partial_control_latents = control_latents[:, c, :, :]

                                partial_zt_tgt = zt_tgt[:, c, :, :]
                                vt_tgt_context, new_teacache = predict_with_cfg(
                                    partial_zt_tgt, cfg[idx], 
                                    positive, text_embeds["negative_prompt_embeds"],
                                    timestep, idx, partial_img_emb, partial_control_latents,
                                    clip_fea, current_teacache)
                                
                                if cache_args is not None:
                                    self.window_tracker.cache_states[window_id] = new_teacache
                                
                                window_mask = create_window_mask(vt_tgt_context, c, latent_video_length, context_overlap)
                                vt_tgt[:, c, :, :] += vt_tgt_context * window_mask
                                counter[:, c, :, :] += window_mask
                            vt_tgt /= counter
                        else:
                            vt_tgt, self.cache_state = predict_with_cfg(
                                zt_tgt, cfg[idx], 
                                text_embeds["prompt_embeds"], 
                                text_embeds["negative_prompt_embeds"], 
                                timestep, idx, image_cond, clip_fea, control_latents,
                                cache_state=self.cache_state)
                        v_delta = vt_tgt - vt_src
                        x_tgt = x_tgt.to(torch.float32)
                        v_delta = v_delta.to(torch.float32)
                        x_tgt = x_tgt + (sigma_prev - sigma) * v_delta
                        x0 = x_tgt
                    #region context windowing
                    elif context_options is not None:
                        print("!!!!!Using context windowing")
                        counter = torch.zeros_like(latent_model_input, device=intermediate_device)
                        noise_pred = torch.zeros_like(latent_model_input, device=intermediate_device)
                        context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                        fraction_per_context = 1.0 / len(context_queue)
                        context_pbar = ProgressBar(steps)
                        step_start_progress = idx

                        # Validate all context windows before processing
                        max_idx = latent_model_input.shape[1] if latent_model_input.ndim > 1 else 0
                        for window_indices in context_queue:
                            if not all(0 <= idx < max_idx for idx in window_indices):
                                raise ValueError(f"Invalid context window indices {window_indices} for latent_model_input with shape {latent_model_input.shape}")

                        # rolling extra latent support (5B models) - carries last frame x0 to next window's first frame
                        rolling_extra_enabled = context_options.get("rolling_extra", False)
                        rolling_extra_latent = None  # shape (C,1,H,W)

                        for i, c in enumerate(context_queue):
                            print(f"!!!!! context windowing {i}")
                            window_id = self.window_tracker.get_window_id(c)
                            
                            if cache_args is not None:
                                current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                            else:
                                current_teacache = None

                            prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                            if context_options["verbose"]:
                                log.info(f"Prompt index: {prompt_index}")
                            
                            # Use the appropriate prompt for this section
                            if len(text_embeds["prompt_embeds"]) > 1:
                                positive = [text_embeds["prompt_embeds"][prompt_index]]
                            else:
                                positive = text_embeds["prompt_embeds"]

                            partial_img_emb = None
                            partial_control_latents = None
                            if image_cond is not None:
                                partial_img_emb = image_cond[:, c]
                                if c[0] != 0 and context_reference_latent is not None:
                                    if context_reference_latent.shape[0] == 1: #only single extra init latent
                                        new_init_image = context_reference_latent[0, :, 0].to(intermediate_device)
                                        # Concatenate the first 4 channels of partial_img_emb with new_init_image to match the required shape
                                        partial_img_emb[:, 0] = torch.cat([image_cond[:4, 0], new_init_image], dim=0)
                                    elif context_reference_latent.shape[0] > 1:
                                        num_extra_inits = context_reference_latent.shape[0]
                                        section_size = (latent_video_length / num_extra_inits)
                                        extra_init_index = min(int(max(c) / section_size), num_extra_inits - 1)
                                        if context_options["verbose"]:
                                            log.info(f"extra init image index: {extra_init_index}")
                                        new_init_image = context_reference_latent[extra_init_index, :, 0].to(intermediate_device)
                                        partial_img_emb[:, 0] = torch.cat([image_cond[:4, 0], new_init_image], dim=0)
                                else:
                                    new_init_image = image_cond[:, 0].to(intermediate_device)
                                    partial_img_emb[:, 0] = new_init_image

                                if control_latents is not None:
                                    partial_control_latents = control_latents[:, c]
                            
                            partial_control_camera_latents = None
                            if control_camera_latents is not None:
                                partial_control_camera_latents = control_camera_latents[:, :, c]
                            
                            partial_vace_context = None
                            if vace_data is not None:
                                window_vace_data = []
                                for vace_entry in vace_data:
                                    partial_context = vace_entry["context"][0][:, c]
                                    if has_ref:
                                        partial_context[:, 0] = vace_entry["context"][0][:, 0]
                                    
                                    window_vace_data.append({
                                        "context": [partial_context], 
                                        "scale": vace_entry["scale"],
                                        "start": vace_entry["start"], 
                                        "end": vace_entry["end"],
                                        "seq_len": vace_entry["seq_len"]
                                    })
                                
                                partial_vace_context = window_vace_data

                            partial_audio_proj = None
                            if fantasytalking_embeds is not None:
                                partial_audio_proj = audio_proj[:, c]

                            partial_fantasy_portrait_input = None
                            if fantasy_portrait_input is not None:
                                partial_fantasy_portrait_input = fantasy_portrait_input.copy()
                                partial_fantasy_portrait_input["adapter_proj"] = fantasy_portrait_input["adapter_proj"][:, c]

                            print(f"setting default partial_latent_model_input for index {i}")
                            partial_latent_model_input = latent_model_input[:, c]
                            if latents_to_insert is not None and c[0] != 0:
                                print(f"Setting partial_latent_model_input to latents_to_insert for index {i}")
                                partial_latent_model_input[:, :1] = latents_to_insert
                            # rolling latent injection (only after first window)
                            if rolling_extra_enabled and is_5b and rolling_extra_latent is not None and c[0] != 0:
                                print(f"Injecting rolling extra latent for index {i}")
                                partial_latent_model_input[:, :1] = rolling_extra_latent.to(partial_latent_model_input)

                            partial_unianim_data = None
                            if unianim_data is not None:
                                partial_dwpose = dwpose_data[:, :, c]
                                partial_unianim_data = {
                                    "dwpose": partial_dwpose,
                                    "random_ref": unianim_data["random_ref"],
                                    "strength": unianimate_poses["strength"],
                                    "start_percent": unianimate_poses["start_percent"],
                                    "end_percent": unianimate_poses["end_percent"]
                                }

                            partial_mtv_motion_tokens = None
                            if mtv_input is not None:
                                start_token_index = c[0] * 24
                                end_token_index = (c[-1] + 1) * 24
                                partial_mtv_motion_tokens = mtv_motion_tokens[:, start_token_index:end_token_index, :]
                                if context_options["verbose"]:
                                    log.info(f"context window: {c}")
                                    log.info(f"motion_token_indices: {start_token_index}-{end_token_index}")

                            partial_s2v_audio_input = None
                            if s2v_audio_input is not None:
                                indices = (torch.arange(4 + 1) - 2) * 1
                                audio_start = c[0] * 4
                                audio_end = c[-1] * 4 + 1
                                center_indices = torch.arange(audio_start, audio_end, 1)
                                center_indices = torch.clamp(center_indices, min=0, max=s2v_audio_input.shape[-1] - 1)
                                partial_s2v_audio_input = s2v_audio_input[..., center_indices]

                            partial_s2v_pose = None
                            if s2v_pose is not None:
                                partial_s2v_pose = s2v_pose[:, :, c].to(device, dtype)

                            partial_add_cond = None
                            if add_cond is not None:
                                partial_add_cond = add_cond[:, :, c].to(device, dtype)

                            if len(timestep.shape) != 1:
                                partial_timestep = timestep[:, c]
                                partial_timestep[:, :1] = 0
                            else:
                                partial_timestep = timestep
                            #print("Partial timestep:", partial_timestep)

                            noise_pred_context, new_teacache = predict_with_cfg(
                                partial_latent_model_input, 
                                cfg[idx], positive, 
                                text_embeds["negative_prompt_embeds"], 
                                partial_timestep, idx, partial_img_emb, clip_fea, partial_control_latents, partial_vace_context, partial_unianim_data,partial_audio_proj,
                                partial_control_camera_latents, partial_add_cond, current_teacache, context_window=c, fantasy_portrait_input=partial_fantasy_portrait_input,
                                mtv_motion_tokens=partial_mtv_motion_tokens, s2v_audio_input=partial_s2v_audio_input, s2v_motion_frames=[1, 0], s2v_pose=partial_s2v_pose)

                            # capture rolling extra latent (x0 estimate of last frame) for next window
                            if rolling_extra_enabled and is_5b:
                                print("Capturing rolling extra latent")
                                with torch.no_grad():
                                    try:
                                        if isinstance(partial_timestep, torch.Tensor):
                                            if partial_timestep.ndim > 1:
                                                ts_last = partial_timestep[:, -1].float()
                                            else:
                                                ts_last = partial_timestep.float().view(1)
                                        else:
                                            ts_last = torch.tensor([float(t)], device=partial_latent_model_input.device)
                                        # reshape for broadcast
                                        while ts_last.ndim < noise_pred_context[:, -1:].ndim:
                                            ts_last = ts_last.view(ts_last.shape[0], *([1]*(noise_pred_context[:, -1:].ndim-1)))
                                        rolling_extra_latent = (partial_latent_model_input[:, -1:].to(noise_pred_context.dtype) - noise_pred_context[:, -1:] * (ts_last / 1000.0)).detach()
                                        print("Captured rolling extra latent")
                                    except Exception as e:
                                        log.warning(f"Failed to compute rolling extra latent: {e}")

                            if cache_args is not None:
                                self.window_tracker.cache_states[window_id] = new_teacache

                            window_mask = create_window_mask(noise_pred_context, c, latent_video_length, context_overlap, looped=is_looped, window_type=context_options["fuse_method"])                    
                            noise_pred[:, c] += noise_pred_context * window_mask
                            counter[:, c] += window_mask
                            context_pbar.update_absolute(step_start_progress + (i + 1) * fraction_per_context, len(timesteps))
                        noise_pred /= counter
                    #region multitalk
                    elif multitalk_sampling:
                        mode = image_embeds.get("multitalk_mode", "multitalk")
                        if mode == "auto":
                            mode = transformer.multitalk_model_type.lower()
                        log.info(f"Multitalk mode: {mode}")
                        cond_frame = None
                        offload = image_embeds.get("force_offload", False)
                        offloaded = False
                        tiled_vae = image_embeds.get("tiled_vae", False)
                        frame_num = clip_length = image_embeds.get("num_frames", 81)
                        vae = image_embeds.get("vae", None)
                        clip_embeds = image_embeds.get("clip_context", None)
                        if clip_embeds is not None:
                            clip_embeds = clip_embeds.to(dtype)
                        colormatch = image_embeds.get("colormatch", "disabled")
                        motion_frame = image_embeds.get("motion_frame", 25)
                        target_w = image_embeds.get("target_w", None)
                        target_h = image_embeds.get("target_h", None)
                        original_images = cond_image = image_embeds.get("multitalk_start_image", None)
                        if original_images is None:
                            original_images = torch.zeros([noise.shape[0], 1, target_h, target_w], device=device)

                        output_path = image_embeds.get("output_path", "")
                        img_counter = 0

                        if len(multitalk_embeds['audio_features'])==2 and (multitalk_embeds['ref_target_masks'] is None):
                            face_scale = 0.1
                            x_min, x_max = int(target_h * face_scale), int(target_h * (1 - face_scale))
                            lefty_min, lefty_max = int((target_w//2) * face_scale), int((target_w//2) * (1 - face_scale))
                            righty_min, righty_max = int((target_w//2) * face_scale + (target_w//2)), int((target_w//2) * (1 - face_scale) + (target_w//2))
                            human_mask1, human_mask2 = (torch.zeros([target_h, target_w]) for _ in range(2))
                            human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                            human_mask2[x_min:x_max, righty_min:righty_max] = 1
                            background_mask = torch.where((human_mask1 + human_mask2) > 0, torch.tensor(0), torch.tensor(1))
                            human_masks = [human_mask1, human_mask2, background_mask]
                            ref_target_masks = torch.stack(human_masks, dim=0)
                            multitalk_embeds['ref_target_masks'] = ref_target_masks

                        gen_video_list = []
                        is_first_clip = True
                        arrive_last_frame = False
                        cur_motion_frames_num = 1
                        audio_start_idx = iteration_count = step_iteration_count= 0
                        audio_end_idx = audio_start_idx + clip_length
                        indices = (torch.arange(4 + 1) - 2) * 1
                        current_condframe_index = 0

                        audio_embedding = multitalk_audio_embedding
                        human_num = len(audio_embedding)
                        audio_embs = None
                        
                        pcd_data = pcd_data_input = None
                        if uni3c_embeds is not None:
                            transformer.controlnet = uni3c_embeds["controlnet"]
                            pcd_data = {
                                "render_latent": uni3c_embeds["render_latent"],
                                "render_mask": uni3c_embeds["render_mask"],
                                "camera_embedding": uni3c_embeds["camera_embedding"],
                                "controlnet_weight": uni3c_embeds["controlnet_weight"],
                                "start": uni3c_embeds["start"],
                                "end": uni3c_embeds["end"],
                            }

                        total_frames = max_multitalk_frames
                        if total_frames is None:
                            total_frames = len(audio_embedding[0])
                        encoded_silence = None
                       
                        try:
                            silence_path = os.path.join(script_directory, "multitalk", "encoded_silence.safetensors")
                            encoded_silence = load_torch_file(silence_path)["audio_emb"].to(dtype)
                        except:
                             log.warning("No encoded silence file found, padding with end of audio embedding instead.")

                        estimated_iterations = total_frames // (frame_num - motion_frame) + 1
                        callback = prepare_callback(patcher, estimated_iterations)

                        if frame_num >= total_frames:
                            arrive_last_frame = True
                            estimated_iterations = 1

                        log.info(f"Sampling {total_frames} frames in {estimated_iterations} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")
                        #region Multi-talk Loop Start
                        while True: # start video generation iteratively
                            self.cache_state = [None, None]

                            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)
                            if mode == "infinitetalk":
                                cond_image = original_images[:, :, current_condframe_index:current_condframe_index+1] if cond_image is not None else None
                            if multitalk_embeds is not None:
                                audio_embs = []
                                # split audio with window size
                                for human_idx in range(human_num):   
                                    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
                                    center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0]-1)
                                    audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                                    audio_embs.append(audio_emb)
                                audio_embs = torch.concat(audio_embs, dim=0).to(dtype)

                            h, w = (cond_image.shape[-2], cond_image.shape[-1]) if cond_image is not None else (target_h, target_w)
                            lat_h, lat_w = h // VAE_STRIDE[1], w // VAE_STRIDE[2]
                            seq_len = ((frame_num - 1) // VAE_STRIDE[0] + 1) * lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
                            latent_frame_num = (frame_num - 1) // 4 + 1

                            noise = torch.randn(
                                16, latent_frame_num,
                                lat_h, lat_w, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g).to(device)
                            
                            # Calculate the correct latent slice based on current iteration
                            if is_first_clip:
                                latent_start_idx = 0
                                latent_end_idx = noise.shape[1]
                            else:
                                new_frames_per_iteration = frame_num - motion_frame
                                new_latent_frames_per_iteration = ((new_frames_per_iteration - 1) // 4 + 1)
                                latent_start_idx = iteration_count * new_latent_frames_per_iteration
                                latent_end_idx = latent_start_idx + noise.shape[1]
                            
                            if samples is not None:
                                input_samples = samples["samples"].squeeze(0).to(noise)
                                # Check if we have enough frames in input_samples
                                if latent_end_idx > input_samples.shape[1]:
                                    # We need more frames than available - pad the input_samples at the end
                                    pad_length = latent_end_idx - input_samples.shape[1]
                                    last_frame = input_samples[:, -1:].repeat(1, pad_length, 1, 1)
                                    input_samples = torch.cat([input_samples, last_frame], dim=1)
                                input_samples = input_samples[:, latent_start_idx:latent_end_idx]
                                if noise_mask is not None:
                                    original_image = input_samples.to(device)

                                assert input_samples.shape[1] == noise.shape[1], f"Slice mismatch: {input_samples.shape[1]} vs {noise.shape[1]}"
                                
                                if add_noise_to_samples:
                                    latent_timestep = timesteps[0]
                                    noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples
                                else:
                                    noise = input_samples

                                # diff diff prep
                                noise_mask = samples.get("noise_mask", None)
                                if noise_mask is not None:
                                    if len(noise_mask.shape) == 4:
                                        noise_mask = noise_mask.squeeze(1)
                                    if noise_mask.shape[0] < noise.shape[1]:
                                        noise_mask = noise_mask.repeat(noise.shape[1] // noise_mask.shape[0], 1, 1)
                                    else:
                                        noise_mask = noise_mask[latent_start_idx:latent_end_idx]
                                    noise_mask = torch.nn.functional.interpolate(
                                        noise_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                                        size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                                        mode='trilinear',
                                        align_corners=False
                                    ).repeat(1, noise.shape[0], 1, 1, 1)

                                    thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
                                    thresholds = thresholds.reshape(-1, 1, 1, 1, 1).to(device)
                                    masks = (1-noise_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device)) > thresholds

                            # zero padding and vae encode for img cond
                            if cond_image is not None:
                                video_frames = torch.zeros(1, 3, frame_num-cond_image.shape[2], target_h, target_w, device=device, dtype=vae.dtype)
                                padding_frames_pixels_values = torch.concat([cond_image.to(device, vae.dtype), video_frames], dim=2)

                                # encode
                                vae.to(device)
                                y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
                                
                                if mode == "multitalk":
                                    latent_motion_frames = y[:, :cur_motion_frames_latent_num] # C T H W
                                else:
                                    cond_ = cond_image if is_first_clip else cond_frame
                                    latent_motion_frames = vae.encode(cond_.to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
                                vae.model.clear_cache()
                                vae.to(offload_device)

                                #motion_frame_index = cur_motion_frames_latent_num if mode == "infinitetalk" else 1
                                msk = torch.zeros(4, latent_frame_num, lat_h, lat_w, device=device, dtype=dtype)
                                msk[:, :1] = 1
                                y = torch.cat([msk, y]) # 4+C T H W
                                mm.soft_empty_cache()
                            else:
                                y = None
                                latent_motion_frames = noise[:, :1]

                            if scheduler == "multitalk":
                                timesteps = list(np.linspace(1000, 1, steps, dtype=np.float32))
                                timesteps.append(0.)
                                timesteps = [torch.tensor([t], device=device) for t in timesteps]
                                timesteps = [timestep_transform(t, shift=shift, num_timesteps=1000) for t in timesteps]
                            else:
                                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas, keep_number_of_steps=True)
                                timesteps = [torch.tensor([float(t)], device=device) for t in timesteps] + [torch.tensor([0.], device=device)]
                            
                            # sample videos
                            latent = noise

                            # injecting motion frames
                            if not is_first_clip and mode == "multitalk":
                                latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                                motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                                add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                                latent[:, :add_latent.shape[1]] = add_latent

                            if offloaded:
                                # Load weights
                                if transformer.patched_linear and gguf_reader is None:
                                    load_weights(patcher.model.diffusion_model, patcher.model["sd"], weight_dtype, base_dtype=dtype, transformer_load_device=device, block_swap_args=block_swap_args)
                                elif gguf_reader is not None: #handle GGUF
                                    load_weights(transformer, patcher.model["sd"], base_dtype=dtype, transformer_load_device=device, patcher=patcher, gguf=True, reader=gguf_reader, block_swap_args=block_swap_args)

                                #blockswap init
                                if not transformer.patched_linear:
                                    if block_swap_args is not None:
                                        for name, param in transformer.named_parameters():
                                            if "block" not in name:
                                                param.data = param.data.to(device)
                                            if "control_adapter" in name:
                                                param.data = param.data.to(device)
                                            elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                                                param.data = param.data.to(offload_device)
                                            elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                                                param.data = param.data.to(offload_device)

                                        transformer.block_swap(
                                            block_swap_args["blocks_to_swap"] - 1 ,
                                            block_swap_args["offload_txt_emb"],
                                            block_swap_args["offload_img_emb"],
                                            vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
                                        )
                                    elif model["auto_cpu_offload"]:
                                        for module in transformer.modules():
                                            if hasattr(module, "offload"):
                                                module.offload()
                                            if hasattr(module, "onload"):
                                                module.onload()
                                        for block in transformer.blocks:
                                            block.modulation = torch.nn.Parameter(block.modulation.to(device))
                                        transformer.head.modulation = torch.nn.Parameter(transformer.head.modulation.to(device))
                                    else:
                                        transformer.to(device)

                            # Use the appropriate prompt for this section
                            text_embeds_index = iteration_count
                            if text_embeds_index >= len(text_embeds_list):
                                text_embeds_index = len(text_embeds_list) - 1

                            print(f"!!!!Using text embeds index: {text_embeds_index}/{len(text_embeds_list)} for iteration {iteration_count}")
                            text_embeds = text_embeds_list[text_embeds_index]

                            if len(text_embeds["prompt_embeds"]) > 1:
                                prompt_index = min(iteration_count, len(text_embeds["prompt_embeds"]) - 1)
                                positive = [text_embeds["prompt_embeds"][prompt_index]]
                                log.info(f"Using prompt index: {prompt_index}")
                            else:
                                positive = text_embeds["prompt_embeds"]

                            window_vace_data = None
                            # if vace_data is not None:
                            #     window_vace_data = []
                            #     for vace_entry in vace_data:
                            #         partial_context = vace_entry["context"][0][:, latent_start_idx:latent_end_idx]
                            #         if has_ref:
                            #             partial_context[:, 0] = vace_entry["context"][0][:, 0]
                                    
                            #         window_vace_data.append({
                            #             "context": [partial_context], 
                            #             "scale": vace_entry["scale"],
                            #             "start": vace_entry["start"], 
                            #             "end": vace_entry["end"],
                            #             "seq_len": vace_entry["seq_len"]
                            #         })

                            # uni3c slices
                            if uni3c_embeds is not None:
                                vae.to(device)
                                # Pad original_images if needed
                                num_frames = original_images.shape[2]
                                required_frames = audio_end_idx - audio_start_idx
                                if audio_end_idx > num_frames:
                                    pad_len = audio_end_idx - num_frames
                                    last_frame = original_images[:, :, -1:].repeat(1, 1, pad_len, 1, 1)
                                    padded_images = torch.cat([original_images, last_frame], dim=2)
                                else:
                                    padded_images = original_images
                                render_latent = vae.encode(
                                    padded_images[:, :, audio_start_idx:audio_end_idx].to(device, vae.dtype),
                                    device=device, tiled=tiled_vae
                                ).to(dtype)
                                vae.model.clear_cache()
                                vae.to(offload_device)
                                pcd_data['render_latent'] = render_latent

                            # unianimate poses - process from original images per chunk
                            partial_unianim_data = None
                            if unianimate_poses is not None:
                                print(f"[UniAnimate Debug] Processing UniAnimate poses for audio indices {audio_start_idx} to {audio_end_idx}")
                                
                                # Get the original pose images for this audio window
                                original_pose_images = unianimate_poses["pose"]  # Original tensor
                                total_pose_frames = original_pose_images.shape[2]
                                
                                # Calculate pose frame indices based on audio indices
                                # Poses align 1:1 with audio frames (both at video frame rate)
                                pose_start_idx = audio_start_idx
                                pose_end_idx = min(audio_end_idx, total_pose_frames)
                                
                                print(f"[UniAnimate Debug] Extracting pose frames {pose_start_idx} to {pose_end_idx} from total {total_pose_frames}")
                                print(f"[UniAnimate Debug] Expected frame count for this batch: {frame_num}")
                                
                                # Handle bounds checking
                                if pose_start_idx >= total_pose_frames:
                                    # Use the last pose frame and repeat it
                                    pose_chunk = original_pose_images[:, :, -1:].repeat(1, 1, frame_num, 1, 1)
                                    print(f"[UniAnimate Debug] WARNING: Pose start index {pose_start_idx} exceeds available frames {total_pose_frames}, using last frame")
                                else:
                                    # Extract the relevant pose frames
                                    pose_chunk = original_pose_images[:, :, pose_start_idx:pose_end_idx]
                                    
                                    # Pad if needed
                                    required_frames = frame_num
                                    actual_frames = pose_chunk.shape[2]
                                    
                                    if actual_frames < required_frames:
                                        # Pad with the last available frame
                                        pad_frames = required_frames - actual_frames
                                        if actual_frames > 0:
                                            last_frame = pose_chunk[:, :, -1:].repeat(1, 1, pad_frames, 1, 1)
                                            pose_chunk = torch.cat([pose_chunk, last_frame], dim=2)
                                        else:
                                            # No frames available, use zeros or skip
                                            pose_chunk = torch.zeros(1, 1, required_frames, original_pose_images.shape[3], original_pose_images.shape[4])
                                        print(f"[UniAnimate Debug] Padded pose chunk from {actual_frames} to {required_frames} frames")
                                    elif actual_frames > required_frames:
                                        # Truncate to required frames
                                        pose_chunk = pose_chunk[:, :, :required_frames]
                                        print(f"[UniAnimate Debug] Truncated pose chunk from {actual_frames} to {required_frames} frames")
                                
                                # Now process this chunk the same way as in the highlighted code
                                transformer.dwpose_embedding.to(device, dtype)
                                dwpose_data_chunk = pose_chunk.to(device, dtype)
                                dwpose_data_chunk = torch.cat([dwpose_data_chunk[:,:,:1].repeat(1,1,3,1,1), dwpose_data_chunk], dim=2)
                                dwpose_data_chunk = transformer.dwpose_embedding(dwpose_data_chunk)
                                
                                print(f"[UniAnimate Debug] Processed dwpose chunk shape: {dwpose_data_chunk.shape}")
                                
                                # Handle reference pose if needed
                                random_ref_dwpose_data = None
                                if image_cond is not None:
                                    transformer.randomref_embedding_pose.to(device, dtype)
                                    random_ref_dwpose = unianimate_poses.get("ref", None)
                                    if random_ref_dwpose is not None:
                                        random_ref_dwpose_data = transformer.randomref_embedding_pose(
                                            random_ref_dwpose.to(device, dtype)
                                            ).unsqueeze(2).to(model["dtype"])
                                    del random_ref_dwpose
                                
                                partial_unianim_data = {
                                    "dwpose": dwpose_data_chunk,
                                    "random_ref": random_ref_dwpose_data.squeeze(0) if random_ref_dwpose_data is not None else None,
                                    "strength": unianimate_poses["strength"],
                                    "start_percent": unianimate_poses["start_percent"],
                                    "end_percent": unianimate_poses["end_percent"]
                                }
                                
                                print(f"[UniAnimate Debug] Successfully created partial_unianim_data with dwpose shape: {dwpose_data_chunk.shape}")
                                
                                # Clean up
                                del pose_chunk, dwpose_data_chunk

                            # fantasy portrait slices
                            partial_fantasy_portrait_input = None
                            if fantasy_portrait_input is not None:
                                adapter_proj = fantasy_portrait_input["adapter_proj"]
                                if latent_end_idx > adapter_proj.shape[1]:
                                    pad_len = latent_end_idx - adapter_proj.shape[1]
                                    last_frame = adapter_proj[:, -1:, :, :].repeat(1, pad_len, 1, 1)
                                    padded_proj = torch.cat([adapter_proj, last_frame], dim=1)
                                else:
                                    padded_proj = adapter_proj
                                partial_fantasy_portrait_input = fantasy_portrait_input.copy()
                                partial_fantasy_portrait_input["adapter_proj"] = padded_proj[:, latent_start_idx:latent_end_idx]

                            mm.soft_empty_cache()
                            gc.collect()
                            # sampling loop
                            sampling_pbar = tqdm(total=len(timesteps)-1, desc=f"Sampling audio indices {audio_start_idx}-{audio_end_idx}", position=0, leave=True)
                            for i in range(len(timesteps)-1):
                                timestep = timesteps[i]
                                latent_model_input = latent.to(device)
                                if mode == "infinitetalk":
                                    latent_model_input[:, :cur_motion_frames_latent_num] = latent_motion_frames

                                noise_pred, self.cache_state = predict_with_cfg(
                                    latent_model_input, cfg[min(i, len(timesteps)-1)], positive, text_embeds["negative_prompt_embeds"],
                                    timestep, i, y, clip_embeds, control_latents, window_vace_data, partial_unianim_data, audio_proj, control_camera_latents, add_cond,
                                    cache_state=self.cache_state, multitalk_audio_embeds=audio_embs, fantasy_portrait_input=partial_fantasy_portrait_input,
                                    text_embeds_override=text_embeds)

                                if callback is not None:
                                    callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                    callback(step_iteration_count, callback_latent, None, estimated_iterations*(len(timesteps)-1))
                                    del callback_latent

                                sampling_pbar.update(1)
                                step_iteration_count += 1

                                # update latent
                                if scheduler == "multitalk":
                                    noise_pred = -noise_pred
                                    dt = (timesteps[i] - timesteps[i + 1]) / 1000
                                    latent = latent + noise_pred * dt[:, None, None, None]
                                else:
                                    latent = sample_scheduler.step(noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0).to(noise_pred.device), **scheduler_step_args)[0].squeeze(0)
                                del noise_pred, latent_model_input, timestep
                                
                                # differential diffusion inpaint
                                if masks is not None:
                                    if i < len(timesteps) - 1:
                                        image_latent = add_noise(original_image.to(device), noise.to(device), timesteps[i+1])
                                        mask = masks[i].to(latent)
                                        latent = image_latent * mask + latent * (1-mask)

                                # injecting motion frames
                                if not is_first_clip and mode == "multitalk":
                                    latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                                    motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                                    add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                                    latent[:, :add_latent.shape[1]] = add_latent
                                else:
                                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames

                            del noise
                            if offload:
                                offload_transformer(transformer)
                                offloaded = True
                            
                            # Optional denoising step before VAE decoding
                            if denoise_latents:
                                with torch.no_grad():
                                    # Calculate denoised latents using the final timestep
                                    final_timestep = timesteps[-1] if len(timesteps) > 0 else torch.tensor([0.], device=device)
                                    latent_model_input = latent.to(device)
                                    
                                    # Apply motion frame injection for denoising step
                                    if mode == "infinitetalk":
                                        # Save motion frames before deletion
                                        saved_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                                        latent_model_input[:, :cur_motion_frames_latent_num] = saved_motion_frames
                                    
                                    # Run final prediction to get denoised result
                                    final_cfg = cfg[-1] if isinstance(cfg, list) else cfg
                                    final_noise_pred, _ = predict_with_cfg(
                                        latent_model_input, final_cfg, positive, text_embeds["negative_prompt_embeds"], 
                                        final_timestep, len(timesteps)-1, y, clip_embeds, control_latents, window_vace_data, 
                                        partial_unianim_data, audio_proj, control_camera_latents, add_cond,
                                        cache_state=self.cache_state, multitalk_audio_embeds=audio_embs, 
                                        fantasy_portrait_input=partial_fantasy_portrait_input, text_embeds_override=text_embeds)
                                    
                                    # Calculate denoised latents using the same formula as normal inference
                                    denoised_latents = (latent_model_input.to(device) - final_noise_pred.to(device) * final_timestep.to(device) / 1000).detach()
                                    latent = denoised_latents
                                    
                                    del final_noise_pred, latent_model_input, denoised_latents
                                    if mode == "infinitetalk":
                                        del saved_motion_frames
                            
                            del latent_motion_frames
                            vae.to(device)
                            videos = vae.decode(latent.unsqueeze(0).to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False)[0].cpu()
                            vae.model.clear_cache()
                            vae.to(offload_device)

                            sampling_pbar.close()
                            
                            # optional color correction (less relevant for InfiniteTalk)
                            if colormatch != "disabled":
                                videos = videos.permute(1, 2, 3, 0).float().numpy()
                                from color_matcher import ColorMatcher
                                cm = ColorMatcher()
                                cm_result_list = []
                                for img in videos:
                                    if mode == "multitalk":
                                        cm_result = cm.transfer(src=img, ref=original_images[0].permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                                    else:
                                        cm_result = cm.transfer(src=img, ref=cond_image[0].permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                                    cm_result_list.append(torch.from_numpy(cm_result).to(vae.dtype))
                        
                                videos = torch.stack(cm_result_list, dim=0).permute(3, 0, 1, 2)

                            # optionally save generated samples to disk
                            if output_path:
                                video_np = videos.clamp(-1.0, 1.0).add(1.0).div(2.0).mul(255).cpu().float().numpy().transpose(1, 2, 3, 0).astype('uint8')
                                num_frames_to_save = video_np.shape[0] if is_first_clip else video_np.shape[0] - cur_motion_frames_num
                                log.info(f"Saving {num_frames_to_save} generated frames to {output_path}")
                                start_idx = 0 if is_first_clip else cur_motion_frames_num
                                for i in range(start_idx, video_np.shape[0]):
                                    im = Image.fromarray(video_np[i])
                                    im.save(os.path.join(output_path, f"frame_{img_counter:05d}.png"))
                                    img_counter += 1
                            else:
                                gen_video_list.append(videos if is_first_clip else videos[:, cur_motion_frames_num:])

                            current_condframe_index += 1
                            iteration_count += 1

                            # decide whether is done
                            if arrive_last_frame: 
                                break

                            # update next condition frames
                            is_first_clip = False
                            cur_motion_frames_num = motion_frame

                            cond_ = videos[:, -cur_motion_frames_num:].unsqueeze(0)
                            if mode == "infinitetalk":
                                cond_frame = cond_
                            else:
                                cond_image = cond_

                            del videos, latent

                            # Repeat audio emb
                            if multitalk_embeds is not None:
                                audio_start_idx += (frame_num - cur_motion_frames_num)
                                audio_end_idx = audio_start_idx + clip_length
                                if audio_end_idx >= total_frames:
                                    arrive_last_frame = True
                                    miss_lengths = []
                                    source_frames = []
                                    for human_inx in range(human_num):
                                        source_frame = len(audio_embedding[human_inx])
                                        source_frames.append(source_frame)
                                        if audio_end_idx >= len(audio_embedding[human_inx]):
                                            print(f"Audio embedding for subject {human_inx} not long enough: {len(audio_embedding[human_inx])}, need {audio_end_idx}, padding...")
                                            miss_length = audio_end_idx - len(audio_embedding[human_inx]) + 3
                                            print(f"Padding length: {miss_length}")
                                            if encoded_silence is not None:
                                                add_audio_emb = encoded_silence[-1*miss_length:]
                                            else:
                                                add_audio_emb = torch.flip(audio_embedding[human_inx][-1*miss_length:], dims=[0])
                                            audio_embedding[human_inx] = torch.cat([audio_embedding[human_inx], add_audio_emb.to(device, dtype)], dim=0)
                                            miss_lengths.append(miss_length)
                                        else:
                                            miss_lengths.append(0)
                                if mode == "infinitetalk" and current_condframe_index >= original_images.shape[2]:
                                    last_frame = original_images[:, :, -1:, :, :]
                                    miss_length   = 1
                                    original_images = torch.cat([original_images, last_frame.repeat(1, 1, miss_length, 1, 1)], dim=2)
                        
                        if not output_path:
                            gen_video_samples = torch.cat(gen_video_list, dim=1)
							# Ensure we have exactly the target number of frames
                            if gen_video_samples.shape[1] > total_frames:
                                original_frame_count = gen_video_samples.shape[1]
                                gen_video_samples = gen_video_samples[:, :total_frames]
                                log.info(f"Truncated generated video from {original_frame_count} frames to {total_frames} frames to match audio length")
                        else:
                            gen_video_samples = torch.zeros(3, 1, 64, 64) # dummy output

                        if force_offload:
                            if not model["auto_cpu_offload"]:
                                offload_transformer(transformer)
                        try:
                            print_memory(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        except:
                            pass
                        return {"video": gen_video_samples.permute(1, 2, 3, 0), "output_path": output_path},
                    # region framepack loop
                    elif framepack:
                        framepack_out = []
                        ref_motion_image = None
                        #infer_frames = image_embeds["num_frames"]
                        infer_frames = s2v_audio_embeds.get("frame_window_size", 80)
                        motion_frames = infer_frames - 7 #73 default
                        lat_motion_frames = (motion_frames + 3) // 4
                        lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames
                        
                        step_iteration_count = 0
                        total_frames = s2v_audio_input.shape[-1]

                        s2v_motion_frames = [motion_frames, lat_motion_frames]

                        noise = torch.randn( #C, T, H, W
                            48 if is_5b else 16,
                                lat_target_frames,
                                target_shape[2],
                                target_shape[3],
                                dtype=torch.float32,
                                generator=seed_g,
                                device=torch.device("cpu"))
                        
                        seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

                        if ref_motion_image is None:
                            ref_motion_image = torch.zeros(
                                [1, 3, motion_frames, latent.shape[2]*vae_upscale_factor, latent.shape[3]*vae_upscale_factor],
                                dtype=vae.dtype,
                                device=device)
                        videos_last_frames = ref_motion_image

                        if s2v_pose is not None:
                            pose_cond_list = []
                            for r in range(s2v_num_repeat):
                                pose_start = r * (infer_frames // 4)
                                pose_end = pose_start + (infer_frames // 4)
                            
                                cond_lat = s2v_pose[:, :, pose_start:pose_end]
                            
                                pad_len = (infer_frames // 4) - cond_lat.shape[2]
                                if pad_len > 0:
                                    pad = -torch.ones(cond_lat.shape[0], cond_lat.shape[1], pad_len, cond_lat.shape[3], cond_lat.shape[4], device=cond_lat.device, dtype=cond_lat.dtype)
                                    cond_lat = torch.cat([cond_lat, pad], dim=2)
                                pose_cond_list.append(cond_lat.cpu())

                        log.info(f"Sampling {total_frames} frames in {s2v_num_repeat} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")
                        # sample
                        for r in range(s2v_num_repeat):
                            vae.model.clear_cache()
                            mm.soft_empty_cache()
                            gc.collect()
                            if ref_motion_image is not None:
                                vae.to(device)
                                ref_motion = vae.encode(ref_motion_image.to(vae.dtype), device=device, pbar=False).to(dtype)[0]
                                vae.model.clear_cache()
                                vae.to(offload_device)

                            left_idx = r * infer_frames
                            right_idx = r * infer_frames + infer_frames
                            
                            s2v_audio_input_slice = s2v_audio_input[..., left_idx:right_idx]
                            if s2v_audio_input_slice.shape[-1] < (right_idx - left_idx):
                                pad_len = (right_idx - left_idx) - s2v_audio_input_slice.shape[-1]
                                pad_shape = list(s2v_audio_input_slice.shape)
                                pad_shape[-1] = pad_len
                                pad = torch.zeros(pad_shape, device=s2v_audio_input_slice.device, dtype=s2v_audio_input_slice.dtype)
                                log.info(f"Padding s2v_audio_input_slice from {s2v_audio_input_slice.shape[-1]} to {right_idx - left_idx}")
                                s2v_audio_input_slice = torch.cat([s2v_audio_input_slice, pad], dim=-1)

                            if ref_motion_image is not None:
                                input_motion_latents = ref_motion.clone().unsqueeze(0)
                            else:
                                input_motion_latents = None

                            s2v_pose_slice = None
                            if s2v_pose is not None:
                                s2v_pose_slice = pose_cond_list[r].to(device)

                            sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)

                            latent = noise.to(device)
                            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling audio indices {left_idx}-{right_idx}", position=0)):
                                latent_model_input = latent.to(device)
                                timestep = torch.tensor([t]).to(device)
                                noise_pred, self.cache_state = predict_with_cfg(
                                    latent_model_input, 
                                    cfg[idx], 
                                    text_embeds["prompt_embeds"], 
                                    text_embeds["negative_prompt_embeds"], 
                                    timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                                    cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, mtv_motion_tokens=mtv_motion_tokens, 
                                    s2v_audio_input=s2v_audio_input_slice, s2v_ref_motion=input_motion_latents, s2v_motion_frames=s2v_motion_frames, s2v_pose=s2v_pose_slice)
                            
                                latent = sample_scheduler.step(
                                        noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0),
                                        **scheduler_step_args)[0].squeeze(0)
                                if callback is not None:
                                    callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                    callback(step_iteration_count, callback_latent, None, s2v_num_repeat*(len(timesteps)))
                                    del callback_latent
                                step_iteration_count += 1
                                del latent_model_input, noise_pred
                                
                            
                            vae.to(device)
                            decode_latents = torch.cat([ref_motion.unsqueeze(0), latent.unsqueeze(0)], dim=2)
                            image = vae.decode(decode_latents.to(device, vae.dtype), device=device, pbar=False)[0]
                            del decode_latents
                            image = image.unsqueeze(0)[:, :, -infer_frames:]
                            if r == 0:
                                image = image[:, :, 3:]

                            framepack_out.append(image.cpu())

                            overlap_frames_num = min(motion_frames, image.shape[2])
                            
                            videos_last_frames = torch.cat([
                                videos_last_frames[:, :, overlap_frames_num:],
                                image[:, :, -overlap_frames_num:]], dim=2).to(device, vae.dtype)
                          
                            ref_motion_image = videos_last_frames
                            
                        vae.to(offload_device)
                        vae.model.clear_cache()
                        mm.soft_empty_cache()
                        gen_video_samples = torch.cat(framepack_out, dim=2).squeeze(0).permute(1, 2, 3, 0)

                        if force_offload:
                            if not model["auto_cpu_offload"]:
                                offload_transformer(transformer)
                        try:
                            print_memory(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        except:
                            pass
                        return {"video": gen_video_samples},
                        
                    #region normal inference
                    else:
                        noise_pred, self.cache_state = predict_with_cfg(
                            latent_model_input, 
                            cfg[idx], text_embeds["prompt_embeds"], 
                            text_embeds["negative_prompt_embeds"], 
                            timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                            cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, mtv_motion_tokens=mtv_motion_tokens, s2v_audio_input=s2v_audio_input)
                        if bidirectional_sampling:
                            noise_pred_flipped, self.cache_state = predict_with_cfg(
                            latent_model_input_flipped, 
                            cfg[idx], text_embeds["prompt_embeds"], 
                            text_embeds["negative_prompt_embeds"], 
                            timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                            cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, mtv_motion_tokens=mtv_motion_tokens,reverse_time=True)

                    if latent_shift_loop:
                        #reverse latent shift
                        if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                            noise_pred = torch.cat([noise_pred[:, latent_video_length - shift_idx:]] + [noise_pred[:, :latent_video_length - shift_idx]], dim=1)
                            shift_idx = (shift_idx + latent_skip) % latent_video_length
                        
                    
                    if flowedit_args is None:
                        latent = latent.to(intermediate_device)
                        
                        if len(timestep.shape) != 1 and not is_pusa: #5b
                            # all_indices is a list of indices to skip
                            total_indices = list(range(latent.shape[1]))
                            process_indices = [i for i in total_indices if i not in all_indices]
                            if process_indices:
                                latent_to_process = latent[:, process_indices]
                                noise_pred_to_process = noise_pred[:, process_indices]
                                latent_slice = sample_scheduler.step(
                                    noise_pred_to_process.unsqueeze(0),
                                    orig_timestep,
                                    latent_to_process.unsqueeze(0),
                                    **scheduler_step_args
                                )[0].squeeze(0)
                            # Reconstruct the latent tensor: keep skipped indices as-is, update others
                            new_latent = []
                            for i in total_indices:
                                if i in all_indices:
                                    new_latent.append(latent[:, i:i+1])
                                else:
                                    j = process_indices.index(i)
                                    new_latent.append(latent_slice[:, j:j+1])
                            latent = torch.cat(new_latent, dim=1)
                        else:
                            latent = sample_scheduler.step(
                                noise_pred[:, :orig_noise_len].unsqueeze(0) if recammaster is not None else noise_pred.unsqueeze(0),
                                timestep,
                                latent[:, :orig_noise_len].unsqueeze(0) if recammaster is not None else latent.unsqueeze(0),
                                **scheduler_step_args)[0].squeeze(0)
                            if noise_pred_flipped is not None:
                                latent_backwards = sample_scheduler_flipped.step(
                                    noise_pred_flipped.unsqueeze(0),
                                    timestep,
                                    latent_flipped.unsqueeze(0),
                                    **scheduler_step_args)[0].squeeze(0)
                                latent_backwards = torch.flip(latent_backwards, dims=[1])
                                latent = latent * 0.5 + latent_backwards * 0.5

                        #InfiniteTalk first frame handling
                        if (extra_latents is not None
                            and not multitalk_sampling
                            and transformer.multitalk_model_type=="InfiniteTalk"):
                            for entry in extra_latents:
                                add_index = entry["index"]
                                num_extra_frames = entry["samples"].shape[2]
                                latent[:, add_index:add_index+num_extra_frames] = entry["samples"].to(latent)

                        # differential diffusion inpaint
                        if masks is not None:
                            if idx < len(timesteps) - 1:
                                noise_timestep = timesteps[idx+1]
                                image_latent = sample_scheduler.scale_noise(
                                    original_image.to(device), torch.tensor([noise_timestep]), noise.to(device)
                                )
                                mask = masks[idx].to(latent)
                                latent = image_latent * mask + latent * (1-mask)

                        if freeinit_args is not None:
                            current_latent = latent.clone()

                        if callback is not None:
                            if recammaster is not None:
                                callback_latent = (latent_model_input[:, :orig_noise_len].to(device) - noise_pred[:, :orig_noise_len].to(device) * t.to(device) / 1000).detach()
                            #elif phantom_latents is not None:
                            #    callback_latent = (latent_model_input[:,:-phantom_latents.shape[1]].to(device) - noise_pred[:,:-phantom_latents.shape[1]].to(device) * t.to(device) / 1000).detach()
                            elif humo_image_cond is not None and humo_reference_count > 0:
                                callback_latent = (latent_model_input[:,:-humo_reference_count].to(device) - noise_pred[:,:-humo_reference_count].to(device) * t.to(device) / 1000).detach()
                            else:
                                callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach()
                            callback(idx, callback_latent.permute(1,0,2,3), None, len(timesteps))
                        else:
                            pbar.update(1)
                    else:
                        if callback is not None:
                            callback_latent = (zt_tgt.to(device) - vt_tgt.to(device) * t.to(device) / 1000).detach()
                            callback(idx, callback_latent.permute(1,0,2,3), None, len(timesteps))
                        else:
                            pbar.update(1)
            except Exception as e:
                log.error(f"Error during sampling: {e}")
                if force_offload:
                    if not model["auto_cpu_offload"]:
                        offload_transformer(transformer)
                raise e

        if phantom_latents is not None:
            latent = latent[:,:-phantom_latents.shape[1]]
        if humo_image_cond is not None and humo_reference_count > 0:
            latent = latent[:,:-humo_reference_count]
        
        cache_states = None
        if cache_args is not None:
            cache_report(transformer, cache_args)
            if end_step != -1 and end_step < total_steps:
                cache_states = {
                    "cache_state": self.cache_state,
                    "easycache_state": transformer.easycache_state,
                    "teacache_state": transformer.teacache_state,
                    "magcache_state": transformer.magcache_state,
                }

        if force_offload:
            if not model["auto_cpu_offload"]:
                offload_transformer(transformer)

        try:
            print_memory(device)
            #torch.cuda.memory._dump_snapshot("wanvideowrapper_memory_dump.pt")
            #torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        return ({
            "samples": latent.unsqueeze(0).cpu(), 
            "looped": is_looped, 
            "end_image": end_image if not fun_or_fl2v_model else None, 
            "has_ref": has_ref, 
            "drop_last": drop_last,
            "generator_state": seed_g.get_state(),
            "original_image": original_image.cpu() if original_image is not None else None,
            "cache_states": cache_states
        },{
            "samples": callback_latent.unsqueeze(0).cpu() if callback is not None else None, 
        })

class WanVideoLoopingSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "total_frames": ("INT", {"default": 65, "min": 1, "step": 4, "tooltip": "Total length of the video in frames"}),
                "batch_length": ("INT", {"default": 65, "min": 1, "step": 4, "max": 1000, "tooltip": "Number of new frames to generate in each batch"}),
                "overlap_length": ("INT", {"default": 6, "min": 0, "max": 1000, "tooltip": "Number of previously generated frames to use in each batch"}),
                "vae": ("WANVAE",),
                "encode_latent_Args": ("WANENCODEARGS", ),
                "decode_latent_Args": ("WANDECODEARGS", ),
                "width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "model": ("WANVIDEOMODEL",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed_adjust": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "tooltip": "Adjust the seed for each batch"}),
                "seed_batch_control": (["Seed Adjust", "Randomize"],
                    {"default": "Seed Adjust"}
                ),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (scheduler_list, {"default": scheduler_list[0],}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
            },
            "optional": {
                "starting_images": ("IMAGE", {"tooltip": "The starting images for the video generation. These will not change."}),
                "control_images_1": ("WANVIDEOCONTROLIMAGESOPTIONS", {"tooltip": "The first set of control images."}),
                "control_images_2": ("WANVIDEOCONTROLIMAGESOPTIONS", {"tooltip": "The second set of control images."}),
                "control_images_3": ("WANVIDEOCONTROLIMAGESOPTIONS", {"tooltip": "The third set of control images."}),
                "control_images_4": ("WANVIDEOCONTROLIMAGESOPTIONS", {"tooltip": "The fourth set of control images."}),
                "control_images_5": ("WANVIDEOCONTROLIMAGESOPTIONS", {"tooltip": "The fifth set of control images."}),
                "color_match_args": ("COLOURMATCHARGS", {"tooltip": "Arguments for color matching."}),
                "simple_scale_Args": ("WANSIMPLESCALEARGS", {"tooltip": "Arguments for simple scaling."}),
                "restore_face": ("RESTOREFACEARGS", ),
                "use_restore_face": ("BOOLEAN", {"default": True, "tooltip": "Use provided RestoreFace to restore faces in the generated video"}),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "cache_args": ("CACHEARGS", ),
                "flowedit_args": ("FLOWEDITARGS", ),
                "batched_cfg": ("BOOLEAN", {"default": False, "tooltip": "Batch cond and uncond for faster sampling, possibly faster on some hardware, uses more memory"}),
                "slg_args": ("SLGARGS", ),
                "rope_function": (["default", "comfy", "comfy_chunked"], {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile. Chunked version has reduced peak VRAM usage when not using torch.compile"}),
                "loop_args": ("LOOPARGS", ),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "sigmas": ("SIGMAS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS", ),
                "uni3c_embeds": ("UNI3C_EMBEDS", ),
                "multitalk_embeds": ("MULTITALK_EMBEDS", ),
                "freeinit_args": ("FREEINITARGS", ),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Start step for the sampling, 0 means full sampling, otherwise samples only from this step"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1, "tooltip": "End step for the sampling, -1 means full sampling, otherwise samples only until this step"}),
                "add_noise_to_samples": ("BOOLEAN", {"default": False, "tooltip": "Add noise to the samples before sampling, needed for video2video sampling when starting from clean video"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_images",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def generateWanVaceEmbeds(
            self,
            totalFramesToGenerate,
            starting_images,
            control_images,
            controlImageStartindex,
            controlImageEndIndex,
            vae,
            width,
            height,
            control_after_start,
            startingImagesAreGenerated,
            batchIndex,
            ):
        numberOfStartingImages = starting_images.shape[0] if (starting_images is not None and starting_images.shape is not None) else 0
        print(f"WanVideoLoopingSampler Generating VACE embeds. totalFramesToGenerate: {totalFramesToGenerate}, numberOfStartingImages: {numberOfStartingImages}, controlImageStartindex: {controlImageStartindex}, controlImageEndIndex: {controlImageEndIndex}, width: {width}, height: {height}")
        
        # Create reusable instances once
        if not hasattr(self, '_vace_start_to_end_frame'):
            self._vace_start_to_end_frame = WanVideoVACEStartToEndFrame()
        if not hasattr(self, '_vace_encode'):
            self._vace_encode = WanVideoVACEEncode()
        
        # Process control images sequentially to reduce peak memory usage
        vaceEmbeds = None
        previousVaceEmbeds = None
        
        # Helper function to process a single control image set
        def process_control_image_set(control_image_data, is_fallback=False, defaultStrength=1.0):
            nonlocal vaceEmbeds, previousVaceEmbeds
            
            usedControlImages = None
            usedStrength = defaultStrength
            usedStartPercent = 0.0
            usedEndPercent = 1.0
            usedTiledVae = False
            useProvidedStartingimages=True
            useGeneratedStartingImages=True
            customName = "Fallback"
            
            if not is_fallback and control_image_data is not None:
                usedStrength = control_image_data.get("strength", 0.5)
                if batchIndex > 0:
                    usedStrength = control_image_data.get("strength_later", usedStrength)
                customName = control_image_data.get("custom_name", "NoName")
                if usedStrength == 0.0:
                    print(f"WanVideoLoopingSampler - {customName} control image strength is 0, skipping control images processing.")
                    return usedStrength
                usedControlImages = control_image_data.get("control_images", None)
                if usedControlImages is not None:
                    # use controlImageStartindex & controlImageEndIndex to get the actual images needed, given the shape is (B, H, W, C)
                    usedControlImages = usedControlImages[controlImageStartindex:controlImageEndIndex]
                
                usedStartPercent = control_image_data.get("start_percent", 0.0)
                usedEndPercent = control_image_data.get("end_percent", 1.0)
                usedTiledVae = control_image_data.get("tiled_vae", False)
                useProvidedStartingimages = control_image_data.get("use_provided_starting_images", True)
                useGeneratedStartingImages = control_image_data.get("use_generated_starting_images", True)

            useStartingImages = True
            if startingImagesAreGenerated:
                useStartingImages = useGeneratedStartingImages
            else:
                useStartingImages = useProvidedStartingimages

            usedStartingImages = starting_images if useStartingImages else None
            if usedStartingImages is None and usedControlImages is None:
                print(f"WanVideoLoopingSampler - {customName} no starting images or control images provided, skipping control images processing.")
                return 0.0
            print(f"WanVideoLoopingSampler - Processing {customName} control images: {usedControlImages.shape if usedControlImages is not None else 'None'}, strength: {usedStrength}, start_percent: {usedStartPercent}, end_percent: {usedEndPercent}, tiled_vae: {usedTiledVae}, useProvidedStartingimages: {useProvidedStartingimages}, useGeneratedStartingImages: {useGeneratedStartingImages}, usedStartingImages.shape: {usedStartingImages.shape if usedStartingImages is not None else 'None'}")

            # Process frames and masks immediately without storing
            startToEndFrame = self._vace_start_to_end_frame.process(
                num_frames=totalFramesToGenerate,
                empty_frame_level=1.0,
                start_image=usedStartingImages,
                end_image=None, # end image
                control_images=usedControlImages,
                inpaint_mask=None, # inpaint mask
                start_index=0, # start index
                end_index=-1, # end index
                control_after_start=control_after_start, # control after start
                ignore_start_frame=False, # ignore start frame
            )
            
            usedInputFrames = startToEndFrame[0]
            usedInputMasks = startToEndFrame[1]
            
            # Process VACE embeds immediately
            vaceEmbeds = self._vace_encode.process(
                    vae = vae,
                    width = width,
                    height = height,
                    num_frames = totalFramesToGenerate,
                    strength = usedStrength,
                    vace_start_percent = usedStartPercent,
                    vace_end_percent = usedEndPercent,
                    input_frames = usedInputFrames,
                    ref_images = None, # ref_images
                    input_masks = usedInputMasks,
                    prev_vace_embeds = previousVaceEmbeds,
                    tiled_vae = usedTiledVae
                )[0]
            
            # Clear intermediate data explicitly
            del startToEndFrame, usedInputFrames, usedInputMasks
            if usedControlImages is not None:
                del usedControlImages
            
            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            previousVaceEmbeds = vaceEmbeds
            return usedStrength
        
        # Process each control image set sequentially
        control_images_processed = False
        totalUsedStrength = 0.0
        for i, control_image_data in enumerate(control_images):
            if control_image_data is None:
                continue
            totalUsedStrength += process_control_image_set(control_image_data)
            control_images_processed = True
        
        # if no control images were processed, create fallback
        if not control_images_processed:
            print(f"WanVideoLoopingSampler - No control images processed, processing fallback control images.")
            process_control_image_set(None, is_fallback=True)
        elif totalUsedStrength < 1.0 and batchIndex > 0:
            print(f"WanVideoLoopingSampler - Total used strength is {totalUsedStrength}, processing fallback control images to fill the gap.")
            process_control_image_set(None, is_fallback=True, defaultStrength=1.0-totalUsedStrength)
        
        return vaceEmbeds

    def process(self,
        total_frames, batch_length, overlap_length, vae, encode_latent_Args, decode_latent_Args, width, height,
        model, shift, steps, cfg, seed, seed_adjust, seed_batch_control,
        scheduler, riflex_freq_index,
        starting_images=None, control_images_1=None, control_images_2=None, control_images_3=None, control_images_4=None, control_images_5=None, color_match_args=None, simple_scale_Args=None, restore_face=None, use_restore_face=True,
        text_embeds=None,
        force_offload=True, samples=None, feta_args=None, denoise_strength=1.0, context_options=None, 
        cache_args=None, teacache_args=None, flowedit_args=None, batched_cfg=False, slg_args=None, rope_function="default", loop_args=None, 
        experimental_args=None, sigmas=None, unianimate_poses=None, fantasytalking_embeds=None, uni3c_embeds=None, multitalk_embeds=None, freeinit_args=None, start_step=0, end_step=-1, add_noise_to_samples=False):
        overallStartTime = time.time()  # Start timing
        print("Processing video with WanVideoLoopingSampler")
                
        decodeTile = False
        decodeTileX = 272
        decodeTileY = 272
        decodeTileStrideX = 144
        decodeTileStrideY = 128
        if decode_latent_Args:
            decodeTile = decode_latent_Args.get("enable_vae_tiling", decodeTile)
            decodeTileX = decode_latent_Args.get("tile_x", decodeTileX)
            decodeTileY = decode_latent_Args.get("tile_y", decodeTileY)
            decodeTileStrideX = decode_latent_Args.get("tile_stride_x", decodeTileStrideX)
            decodeTileStrideY = decode_latent_Args.get("tile_stride_y", decodeTileStrideY)

        encodeTile = False
        encodeTileX = 272
        encodeTileY = 272
        encodeTileStrideX = 144
        encodeTileStrideY = 128
        encodeAugStrength = 0.0
        encodeLatentStrength = 1.0
        encodeMask = None
        if encode_latent_Args:
            encodeTile = encode_latent_Args.get("enable_vae_tiling", encodeTile)
            encodeTileX = encode_latent_Args.get("tile_x", encodeTileX)
            encodeTileY = encode_latent_Args.get("tile_y", encodeTileY)
            encodeTileStrideX = encode_latent_Args.get("tile_stride_x", encodeTileStrideX)
            encodeTileStrideY = encode_latent_Args.get("tile_stride_y", encodeTileStrideY)
            encodeAugStrength = encode_latent_Args.get("noise_aug_strength", encodeAugStrength)
            encodeLatentStrength = encode_latent_Args.get("latent_strength", encodeLatentStrength)
            encodeMask = encode_latent_Args.get("mask", encodeMask)

        color_match_source = None
        color_match_used_source = 'disable-match'
        color_match_method = 'hm-mvgd-hm'
        color_match_strength = 0.0
        color_match_provided_frame = 0
        if color_match_args:
            print(f"Color match args is provided, using color match")
            color_match_used_source = color_match_args.get("used_source", color_match_used_source)
            if color_match_used_source != 'disable-match':
                print(f"Color match source is {color_match_used_source}, still using color match")
                if color_match_used_source == 'provided':
                    color_match_source = color_match_args.get("source", color_match_source)
                    if color_match_source is None:
                        color_match_used_source = 'first-frame'
                        print(f"WARNING - Color match source is None, using first frame as source")
                color_match_method = color_match_args.get("color_match_method", color_match_method)
                color_match_strength = color_match_args.get("color_match_strength", color_match_strength)
                color_match_provided_frame = color_match_args.get("provided_frame", color_match_provided_frame)
                print(f"Color match method is {color_match_method}, strength is {color_match_strength}, used source is {color_match_used_source}, color_match_provided_frame is {color_match_provided_frame}")
            else:
                print(f"Color match is set to disable-match, not using color match")
        else:
            print(f"Color match args is not provided, not using color match")

        simple_scale_method = "bilinear"
        simple_crop_method = "disabled"
        if simple_scale_Args:
            simple_scale_method = simple_scale_Args.get("upscale_method", simple_scale_method)
            simple_crop_method = simple_scale_Args.get("crop", simple_crop_method)

        # move each control_images into an array for easy processing if it is not None
        control_images = []
        if control_images_1 is not None:
            control_images.append(control_images_1)
        if control_images_2 is not None:
            control_images.append(control_images_2)
        if control_images_3 is not None:
            control_images.append(control_images_3)
        if control_images_4 is not None:
            control_images.append(control_images_4)
        if control_images_5 is not None:
            control_images.append(control_images_5)

        batch_starting_images = starting_images
        wanVideoDecode = WanVideoDecode()

        if batch_length >= total_frames:
            number_of_batches = 1
            batch_length = total_frames
        else:
            batch_length_no_addition = batch_length - 1
            number_of_batches = math.ceil((total_frames - 1) / batch_length_no_addition) # ensure round up
            batch_length = math.ceil((total_frames - 1) / number_of_batches) # ensure round up
            # batch length needs to be a number divisible by 4, then we add 1
            batch_length_div_4 = math.ceil(batch_length / 4)
            batch_length = int(batch_length_div_4 * 4 + 1)

        print(f"Total frames: {total_frames}, Batch length: {batch_length}, Number of batches: {number_of_batches}, Number of control images: {len(control_images)}")
        batchCount = 0
        remaining_frames = total_frames
        generated_images = None
        isFirstBatch = True

        # Pre-allocate generated_images tensor for better memory efficiency
        # Estimate final shape based on first batch to avoid repeated reallocations
        # Account for dropping the first image in batch 1
        try:
            # Get device from starting_images or use default
            target_device = starting_images.device if starting_images is not None else torch.device('cpu')
            
            # We'll allocate this after processing the first batch to know the exact dimensions
            preallocated_images = None
            preallocated_offset = 0
            # Adjust total frames to account for dropping first image in batch 1
            adjusted_total_frames = total_frames - 1
        except Exception as e:
            print(f"Warning: Could not determine device for pre-allocation: {e}")
            target_device = torch.device('cpu')
            preallocated_images = None
            preallocated_offset = 0
            adjusted_total_frames = total_frames - 1

        batchSeed = seed

        totalVaceTime = 0.0
        totalSamplerTime = 0.0
        totalColorMatchTime = 0.0
        totalRestoreFaceTime = 0.0
        totalMergeImageTime = 0.0
        totalDecodeTime = 0.0
        totalStartingimagesTotalTime = 0.0

        for start_idx in range(0, total_frames, batch_length - 1): # we always throw away the last generated frame
            batchCount += 1
            # Stop the loop early if remaining_frames <= 0
            if remaining_frames <= 0:
                break
            startBatchTime = time.time()  # Start timing  
            print(f"-------- WanVideoLoopingSampler batch {batchCount}/{number_of_batches} START -------------")
            endIndex = min(start_idx + batch_length, total_frames)
            numberOfFramesForBatch = endIndex - start_idx
            remaining_frames -= numberOfFramesForBatch
            if remaining_frames > 0:
                remaining_frames += 1 # allow for the fact that we drop a frame each batch except the last
            isLastBatch = (remaining_frames <= 0)
            batchControlAfterStart = False
            batchControlsEnd = endIndex
            numberOfBatchStartingImages = batch_starting_images.shape[0] if batch_starting_images is not None else 0
            batchControlsStart = max(0, start_idx - numberOfBatchStartingImages)
            totalNumberOfFramesForBatchEmbeds = batchControlsEnd - batchControlsStart
            batchstartingImagesAreGenerated = not isFirstBatch
            print(f"WanVideoLoopingSampler Processing batch {batchCount}/{number_of_batches} from frame {start_idx} to {endIndex} = {numberOfFramesForBatch}. remaining_frames: {remaining_frames}. isLastBatch: {isLastBatch}")

            startVaceTime = time.time()
            batchImageEmbeds = self.generateWanVaceEmbeds(
                totalFramesToGenerate=totalNumberOfFramesForBatchEmbeds,
                starting_images=batch_starting_images,
                control_images=control_images,
                controlImageStartindex=batchControlsStart,
                controlImageEndIndex=batchControlsEnd,
                vae=vae,
                width=width,
                height=height,
                control_after_start=batchControlAfterStart,
                startingImagesAreGenerated=batchstartingImagesAreGenerated,
                batchIndex=batchCount-1
            )

            ClearMemory(
                gc_collect=True,
                empty_cache=True,
                unload_all_models=True
            )
            endVaceTime = time.time()
            batchVaceTime = endVaceTime - startVaceTime
            totalVaceTime += batchVaceTime

            batchForceOffload = isLastBatch
            if isLastBatch:
                batchForceOffload = force_offload
            
            startSamplerTime = time.time()
            wanVideoSampler = WanVideoSampler()
            batchSampledResult = wanVideoSampler.process(
                model=model,
                image_embeds=batchImageEmbeds,
                shift=shift,
                steps=steps,
                cfg=cfg,
                seed=batchSeed,
                scheduler=scheduler,
                riflex_freq_index=riflex_freq_index,
                text_embeds=text_embeds,
                force_offload=batchForceOffload,
                samples=samples,
                feta_args=feta_args,
                denoise_strength=denoise_strength,
                context_options=context_options,
                cache_args=cache_args,
                teacache_args=teacache_args,
                flowedit_args=flowedit_args,
                batched_cfg=batched_cfg,
                slg_args=slg_args,
                rope_function=rope_function,
                loop_args=loop_args,
                experimental_args=experimental_args,
                sigmas=sigmas,
                unianimate_poses=unianimate_poses,
                fantasytalking_embeds=fantasytalking_embeds,
                uni3c_embeds=uni3c_embeds,
                multitalk_embeds=multitalk_embeds,
                freeinit_args=freeinit_args,
                start_step=start_step,
                end_step=end_step,
                add_noise_to_samples=add_noise_to_samples
            )
            
            # Clean up batch image embeds immediately after sampling
            del batchImageEmbeds

            endSamplerTime = time.time()
            batchSamplerTime = endSamplerTime - startSamplerTime
            totalSamplerTime += batchSamplerTime
            
            print(f"WanVideoLoopingSampler Processed batch {batchCount}/{number_of_batches} from frame {start_idx} to {endIndex} = {numberOfFramesForBatch}. remaining_frames: {remaining_frames}. isLastBatch: {isLastBatch}, batchSeed: {batchSeed}, batchControlAfterStart: {batchControlAfterStart}, batchControlsStart: {batchControlsStart}, batchControlsEnd: {batchControlsEnd}, numberOfBatchStartingImages: {numberOfBatchStartingImages}, batchstartingImagesAreGenerated: {batchstartingImagesAreGenerated}")
            # decode the created samples
            batchDecodeStartTime = time.time()
            batchSamplesToDecode = batchSampledResult[1] # the 2nd item in the returned tuple is the denoised samples
            batchDecodedSamples = wanVideoDecode.decode(vae, batchSamplesToDecode, decodeTile, decodeTileX, decodeTileY, decodeTileStrideX, decodeTileStrideY)
            batchDecodedSampleimages = batchDecodedSamples[0] # get the first item from the tuple
            
            # Clean up intermediate decode results
            del batchSamplesToDecode, batchDecodedSamples

            # as the decoded images include the prefix data and an extra unwanted frame at the end,
            # we only want the last numberOfFramesForBatch frames, AND we want to remove the last frame IF we are not on the last batch
            # For batch 1, also drop the first image
            # Optimize: combine slicing operations to avoid intermediate tensor creation
            end_idx = None if isLastBatch else -1
            start_idx_adjustment = 1 if batchCount == 1 else 0  # Drop first image in batch 1
            batchDecodedSampleimages = batchDecodedSampleimages[-(numberOfFramesForBatch - start_idx_adjustment):end_idx]

            gc.collect()

            batchDecodeEndTime = time.time()
            batchDecodeTime = batchDecodeEndTime - batchDecodeStartTime
            totalDecodeTime += batchDecodeTime

            batchColorMatchTime = 0.0
             # color match the generated frames
            if color_match_strength > 0.0:       
                start_time = time.time()  # Start timing             
                if color_match_source is None:
                    colorMatchFrameIndex = 0
                    if color_match_used_source == "provided-frame":
                        colorMatchFrameIndex = color_match_provided_frame

                    # ensure colorMatchFrameIndex is less than the number of frames in batchDecodedSampleimages
                    # and use the last frame if it is greater
                    colorMatchFrameIndex = min(colorMatchFrameIndex, batchDecodedSampleimages.shape[0] - 1)
                    # use the calculated frame of the decoded samples as the color match source
                    color_match_source = batchDecodedSampleimages[colorMatchFrameIndex].unsqueeze(0).clone()
                print(f"WanVideoLoopingSampler Color matching prefix samples with WanColorMatch")
                color_match_result = colormatch(color_match_source, batchDecodedSampleimages, color_match_method, color_match_strength)
                batchDecodedSampleimages = color_match_result[0]
                
                # Clean up color match result
                del color_match_result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                end_time = time.time()  # End timing
                batchColorMatchTime = end_time - start_time
                print(f"WanVideoLoopingSampler Execution time for color matching block: {batchColorMatchTime:.4f} seconds")
            totalColorMatchTime += batchColorMatchTime

            batchRestoreFaceTime = 0.0
            if restore_face and use_restore_face and reactor:
                start_time = time.time()  # Start timing
                faceDetectionModel = restore_face.get("facedetection", "retinaface_resnet50")
                faceRestoreModel = restore_face.get("model", "codeformer")
                faceRestoreVisibility = restore_face.get("visibility", 1.0)
                faceRestoreCodeformerWeight = restore_face.get("codeformer_weight", 0.5)
                faceRestoreSetMaxBatchSize = restore_face.get("max_batch_size", 20)
                faceRestoreUseFp16Onnx = restore_face.get("useFp16_Onnx", False)
                print(f"WanVideoLoopingSampler restoring faces in batchDecodedSampleimages with faceRestoreModel: {faceRestoreModel}, faceRestoreVisibility: {faceRestoreVisibility}, faceRestoreCodeformerWeight: {faceRestoreCodeformerWeight}, faceDetectionModel: {faceDetectionModel}, faceRestoreSetMaxBatchSize: {faceRestoreSetMaxBatchSize}, faceRestoreUseFp16Onnx: {faceRestoreUseFp16Onnx}")
                batchDecodedSampleimages = reactor.restore_face(
                    self,
                    input_image=batchDecodedSampleimages,
                    face_restore_model=faceRestoreModel,
                    face_restore_visibility=faceRestoreVisibility,
                    codeformer_weight=faceRestoreCodeformerWeight,
                    facedetection=faceDetectionModel,
                    setMaxBatchSize=faceRestoreSetMaxBatchSize,
                    use_fp16=faceRestoreUseFp16Onnx
                 )
                reactor.unload_face_restore_model(self)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                end_time = time.time()  # End timing
                batchRestoreFaceTime = end_time - start_time
                print(f"Execution time for face restoration block: {batchRestoreFaceTime:.4f} seconds")
            totalRestoreFaceTime += batchRestoreFaceTime
            
            batchMergeImagesStartTime = time.time()  # Start timing
            # Memory optimization: Pre-allocate or concatenate efficiently
            if generated_images is None:
                print(f"WanVideoLoopingSampler batch {batchCount}/{number_of_batches} creating generated_images with shape {batchDecodedSampleimages.shape}")
                
                # Pre-allocate the full tensor if we can estimate the final size
                if preallocated_images is None:
                    try:
                        # Calculate total expected frames (accounting for overlap removal and dropping first image in batch 1)
                        total_expected_frames = adjusted_total_frames
                        final_shape = (total_expected_frames,) + batchDecodedSampleimages.shape[1:]
                        
                        # Pre-allocate on the same device as the batch images
                        preallocated_images = torch.empty(final_shape, dtype=batchDecodedSampleimages.dtype, device=batchDecodedSampleimages.device)
                        print(f"Pre-allocated tensor with shape {final_shape}")
                    except Exception as e:
                        print(f"Could not pre-allocate tensor: {e}, falling back to concatenation")
                        preallocated_images = None
                
                if preallocated_images is not None:
                    # Use pre-allocated tensor
                    frames_to_copy = batchDecodedSampleimages.shape[0]
                    preallocated_images[preallocated_offset:preallocated_offset + frames_to_copy] = batchDecodedSampleimages
                    preallocated_offset += frames_to_copy
                    generated_images = preallocated_images  # This will be sliced at the end
                else:
                    # Fallback to regular assignment
                    generated_images = batchDecodedSampleimages
            else:
                print(f"WanVideoLoopingSampler batch {batchCount}/{number_of_batches} extending generated_images with shape {batchDecodedSampleimages.shape}")
                
                if preallocated_images is not None:
                    # Use pre-allocated tensor
                    frames_to_copy = batchDecodedSampleimages.shape[0]
                    preallocated_images[preallocated_offset:preallocated_offset + frames_to_copy] = batchDecodedSampleimages
                    preallocated_offset += frames_to_copy
                else:
                    # Fallback to concatenation - use torch.cat for efficiency
                    generated_images = torch.cat((generated_images, batchDecodedSampleimages), dim=0)
            
            # Clean up batch decoded images immediately after use
            del batchDecodedSampleimages
            
            # Clean up batch sampled result
            del batchSampledResult

            batchMergeImagesEndTime = time.time()  # Start timing
            batchMergeImageTime = batchMergeImagesEndTime - batchMergeImagesStartTime
            totalMergeImageTime += batchMergeImageTime
            
            print(f"WanVideoLoopingSampler batch {batchCount}/{number_of_batches} generated_images shape: {generated_images.shape if preallocated_images is None else preallocated_images[:preallocated_offset].shape}")

            batchStartingimagesTotalTime = 0
            # get overlap_length frames from the end of the generated images
            if not isLastBatch:
                batchStartingImagesStartTime = time.time()  # Start timing
                print(f"WanVideoLoopingSampler batch {batchCount}/{number_of_batches} getting {overlap_length} overlap frames from the end of the generated images")
                if preallocated_images is not None:
                    # Get overlap from pre-allocated tensor
                    batch_starting_images = preallocated_images[preallocated_offset - overlap_length:preallocated_offset].clone()
                else:
                    batch_starting_images = generated_images[-overlap_length:].clone()
                batchStartingImagesEndTime = time.time()  # End timing
                batchStartingimagesTotalTime = batchStartingImagesEndTime - batchStartingImagesStartTime
            totalStartingimagesTotalTime += batchStartingimagesTotalTime

            if seed_batch_control == "Randomize":
                # Randomize the seed for each batch
                batchSeed = random.randint(0, 0xffffffffffffffff)
            else:
                batchSeed = (batchSeed + seed_adjust) % 0xffffffffffffffff

            isFirstBatch = False

            ClearMemory(
                gc_collect=True,
                empty_cache=True,
                unload_all_models=True
            )
            
            endBatchTime = time.time()  # End timing
            elapsedBatchTime = endBatchTime - startBatchTime
            print(f"-------- WanVideoLoopingSampler batch {batchCount}/{number_of_batches} timing stats:")
            print(f"  VACE time: {batchVaceTime:.4f} seconds")
            print(f"  Sampler time: {batchSamplerTime:.4f} seconds")
            print(f"  Decode time: {batchDecodeTime:.4f} seconds")
            print(f"  Color match time: {batchColorMatchTime:.4f} seconds")
            print(f"  Restore face time: {batchRestoreFaceTime:.4f} seconds")
            print(f"  Merge images time: {batchMergeImageTime:.4f} seconds")
            print(f"  Starting images time: {batchStartingimagesTotalTime:.4f} seconds")
            print(f"  Total Batch time: {elapsedBatchTime:.4f} seconds")
            print(f"-------- WanVideoLoopingSampler batch {batchCount}/{number_of_batches} COMPLETED in {elapsedBatchTime:.4f} seconds -------------")
            
        # Final cleanup and return the properly sized tensor
        if preallocated_images is not None:
            # Return only the frames we actually used
            generated_images = preallocated_images[:preallocated_offset].clone()
            del preallocated_images
                    
        # Final garbage collection
        ClearMemory(
            gc_collect=True,
            empty_cache=True,
            unload_all_models=True
        )

        overallEndTime = time.time()  # Start timing
        overallTotalTime = overallEndTime - overallStartTime

        print(f"-------- WanVideoLoopingSampler FINAL timing stats:")
        print(f"  VACE time: {totalVaceTime:.4f} seconds")
        print(f"  Sampler time: {totalSamplerTime:.4f} seconds")
        print(f"  Decode time: {totalDecodeTime:.4f} seconds")
        print(f"  Color match time: {totalColorMatchTime:.4f} seconds")
        print(f"  Restore face time: {totalRestoreFaceTime:.4f} seconds")
        print(f"  Merge images time: {totalMergeImageTime:.4f} seconds")
        print(f"  Starting images time: {totalStartingimagesTotalTime:.4f} seconds")
        print(f"  Total time: {overallTotalTime:.4f} seconds")
        print(f"!!!!!! ------ WanVideoLoopingSampler final output generated_images shape: {generated_images.shape} COMPLETED in {overallTotalTime:.4f} seconds ------ !!!!!!")
        # Return the generated images as a tuple
        return (generated_images,)


#region VideoDecode
class WanVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": (
                        "Drastically reduces memory use but will introduce seams at tile stride boundaries. "
                        "The location and number of seams is dictated by the tile stride size. "
                        "The visibility of seams can be controlled by increasing the tile size. "
                        "Seams become less obvious at 1.5x stride and are barely noticeable at 2x stride size. "
                        "Which is to say if you use a stride width of 160, the seams are barely noticeable with a tile width of 320."
                    )}),
                    "tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile width in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile height in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride width in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride height in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    },
                    "optional": {
                        "normalization": (["default", "minmax"], {"advanced": True}),
                    }
                }

    @classmethod
    def VALIDATE_INPUTS(s, tile_x, tile_y, tile_stride_x, tile_stride_y):
        if tile_x <= tile_stride_x:
            return "Tile width must be larger than the tile stride width."
        if tile_y <= tile_stride_y:
            return "Tile height must be larger than the tile stride height."
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, normalization="default"):
        mm.soft_empty_cache()
        images = samples.get("images", None)
        if images is not None:
            print("WanVideoDecode - samples contains pre-decoded images, returning it directly.")
            return (images,)
    
        video = samples.get("video", None)
        if video is not None:
            video.clamp_(-1.0, 1.0)
            video.add_(1.0).div_(2.0)
            return video.cpu().float(),
        latents = samples["samples"]
        end_image = samples.get("end_image", None)
        has_ref = samples.get("has_ref", False)
        drop_last = samples.get("drop_last", False)
        is_looped = samples.get("looped", False)

        vae.to(device)

        latents = latents.to(device = device, dtype = vae.dtype)

        mm.soft_empty_cache()

        if has_ref:
            latents = latents[:, :, 1:]
        if drop_last:
            latents = latents[:, :, :-1]

        if type(vae).__name__ == "TAEHV":      
            images = vae.decode_video(latents.permute(0, 2, 1, 3, 4))[0].permute(1, 0, 2, 3)
            images = torch.clamp(images, 0.0, 1.0)
            images = images.permute(1, 2, 3, 0).cpu().float()
            return (images,)
        else:
            if end_image is not None:
                enable_vae_tiling = False
            images = vae.decode(latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//8, tile_y//8), tile_stride=(tile_stride_x//8, tile_stride_y//8))[0]
            vae.model.clear_cache()
        
        images = images.cpu().float()

        if normalization == "minmax":
            images.sub_(images.min()).div_(images.max() - images.min())
        else:  
            images.clamp_(-1.0, 1.0)
            images.add_(1.0).div_(2.0)
        
        if is_looped:
            temp_latents = torch.cat([latents[:, :, -3:]] + [latents[:, :, :2]], dim=2)
            temp_images = vae.decode(temp_latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))[0]
            temp_images = temp_images.cpu().float()
            temp_images = (temp_images - temp_images.min()) / (temp_images.max() - temp_images.min())
            images = torch.cat([temp_images[:, 9:].to(images), images[:, 5:]], dim=1)

        if end_image is not None: 
            images = images[:, 0:-1]

        vae.model.clear_cache()
        vae.to(offload_device)
        mm.soft_empty_cache()

        images.clamp_(0.0, 1.0)

        return (images.permute(1, 2, 3, 0),)

#region VideoEncode
class WanVideoEncodeLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "images": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes a batch of images individually to create a latent video batch where each video is a single frame, useful for I2V init purposes, for example as multiple context window inits"

    def encode(self, vae, images, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, latent_strength=1.0):
        vae.to(device)

        images = images.clone()

        B, H, W, C = images.shape
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
            images = common_upscale(images.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

        if images.shape[-1] == 4:
            images = images[..., :3]
        images = images.to(vae.dtype).to(device) * 2.0 - 1.0

        latent_list = []
        for img in images:
            if enable_vae_tiling and tile_x is not None:
                latent = vae.encode(img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3), device=device, tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))
            else:
                latent = vae.encode(img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3), device=device, tiled=enable_vae_tiling)
            vae.model.clear_cache()
            if latent_strength != 1.0:
                latent *= latent_strength
            latent_list.append(latent.squeeze(0).cpu())
        latents_out = torch.stack(latent_list, dim=0)

        log.info(f"WanVideoEncode: Encoded latents shape {latents_out.shape}")
        vae.to(offload_device)
        mm.soft_empty_cache()

        return ({"samples": latents_out},)

class WanVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                    "optional": {
                        "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for leapfusion I2V where some noise can add motion and give sharper results"}),
                        "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for leapfusion I2V where lower values allow for more motion"}),
                        "mask": ("MASK", ),
                        "passthrough": ("BOOLEAN", {"default": False, "tooltip": "The node will do nothing and return a None value"}),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, vae, image, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, noise_aug_strength=0.0, latent_strength=1.0, mask=None,
               passthrough=False):
        if passthrough:
            print("WanVideoEncode passthrough enabled, returning None")
            return {"images": image},
        
        vae.to(device)
        image = image.clone()

        B, H, W, C = image.shape
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
            image = common_upscale(image.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W        

        if noise_aug_strength > 0.0:
            image = add_noise_to_reference_video(image, ratio=noise_aug_strength)

        if isinstance(vae, TAEHV):
            latents = vae.encode_video(image.permute(0, 2, 1, 3, 4), parallel=False)# B, T, C, H, W
            latents = latents.permute(0, 2, 1, 3, 4)
        else:
            latents = vae.encode(image * 2.0 - 1.0, device=device, tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))
            vae.model.clear_cache()
            vae.to(offload_device)
        if latent_strength != 1.0:
            latents *= latent_strength

        log.info(f"WanVideoEncode: Encoded latents shape {latents.shape}")
        mm.soft_empty_cache()
 
        return ({"samples": latents, "noise_mask": mask},)

NODE_CLASS_MAPPINGS = {
    "WanVideoSampler": WanVideoSampler,
    "WanVideoLoopingSampler": WanVideoLoopingSampler,
    "WanVideoDecode": WanVideoDecode,
    "WanVideoTextEncode": WanVideoTextEncode,
    "WanVideoTextEncodeSingle": WanVideoTextEncodeSingle,
    "WanVideoClipVisionEncode": WanVideoClipVisionEncode,
    "WanVideoImageToVideoEncode": WanVideoImageToVideoEncode,
    "WanVideoEncode": WanVideoEncode,
    "WanVideoEncodeLatentBatch": WanVideoEncodeLatentBatch,
    "WanFaceRestoreArgs_VTS": WanFaceRestoreArgs_VTS,
    "WanColorMatchArgs_VTS": WanColorMatchArgs_VTS,
    "WanVideoDecodeSettings_VTS": WanVideoDecodeSettings_VTS,
    "WanVideoEncodeSettings_VTS": WanVideoEncodeSettings_VTS,
    "WanVideoUpscaleWithModelSettings_VTS": WanVideoUpscaleWithModelSettings_VTS,
    "WanVideoSimpleScaleSettings_VTS": WanVideoSimpleScaleSettings_VTS,
    "WanVideoEmptyEmbeds": WanVideoEmptyEmbeds,
    "WanVideoEnhanceAVideo": WanVideoEnhanceAVideo,
    "WanVideoContextOptions": WanVideoContextOptions,
    "WanVideoTextEmbedBridge": WanVideoTextEmbedBridge,
    "WanVideoTextEmbedReverseBridge": WanVideoTextEmbedReverseBridge,
    "WanVideoLoopingControlImagesOptions": WanVideoLoopingControlImagesOptions,
    "WanVideoFlowEdit": WanVideoFlowEdit,
    "WanVideoControlEmbeds": WanVideoControlEmbeds,
    "WanVideoSLG": WanVideoSLG,
    "WanVideoLoopArgs": WanVideoLoopArgs,
    "WanVideoSetBlockSwap": WanVideoSetBlockSwap,
    "WanVideoExperimentalArgs": WanVideoExperimentalArgs,
    "WanVideoVACEEncode": WanVideoVACEEncode,
    "WanVideoPhantomEmbeds": WanVideoPhantomEmbeds,
    "WanVideoRealisDanceLatents": WanVideoRealisDanceLatents,
    "WanVideoApplyNAG": WanVideoApplyNAG,
    "VTS_WanVideoApplyMultiNAG": VTS_WanVideoApplyMultiNAG,
    "WanVideoMiniMaxRemoverEmbeds": WanVideoMiniMaxRemoverEmbeds,
    "WanVideoFreeInitArgs": WanVideoFreeInitArgs,
    "WanVideoSetRadialAttention": WanVideoSetRadialAttention,
    "WanVideoBlockList": WanVideoBlockList,
    "WanVideoTextEncodeCached": WanVideoTextEncodeCached,
    "VTS_WanVideoMultiTextEncodeCached": VTS_WanVideoMultiTextEncodeCached,
    "WanVideoAddExtraLatent": WanVideoAddExtraLatent,
    "WanVideoScheduler": WanVideoScheduler,
    "WanVideoAddStandInLatent": WanVideoAddStandInLatent,
    "WanVideoAddControlEmbeds": WanVideoAddControlEmbeds,
    "WanVideoAddMTVMotion": WanVideoAddMTVMotion,
    "WanVideoRoPEFunction": WanVideoRoPEFunction,
    "WanVideoAddPusaNoise": WanVideoAddPusaNoise
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoSampler": "WanVideo Sampler",
    "WanVideoLoopingSampler": "WanVideo Looping Sampler",
    "WanVideoDecode": "WanVideo Decode",
    "WanVideoTextEncode": "WanVideo TextEncode",
    "WanVideoTextEncodeSingle": "WanVideo TextEncodeSingle",
    "WanVideoTextImageEncode": "WanVideo TextImageEncode (IP2V)",
    "WanVideoClipVisionEncode": "WanVideo ClipVision Encode",
    "WanVideoImageToVideoEncode": "WanVideo ImageToVideo Encode",
    "WanVideoEncode": "WanVideo Encode",
    "WanVideoEncodeLatentBatch": "WanVideo Encode Latent Batch",
    "WanFaceRestoreArgs_VTS": "WanFace Restore Args VTS",
    "WanColorMatchArgs_VTS": "WanColor Match Args VTS",
    "WanVideoDecodeSettings_VTS": "WanVideo Decode Settings VTS",
    "WanVideoEncodeSettings_VTS": "WanVideo Encode Settings VTS",
    "WanVideoUpscaleWithModelSettings_VTS": "WanVideo Upscale With Model Settings VTS",
    "WanVideoSimpleScaleSettings_VTS": "WanVideo Simple Scale Settings VTS",
    "WanVideoEmptyEmbeds": "WanVideo Empty Embeds",
    "WanVideoEnhanceAVideo": "WanVideo Enhance-A-Video",
    "WanVideoContextOptions": "WanVideo Context Options",
    "WanVideoTextEmbedBridge": "WanVideo TextEmbed Bridge",
    "WanVideoTextEmbedReverseBridge": "WanVideo TextEmbed Reverse Bridge",
    "WanVideoLoopingControlImagesOptions": "WanVideo Looping Control Images Options",
    "WanVideoFlowEdit": "WanVideo FlowEdit",
    "WanVideoControlEmbeds": "WanVideo Control Embeds",
    "WanVideoSLG": "WanVideo SLG",
    "WanVideoLoopArgs": "WanVideo Loop Args",
    "WanVideoSetBlockSwap": "WanVideo Set BlockSwap",
    "WanVideoExperimentalArgs": "WanVideo Experimental Args",
    "WanVideoVACEEncode": "WanVideo VACE Encode",
    "WanVideoPhantomEmbeds": "WanVideo Phantom Embeds",
    "WanVideoRealisDanceLatents": "WanVideo RealisDance Latents",
    "WanVideoApplyNAG": "WanVideo Apply NAG",
    "VTS_WanVideoApplyMultiNAG": "VTS WanVideo Apply Multi NAG",
    "WanVideoMiniMaxRemoverEmbeds": "WanVideo MiniMax Remover Embeds",
    "WanVideoFreeInitArgs": "WanVideo Free Init Args",
    "WanVideoSetRadialAttention": "WanVideo Set Radial Attention",
    "WanVideoBlockList": "WanVideo Block List",
    "WanVideoTextEncodeCached": "WanVideo TextEncode Cached",
    "VTS_WanVideoMultiTextEncodeCached": "VTS WanVideo MultiTextEncode Cached",
    "WanVideoAddExtraLatent": "WanVideo Add Extra Latent",
    "WanVideoAddStandInLatent": "WanVideo Add StandIn Latent",
    "WanVideoAddControlEmbeds": "WanVideo Add Control Embeds",
    "WanVideoAddMTVMotion": "WanVideo MTV Crafter Motion",
    "WanVideoRoPEFunction": "WanVideo RoPE Function",
    "WanVideoAddPusaNoise": "WanVideo Add Pusa Noise",
}
