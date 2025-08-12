import os
import sys
import random
import torch
import gc
from ..utils import log, print_memory, fourier_filter, optimized_scale, setup_radial_attention, compile_model
import math
from tqdm import tqdm
import numpy as np
import importlib.util
import comfy
import comfy.utils
import logging
import time
import traceback

from spandrel import ModelLoader, ImageModelDescriptor
from spandrel.__helpers.size_req import pad_tensor
from comfy import model_management

from ..wanvideo.modules.model import rope_params
from ..wanvideo.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from ..nodes import WanVideoDecode, WanVideoEncode, optimized_scale


from ..fp8_optimization import convert_linear_with_lora_and_scale, remove_lora_from_module
from ..wanvideo.schedulers.scheduling_flow_match_lcm import FlowMatchLCMScheduler
from ..gguf.gguf import set_lora_params
from einops import rearrange

from ..enhance_a_video.globals import disable_enhance

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
from comfy.clip_vision import clip_preprocess, ClipVisionModel
from comfy.cli_args import args, LatentPreviewMethod

import node_helpers
from nodes import MAX_RESOLUTION

try:
    # Get the absolute path to the ComfyUI-ReActor folder
    comfyui_reactor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ComfyUI-ReActor'))

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



def colormatch(image_ref, image_target, method, strength=1.0, editInPlace=False, gc_interval=50):
    try:
        from color_matcher import ColorMatcher
    except ImportError:
        raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
    
    # Early validation
    if image_ref.dim() != 4 or image_target.dim() != 4:
        raise ValueError("ColorMatch: Expected 4D tensors (batch, height, width, channels)")
    
    batch_size = image_target.size(0)
    ref_batch_size = image_ref.size(0)
    
    # Validate batch sizes early
    if ref_batch_size > 1 and ref_batch_size != batch_size:
        raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")
    
    # Move to CPU efficiently (avoid redundant moves)
    if image_ref.device != torch.device('cpu'):
        image_ref = image_ref.cpu()
    if image_target.device != torch.device('cpu'):
        image_target = image_target.cpu()
    
    # Handle output tensor allocation
    if editInPlace:
        out = image_target
    else:
        out = torch.empty_like(image_target, dtype=torch.float32, device='cpu')
    
    # Initialize ColorMatcher once
    cm = ColorMatcher()
    
    # Process each image in the batch
    for i in range(batch_size):
        # Get individual images (avoid squeeze - use direct indexing)
        target_img = image_target[i]  # Shape: [H, W, C]
        ref_img = image_ref[0] if ref_batch_size == 1 else image_ref[i]  # Shape: [H, W, C]
        
        # Convert to numpy only when needed
        target_np = target_img.numpy()
        ref_np = ref_img.numpy()
        
        try:
            # Perform color matching
            result_np = cm.transfer(src=target_np, ref=ref_np, method=method)
            
            # Apply strength multiplier efficiently
            if strength != 1.0:
                result_np = target_np + strength * (result_np - target_np)
            
            # Convert back to tensor and update output
            result_tensor = torch.from_numpy(result_np)
            
            if editInPlace:
                image_target[i].copy_(result_tensor)
            else:
                out[i].copy_(result_tensor)
            
            # Clean up intermediate variables
            del target_np, ref_np, result_np, result_tensor
            
            # Garbage collection at intervals
            if gc_interval > 0 and (i + 1) % gc_interval == 0:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Error occurred during transfer for image {i}: {e}")
            # Continue processing other images rather than breaking
            continue
    
    # Ensure output is float32 and properly clamped
    if not editInPlace and out.dtype != torch.float32:
        out = out.to(torch.float32)
    out.clamp_(0, 1)
    
    return (out,)

def generate_timestep_matrix(
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        shrink_interval_with_mask=False,
        denoise_strength=1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert ar_step >= min_ar_step, f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, casual_block_size, num_frames_block, base_num_frames_block)
        # print(f"generate_timestep_matrix: num_frames:{num_frames}, num_iterations:{num_iterations}, step_template:{step_template.shape}, base_num_frames:{base_num_frames}, ar_step:{ar_step}, num_pre_ready:{num_pre_ready}, casual_block_size:{casual_block_size}, num_frames_block:{num_frames_block}, base_num_frames_block:{base_num_frames_block}")
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        initial_value = num_iterations - (num_iterations * denoise_strength)
        # set all pre_row values to the initial value
        pre_row[:] = initial_value
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        row_count = 0
        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)
            # print(f"generate_timestep_matrix new_row[{row_count}]: {new_row}")
            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row
            row_count += 1

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            valid_interval = [(s * casual_block_size, e * casual_block_size) for s, e in valid_interval]

        # print(f"generate_timestep_matrix = step_matrix: {step_matrix.shape}, step_index: {step_index.shape}, step_update_mask: {step_update_mask.shape}, valid_interval: {valid_interval}")
        return step_matrix, step_index, step_update_mask, valid_interval

#region Sampler
class WanVideoDiffusionForcingSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "addnoise_condition": ("INT", {"default": 10, "min": 0, "max": 1000, "tooltip": "Improves consistency in long video generation"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (["unipc", "unipc/beta", "euler", "euler/beta", "lcm", "lcm/beta"],
                    {
                        "default": 'unipc'
                    }),
            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "prefix_samples": ("LATENT", {"tooltip": "prefix latents"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "denoising_skew" : ("FLOAT", {"default": 0.0, "min": -100.0, "max": 10.0, "step": 0.001}),
                "cache_args": ("CACHEARGS", ),
                "slg_args": ("SLGARGS", ),
                "rope_function": (["default", "comfy"], {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile"}),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, model, text_embeds, image_embeds, shift, fps, steps, addnoise_condition, cfg, seed, scheduler, 
        force_offload=True, samples=None, prefix_samples=None, denoise_strength=1.0, denoising_skew=0.0, slg_args=None, rope_function="default", cache_args=None, teacache_args=None, 
        experimental_args=None, unianimate_poses=None, noise_reduction_factor=1.0, denoising_multiplier=1.0, denoising_multiplier_end=None):
        #assert not (context_options and teacache_args), "Context options cannot currently be used together with teacache."
        if denoising_multiplier_end is None:
            denoising_multiplier_end = denoising_multiplier
        patcher = model
        model = model.model
        transformer = model.diffusion_model
        dtype = model["dtype"]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        gguf = model["gguf"]
        transformer_options = patcher.model_options.get("transformer_options", None)

        if len(patcher.patches) != 0 and transformer_options.get("linear_patched", False) is True:
            log.info(f"Using {len(patcher.patches)} LoRA weight patches for WanVideo model")
            if not gguf:
                convert_linear_with_lora_and_scale(transformer, patches=patcher.patches)
            else:
                set_lora_params(transformer, patcher.patches)
        else:
            log.info("Unloading all LoRAs")
            remove_lora_from_module(transformer)

        #torch.compile
        if model["auto_cpu_offload"] is False:
            transformer = compile_model(transformer, model["compile_args"])
        
        timestep_steps = int(steps/denoise_strength)

        timesteps = None
        if 'unipc' in scheduler:
            sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
            sample_scheduler.set_timesteps(timestep_steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        elif 'euler' in scheduler:
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
            sample_scheduler.set_timesteps(timestep_steps, device=device)
        elif 'lcm' in scheduler:
            sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
            sample_scheduler.set_timesteps(timestep_steps, device=device) 
        
        init_timesteps = sample_scheduler.timesteps
        timesteps = init_timesteps[:]
        if denoise_strength < 1.0:
            # steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1):] 
        
        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)
       
        clip_fea, clip_fea_neg = None, None
        vace_data, vace_context, vace_scale = None, None, None

        image_cond = image_embeds.get("image_embeds", None)

        target_shape = image_embeds.get("target_shape", None)
        if target_shape is None:
            raise ValueError("Empty image embeds must be provided for T2V (Text to Video")
        
        has_ref = image_embeds.get("has_ref", False)
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
                target_shape[0],
                target_shape[1] + 1 if has_ref else target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=seed_g)
        
        latent_video_length = noise.shape[1]  
        seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

        
               
        if samples is not None:
            input_samples = samples["samples"].squeeze(0).to(noise)
            if input_samples.shape[1] != noise.shape[1]:
                input_samples = torch.cat([input_samples[:, :1].repeat(1, noise.shape[1] - input_samples.shape[1], 1, 1), input_samples], dim=1)
            original_image = input_samples.to(device)
            if denoise_strength < 1.0:
                latent_timestep = timesteps[:1].to(noise)

                noise = (noise * latent_timestep / 1000 * noise_reduction_factor) + \
                        ((1 - latent_timestep / 1000) * input_samples)

            mask = samples.get("mask", None)
            if mask is not None:
                if mask.shape[2] != noise.shape[1]:
                    mask = torch.cat([torch.zeros(1, noise.shape[0], noise.shape[1] - mask.shape[2], noise.shape[2], noise.shape[3]), mask], dim=2)

        latents = noise.to(device)
        
        fps_embeds = None
        if hasattr(transformer, "fps_embedding"):
            fps = round(fps, 2)
            log.info(f"Model has fps embedding, using {fps} fps")
            fps_embeds = [fps]
            fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]

        prefix_video = prefix_samples["samples"].to(noise) if prefix_samples is not None else None
        prefix_video_latent_length = prefix_video.shape[2] if prefix_video is not None else 0
        if prefix_video is not None:
            log.info(f"Prefix video of length: {prefix_video_latent_length}")
            latents[:, :prefix_video_latent_length] = prefix_video[0]
        #base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_video_length
        base_num_frames=latent_video_length

        ar_step = 0
        causal_block_size = 1
        step_matrix, _, step_update_mask, valid_interval = generate_timestep_matrix(
                latent_video_length, init_timesteps, base_num_frames, ar_step, prefix_video_latent_length, causal_block_size,
                shrink_interval_with_mask=False, denoise_strength=denoise_strength
            )
        
        sample_schedulers = []
        for _ in range(latent_video_length):
            if 'unipc' in scheduler:
                sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
                sample_scheduler.set_timesteps(timestep_steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
            elif 'euler' in scheduler:
                sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
                sample_scheduler.set_timesteps(timestep_steps, device=device)
            elif 'lcm' in scheduler:
                sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
                sample_scheduler.set_timesteps(timestep_steps, device=device) 
            
            sample_schedulers.append(sample_scheduler)
        sample_schedulers_counter = [0] * latent_video_length

        unianim_data = None
        if unianimate_poses is not None:
            transformer.dwpose_embedding.to(device)
            transformer.randomref_embedding_pose.to(device)
            dwpose_data = unianimate_poses["pose"]
            dwpose_data = transformer.dwpose_embedding(
                (torch.cat([dwpose_data[:,:,:1].repeat(1,1,3,1,1), dwpose_data], dim=2)
                    ).to(device)).to(model["dtype"])
            log.info(f"UniAnimate pose embed shape: {dwpose_data.shape}")
            if dwpose_data.shape[2] > latent_video_length:
                log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is longer than the video length {latent_video_length}, truncating")
                dwpose_data = dwpose_data[:,:, :latent_video_length]
            elif dwpose_data.shape[2] < latent_video_length:
                log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is shorter than the video length {latent_video_length}, padding with last pose")
                pad_len = latent_video_length - dwpose_data.shape[2]
                pad = dwpose_data[:,:,:1].repeat(1,1,pad_len,1,1)
                dwpose_data = torch.cat([dwpose_data, pad], dim=2)
            dwpose_data_flat = rearrange(dwpose_data, 'b c f h w -> b (f h w) c').contiguous()
            
            random_ref_dwpose_data = None
            if image_cond is not None:
                random_ref_dwpose = unianimate_poses.get("ref", None)
                if random_ref_dwpose is not None:
                    random_ref_dwpose_data = transformer.randomref_embedding_pose(
                        random_ref_dwpose.to(device)
                        ).unsqueeze(2).to(model["dtype"]) # [1, 20, 104, 60]
                
            unianim_data = {
                "dwpose": dwpose_data_flat,
                "random_ref": random_ref_dwpose_data.squeeze(0) if random_ref_dwpose_data is not None else None,
                "strength": unianimate_poses["strength"],
                "start_percent": unianimate_poses["start_percent"],
                "end_percent": unianimate_poses["end_percent"]
            }
        
        disable_enhance() #not sure if this can work, disabling for now to avoid errors if it's enabled by another sampler

        freqs = None
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        if rope_function=="comfy":
            transformer.rope_embedder.k = 0
            transformer.rope_embedder.num_frames = latent_video_length
        else:
            d = transformer.dim // transformer.num_heads
            freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=0),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)

        if not isinstance(cfg, list):
            cfg = [cfg] * (steps +1)

        log.info(f"Seq len: {seq_len}")
           
        pbar = ProgressBar(steps)

        if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]: #default for latent2rgb
            from latent_preview import prepare_callback
        else:
            from ..latent_preview import prepare_callback #custom for tiny VAE previews
        callback = prepare_callback(patcher, steps)

        #blockswap init        
        if transformer_options is not None:
            block_swap_args = transformer_options.get("block_swap_args", None)

        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", False)
            for name, param in transformer.named_parameters():
                if "block" not in name:
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
        elif model["manual_offloading"]:
            transformer.to(device)

        # Initialize Cache if enabled
        transformer.enable_teacache = transformer.enable_magcache = False
        if teacache_args is not None: #for backward compatibility on old workflows
            cache_args = teacache_args
        if cache_args is not None:            
            transformer.cache_device = cache_args["cache_device"]
            if cache_args["cache_type"] == "TeaCache":
                log.info(f"TeaCache: Using cache device: {transformer.cache_device}")
                transformer.teacache_state.clear_all()
                transformer.enable_teacache = True
                transformer.rel_l1_thresh = cache_args["rel_l1_thresh"]
                transformer.teacache_start_step = cache_args["start_step"]
                transformer.teacache_end_step = len(timesteps)-1 if cache_args["end_step"] == -1 else cache_args["end_step"]
                transformer.teacache_use_coefficients = cache_args["use_coefficients"]
                transformer.teacache_mode = cache_args["mode"]
            elif cache_args["cache_type"] == "MagCache":
                log.info(f"MagCache: Using cache device: {transformer.cache_device}")
                transformer.magcache_state.clear_all()
                transformer.enable_magcache = True
                transformer.magcache_start_step = cache_args["start_step"]
                transformer.magcache_end_step = len(init_timesteps)-1 if cache_args["end_step"] == -1 else cache_args["end_step"]
                transformer.magcache_thresh = cache_args["magcache_thresh"]
                transformer.magcache_K = cache_args["magcache_K"]

        if slg_args is not None:
            transformer.slg_blocks = slg_args["blocks"]
            transformer.slg_start_percent = slg_args["start_percent"]
            transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None

        self.teacache_state = [None, None]
        self.teacache_state_source = [None, None]
        self.teacache_states_context = []

        if transformer.attention_mode == "radial_sage_attention":
            setup_radial_attention(transformer, transformer_options, latents, seq_len, latent_video_length)


        use_cfg_zero_star, use_fresca = False, False
        if experimental_args is not None:
            video_attention_split_steps = experimental_args.get("video_attention_split_steps", [])
            if video_attention_split_steps:
                transformer.video_attention_split_steps = [int(x.strip()) for x in video_attention_split_steps.split(",")]
            else:
                transformer.video_attention_split_steps = []
            use_zero_init = experimental_args.get("use_zero_init", True)
            use_cfg_zero_star = experimental_args.get("cfg_zero_star", False)
            zero_star_steps = experimental_args.get("zero_star_steps", 0)

            use_fresca = experimental_args.get("use_fresca", False)
            if use_fresca:
                fresca_scale_low = experimental_args.get("fresca_scale_low", 1.0)
                fresca_scale_high = experimental_args.get("fresca_scale_high", 1.25)
                fresca_freq_cutoff = experimental_args.get("fresca_freq_cutoff", 20)

        #region model pred
        def predict_with_cfg(z, cfg_scale, positive_embeds, negative_embeds, timestep, idx, image_cond=None, clip_fea=None, 
                             vace_data=None, unianim_data=None, teacache_state=None):
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=("fp8" in model["quantization"])):

                if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                    return latent_model_input*0, None

                nonlocal patcher
                current_step_percentage = idx / len(timesteps)
                # print(f"current_step_percentage[{idx}]: {current_step_percentage}")
                control_lora_enabled = False
                
                image_cond_input = image_cond
    
                base_params = {
                    'seq_len': seq_len,
                    'device': device,
                    'freqs': freqs,
                    't': timestep,
                    'current_step': idx,
                    'control_lora_enabled': control_lora_enabled,
                    'vace_data': vace_data,
                    'unianim_data': unianim_data,
                    'fps_embeds': fps_embeds,
                    "nag_params": text_embeds.get("nag_params", {}),
                    "nag_context": text_embeds.get("nag_prompt_embeds", None),
                }

                batch_size = 1

                if not math.isclose(cfg_scale, 1.0) and len(positive_embeds) > 1:
                    negative_embeds = negative_embeds * len(positive_embeds)

                
                #cond
                noise_pred_cond, teacache_state_cond = transformer(
                    [z], context=positive_embeds, y=[image_cond_input] if image_cond_input is not None else None,
                    clip_fea=clip_fea, is_uncond=False, current_step_percentage=current_step_percentage,
                    pred_id=teacache_state[0] if teacache_state else None,
                    **base_params
                )
                noise_pred_cond = noise_pred_cond[0].to(intermediate_device)
                if math.isclose(cfg_scale, 1.0):
                    if use_fresca:
                        noise_pred_cond = fourier_filter(
                            noise_pred_cond,
                            scale_low=fresca_scale_low,
                            scale_high=fresca_scale_high,
                            freq_cutoff=fresca_freq_cutoff,
                        )
                    return noise_pred_cond, [teacache_state_cond]
                #uncond
                noise_pred_uncond, teacache_state_uncond = transformer(
                    [z], context=negative_embeds, clip_fea=clip_fea_neg if clip_fea_neg is not None else clip_fea,
                    y=[image_cond_input] if image_cond_input is not None else None, 
                    is_uncond=True, current_step_percentage=current_step_percentage,
                    pred_id=teacache_state[1] if teacache_state else None,
                    **base_params
                )
                noise_pred_uncond = noise_pred_uncond[0].to(intermediate_device)
            
                #cfg

                #https://github.com/WeichenFan/CFG-Zero-star/
                if use_cfg_zero_star:
                    alpha = optimized_scale(
                        noise_pred_cond.view(batch_size, -1),
                        noise_pred_uncond.view(batch_size, -1)
                    ).view(batch_size, 1, 1, 1)
                else:
                    alpha = 1.0

                #https://github.com/WikiChao/FreSca
                if use_fresca:
                    filtered_cond = fourier_filter(
                        noise_pred_cond - noise_pred_uncond,
                        scale_low=fresca_scale_low,
                        scale_high=fresca_scale_high,
                        freq_cutoff=fresca_freq_cutoff,
                    )
                    noise_pred = noise_pred_uncond * alpha + cfg_scale * filtered_cond * alpha
                else:
                    noise_pred = noise_pred_uncond * alpha + cfg_scale * (noise_pred_cond - noise_pred_uncond * alpha)
                

                return noise_pred, [teacache_state_cond, teacache_state_uncond]

        log.info(f"Sampling {(latent_video_length-1) * 4 + 1} frames at {latents.shape[3]*8}x{latents.shape[2]*8} with {steps} steps")

        intermediate_device = device

        #clear memory before sampling
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        #region main loop start
        # print(f"denoising_multiplier: {denoising_multiplier}, denoising_multiplier_end:{denoising_multiplier_end}, denoising_skew: {denoising_skew}")
        for i, timestep_i in enumerate(tqdm(step_matrix)):
            # Adjust usedDenoising based on denoising_skew
            progress = i / (len(step_matrix) - 1)  # Normalized progress through the loop (0 to 1)
            if denoising_skew != 0:
                # Transition from denoising_multiplier to denoising_multiplier_end
                transition_factor = progress ** (1 / abs(denoising_skew))  # Faster transition for larger magnitude of denoising_skew
                usedDenoising = denoising_multiplier + (denoising_multiplier_end - denoising_multiplier) * transition_factor
            else:
                # No skew, linear transition
                transition_factor = 1.0
                usedDenoising = denoising_multiplier + (denoising_multiplier_end - denoising_multiplier) * progress

            try:
                try:
                    update_mask_i = step_update_mask[i]
                    valid_interval_i = valid_interval[i]
                    valid_interval_start, valid_interval_end = valid_interval_i
                    timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                    # Modify timestep to remove more noise
                    timestep = timestep * usedDenoising
                    # print(f"\nSampling frame {i} with timestep {timestep_i}, progress: {progress}, usedDenoising: {usedDenoising}, transition_factor: {transition_factor}, denoising_multiplier: {denoising_multiplier}, denoising_multiplier_end:{denoising_multiplier_end}, denoising_skew: {denoising_skew}")
                    latent_model_input = latents[:, valid_interval_start:valid_interval_end, :, :].clone()
                    if addnoise_condition > 0 and valid_interval_start < prefix_video_latent_length:
                        noise_factor = 0.001 * addnoise_condition
                        timestep_for_noised_condition = addnoise_condition
                        latent_model_input[:, valid_interval_start:prefix_video_latent_length] = (
                            latent_model_input[:, valid_interval_start:prefix_video_latent_length] * (1.0 - noise_factor)
                            + torch.randn_like(latent_model_input[:, valid_interval_start:prefix_video_latent_length])
                            * noise_factor
                        )
                        timestep[:, valid_interval_start:prefix_video_latent_length] = timestep_for_noised_condition
                except Exception as e:
                    log.error(f"Error during sampling[{i}] part 1: {e}")
                    log.error(f"Call stack:\n{traceback.format_exc()}")
                    raise

                try:
                    # print(f"timestep[{i}]", timestep)
                    noise_pred, self.teacache_state = predict_with_cfg(
                        latent_model_input.to(dtype), 
                        cfg[i], 
                        text_embeds["prompt_embeds"], 
                        text_embeds["negative_prompt_embeds"], 
                        timestep, i, image_cond, clip_fea, unianim_data=unianim_data, vace_data=vace_data,
                        teacache_state=self.teacache_state)
                except Exception as e:
                    log.error(f"Error during sampling[{i}] part 2: {e}")
                    log.error(f"Call stack:\n{traceback.format_exc()}")
                    raise

                try:
                    # print(f"timestep:{i}, denoising_multiplier: {denoising_multiplier}, valid_interval_start: {valid_interval_start}, valid_interval_end: {valid_interval_end}, noise_pred shape: {noise_pred.shape}, latents shape: {latents.shape}")
                    for idx in range(valid_interval_start, valid_interval_end):
                        if update_mask_i[idx].item():
                            # print(f"Sampling frame {idx} with timestep {timestep_i[idx]}. ")
                            # print(f"Sampling frame {idx} with timestep {timestep_i[idx]}. ")
                            latents[:, idx] = sample_schedulers[idx].step(
                                noise_pred[:, idx - valid_interval_start],
                                timestep_i[idx],
                                latents[:, idx],
                                return_dict=False,
                                generator=seed_g,
                            )[0]
                            sample_schedulers_counter[idx] += 1
                except Exception as e:
                    log.error(f"Error during sampling[{i}] part 3: {e}")
                    log.error(f"Call stack:\n{traceback.format_exc()}")
                    raise
                
                x0 = latents.unsqueeze(0)
                if callback is not None:
                    callback_latent = (latent_model_input - noise_pred.to(timestep_i[idx].device) * timestep_i[idx] / 1000).detach().permute(1,0,2,3)
                    callback(i, callback_latent, None, steps)
                else:
                    pbar.update(1)
            except Exception as e:
                log.error(f"Error during sampling[{i}]: {e}")

        if teacache_args is not None:
            states = transformer.teacache_state.states
            state_names = {
                0: "conditional",
                1: "unconditional"
            }
            for pred_id, state in states.items():
                name = state_names.get(pred_id, f"prediction_{pred_id}")
                if 'skipped_steps' in state:
                    log.info(f"TeaCache skipped: {len(state['skipped_steps'])} {name} steps: {state['skipped_steps']}")
            transformer.teacache_state.clear_all()

        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        try:
            print_memory(device)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        return ({
            "samples": x0.cpu(),
            }, )
    
class WanVideoLoopingDiffusionForcingSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_length": ("INT", {"default": 65, "min": 1, "step": 4, "max": 1000, "tooltip": "Number of frames to generate in each batch"}),
                "overlap_length": ("INT", {"default": 6, "min": 0, "max": 1000, "tooltip": "Number of frames to generate in each batch"}),
                "seed_adjust": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "tooltip": "Adjust the seed for each batch"}),
                "seed_batch_control": (["Seed Adjust", "Randomize"],
                    {"default": "Seed Adjust"}
                ),
                "samples_control": (["No Repeat", "Repeat"],
                    {"default": "No Repeat"}
                ),
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "addnoise_condition": ("INT", {"default": 10, "min": 0, "max": 1000, "tooltip": "Improves consistency in long video generation"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (["unipc", "unipc/beta", "euler", "euler/beta", "lcm", "lcm/beta"],
                    {
                        "default": 'unipc'
                    }),
            },
            "optional": {
                "vae": ("WANVAE",),
                "reencode_samples": (["Re-encode", "Ignore"],
                    {"default": "Ignore"}
                ),
                "prefix_samples_control": (["Ignore", "Merge"],
                    {"default": "Ignore"}
                ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "prefix_samples": ("LATENT", {"tooltip": "prefix latents"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache_args": ("CACHEARGS", {"tooltip": "cache for 1st loop"} ),
                "cache_args2": ("CACHEARGS", {"tooltip": "cache for after 1st loop loop"} ),
                "slg_args": ("SLGARGS", ),
                "rope_function": (["default", "comfy"], {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile"}),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "restore_face": ("RESTOREFACEARGS", ),
                "use_restore_face": ("BOOLEAN", {"default": True, "tooltip": "Use provided RestoreFace to restore faces in the generated video"}),
                "encode_latent_Args": ("WANENCODEARGS", ),
                "decode_latent_Args": ("WANDECODEARGS", ),
                "model_upscale_Args": ("WANMODELUPSCALEARGS", ),
                "color_match_args": ("COLOURMATCHARGS", ),
                "use_model_upscale": ("BOOLEAN", {"default": True, "tooltip": "Use provided upscale model to upscale the generated video"}),
                "simple_scale_Args": ("WANSIMPLESCALEARGS", {"tooltip": "Arguments for simple scaling."}),

                "numberOfFirstFrames": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "tooltip": "Number of first frames to use as brightness reference library"}),
                "contrast_stabilization": ("BOOLEAN", {"default": False, "tooltip": "Enable contrast stabilization to prevent shadow/highlight drift"}),
                "shadow_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.5, "step": 0.05, "tooltip": "Luminance threshold to define shadow areas"}),
                "highlight_threshold": ("FLOAT", {"default": 0.7, "min": 0.5, "max": 0.9, "step": 0.05, "tooltip": "Luminance threshold to define highlight areas"}),
                "shadow_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Strength of shadow correction - automatically brightens dark shadows and restores saturation. 0.0=no correction, 1.0=full correction, >1.0=over-correction"}),
                "highlight_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Strength of highlight correction - automatically tones down blown highlights and restores detail. 0.0=no correction, 1.0=full correction, >1.0=over-correction"}),
                "shadow_anti_banding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Anti-banding smoothing for shadows. Higher values = smoother shadows but potentially softer detail"}),
                "highlight_anti_banding": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Anti-banding smoothing for highlights. Higher values = smoother highlights but potentially softer detail"}),

                "noise_reduction_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "reduction_factor_change": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "denoising_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001, "tooltip": "Make the denoising process more or less aggressive"}),
                "denoising_multiplier_end": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001, "tooltip": "Make the denoising process more or less aggressive at the end of the video"}),
                "denoising_skew": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001, "tooltip": "How quickly do we transition from denoising_multiplier to denoising_multiplier_end. 0.0=linear."}),
                "prefix_denoise_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "How much the provided prefix_samples are processed prior to use."}),
                "prefix_denoising_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001, "tooltip": "Make the denoising process more or less aggressive"}),
                "prefix_denoising_multiplier_end": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001, "tooltip": "Make the denoising process more or less aggressive at the end of the video"}),
                "prefix_steps": ("INT", {"default": 6, "min": 1}),
                "prefix_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "prefix_frame_count": ("INT", {"default": 1, "min": 1}),
                "prefix_noise_reduction_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("samples", "prefix_samples", "generated_prefix_samples", "generated_samples","images","color_match_source","generated_first_frame",)
    OUTPUT_IS_LIST = (
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    FUNCTION = "process"
    CATEGORY = "VTS"

    # WanVideoLoopingDiffusionForcingSampler. total_samples: 16, total_prefix_samples: 16, total_frames: 26
    # Processing batch 0: frames 0 to 26 (total 26)
    # number of Bach image embeds: 3
    # Batch samples shape: torch.Size([1, 16, 25, 58, 104])
    # Prefix video of length: 2

    def process(self, batch_length, overlap_length, seed_adjust, seed_batch_control, samples_control, model, text_embeds, image_embeds, shift, fps, steps, addnoise_condition, cfg, seed, scheduler, 
                force_offload=True, vae=None, prefix_samples_control="Ignore", samples=None, prefix_samples=None, denoise_strength=1.0, slg_args=None, rope_function="default", cache_args=None, cache_args2=None,
                experimental_args=None, unianimate_poses=None, noise_reduction_factor=1.0, reduction_factor_change=0.0, denoising_multiplier=1.0, denoising_multiplier_end=None, denoising_skew=0.0, reencode_samples="Ignore", restore_face=None, use_restore_face=True,
                encode_latent_Args=None, decode_latent_Args=None, model_upscale_Args=None, use_model_upscale=True, simple_scale_Args=None, prefix_denoise_strength=0.0, prefix_denoising_multiplier=1.0, prefix_denoising_multiplier_end=None, prefix_steps=None,
                prefix_shift=None, prefix_frame_count=1, prefix_noise_reduction_factor=None, color_match_args=None,
                numberOfFirstFrames=20, contrast_stabilization=False, shadow_threshold=0.3, highlight_threshold=0.7,
                shadow_strength=0.8, highlight_strength=0.8, shadow_anti_banding=0.3, highlight_anti_banding=0.2):
        vae_stride = (4, 8, 8)
        
        # Initialize brightness lookup
        self.brightness_lookup = []
        self.numberOfFirstFrames = numberOfFirstFrames
        
        if cache_args2 is None:
            cache_args2 = cache_args
        if denoising_multiplier_end is None:
            denoising_multiplier_end = denoising_multiplier
        if prefix_denoising_multiplier_end is None:
            prefix_denoising_multiplier_end = prefix_denoising_multiplier
        if prefix_steps is None:
            prefix_steps = steps
        if prefix_shift is None:
            prefix_shift = shift
        if prefix_noise_reduction_factor is None:
            prefix_noise_reduction_factor = noise_reduction_factor
        prefix_samples_output = None
        generated_samples_output = None
        generated_images = None
        decoded_sample_images = None
        used_decoded_sample_images = None

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

        upscale_model = None
        upscale_model_device_preference = "auto"
        if model_upscale_Args:
            upscale_model = model_upscale_Args.get("upscale_model", upscale_model)
            upscale_model_device_preference = model_upscale_Args.get("device_preference", upscale_model_device_preference)

        color_match_source = None
        generated_first_frame = None
        color_match_used_source = 'disable-match'
        color_match_method = 'hm-mvgd-hm'
        color_match_strength = 0.0
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
                print(f"Color match method is {color_match_method}, strength is {color_match_strength}, used source is {color_match_used_source}")
            else:
                print(f"Color match is set to disable-match, not using color match")
        else:
            print(f"Color match args is not provided, not using color match")

        simple_scale_method = "bilinear"
        simple_crop_method = "disabled"
        if simple_scale_Args:
            simple_scale_method = simple_scale_Args.get("upscale_method", simple_scale_method)
            simple_crop_method = simple_scale_Args.get("crop", simple_crop_method)

        original_overlap_length = overlap_length
        # overlap_length needs to be exactly divisible by 4
        # if this is already the case then it does not need to change
        if (overlap_length) % 4 != 0:
            overlap_length_div_4 = math.ceil(overlap_length / 4)
            overlap_length = int(overlap_length_div_4 * 4)

        prefix_sample_num_latents = 0
        prefix_sample_num_frames = 0
        prefix_sample_shape = None
        overlap_number_of_latents = (overlap_length) // vae_stride[0]

        # Initialize the final samples list
        final_samples = None

        wanVideoDecode = None
        wanVideoEncode = None
        if vae:
            print(f"WanVideoLoopingDiffusionForcingSampler Using VAE")
            wanVideoDecode = WanVideoDecode()
            if reencode_samples == "Re-encode":
                print(f"WanVideoLoopingDiffusionForcingSampler Re-encoding samples")
                # we need to reencode the samples
                wanVideoEncode = WanVideoEncode()

        if overlap_length < 1:
            prefix_samples = None

        if (prefix_samples):
            generated_prefix_samples = prefix_samples
            prefix_sample_num_latents = prefix_samples["samples"].shape[2] # the actual number of sample latents, not image frames
            prefix_sample_num_frames = ((prefix_sample_num_latents - 1) * vae_stride[0]) + 1

            if prefix_denoise_strength > 0.0:
                print(f"Prefix samples provided, denoising with strength {prefix_denoise_strength}")
                if prefix_frame_count > prefix_sample_num_frames:
                    print(f"Prefix frame count {prefix_frame_count} is greater than prefix sample frames {prefix_sample_num_frames}, adjusting to {prefix_sample_num_frames}")
                    # we need to increase the prefix_samples number of latents and frames to match the prefix sample frames

                # we need to process the prefix samples with the denoising strength
                # Create a new instance of WanVideoDiffusionForcingSampler
                sampler = WanVideoDiffusionForcingSampler()
                target_shape = (16, 
                            prefix_sample_num_latents,
                            image_embeds["target_shape"][2],
                            image_embeds["target_shape"][3],)
                batch_image_embeds = {
                    "target_shape": target_shape,
                    "num_frames": prefix_sample_num_frames,
                }
                # Call the process method of WanVideoDiffusionForcingSampler
                result = sampler.process(
                    model=model,
                    text_embeds=text_embeds,
                    image_embeds=batch_image_embeds,
                    shift=prefix_shift,
                    fps=fps,
                    steps=prefix_steps,
                    addnoise_condition=addnoise_condition,
                    cfg=cfg,
                    seed=seed,
                    scheduler=scheduler,
                    force_offload=False,
                    samples=prefix_samples,
                    prefix_samples=None,
                    denoise_strength=prefix_denoise_strength,
                    slg_args=slg_args,
                    rope_function=rope_function,
                    cache_args=cache_args,
                    experimental_args=experimental_args,
                    unianimate_poses=unianimate_poses,
                    noise_reduction_factor=prefix_noise_reduction_factor,
                    denoising_multiplier=prefix_denoising_multiplier,
                    denoising_multiplier_end=prefix_denoising_multiplier_end,
                )
                prefix_samples = result[0]
                generated_prefix_samples = prefix_samples
                prefix_sample_num_latents = prefix_samples["samples"].shape[2] # the actual number of sample latents, not image frames
                prefix_sample_num_frames = ((prefix_sample_num_latents - 1) * vae_stride[0]) + 1

            if prefix_samples_control == "Merge":
                prefix_sample_num_latents = prefix_samples["samples"].shape[2] # the actual number of sample latents, not image frames
                prefix_sample_num_frames = ((prefix_sample_num_latents - 1) * vae_stride[0]) + 1
                print(f"Prefix samples provided, merging with samples")
                non_overlapping_samples = prefix_samples["samples"][:, :, :-1]
                final_samples = {
                    "samples": non_overlapping_samples
                }

            # the last sample of any batch only contains a single frame, rather than the usual 4 frames
            # so we need to remove the last sample from the provided prefix_samples by taking the first prefix_sample_num_latents - 1 samples
            number_of_prefixSamples_to_use = prefix_sample_num_latents - 1
            if number_of_prefixSamples_to_use > 0:
                prefix_samples = {
                    "samples": prefix_samples["samples"][:, :, :-1]
                }
                prefix_sample_num_latents = prefix_samples["samples"].shape[2]
                prefix_sample_num_frames = prefix_sample_num_latents * vae_stride[0]
            
            
            if prefix_sample_num_frames > overlap_length:
                # we need to reduce the prefix_samples
                prefix_samples = {"samples": prefix_samples["samples"][:,  :,  -overlap_number_of_latents:]}
                prefix_sample_num_latents = prefix_samples["samples"].shape[2] # the actual number of sample latents, not image frames
                prefix_sample_num_frames = ((prefix_sample_num_latents - 1) * vae_stride[0]) + 1

            if prefix_sample_num_frames > prefix_frame_count:
                # we need to reduce the prefix_samples
                prefix_samples = {"samples": prefix_samples["samples"][:, :, :prefix_frame_count]}
                prefix_sample_num_latents = prefix_samples["samples"].shape[2]
                prefix_sample_num_frames = ((prefix_sample_num_latents - 1) * vae_stride[0]) + 1

            if wanVideoDecode is not None and wanVideoEncode is not None and color_match_strength > 0.0:
                # we need to decode the prefix samples
                print(f"Decoding prefix samples with wanVideoDecode")
                decoded_prefix_samples = wanVideoDecode.decode(vae, prefix_samples, decodeTile, decodeTileX, decodeTileY, decodeTileStrideX, decodeTileStrideY)
                used_decoded_sample_images = decoded_prefix_samples[0]
                print(f"Decoded prefix samples with wanVideoDecode. shape: {used_decoded_sample_images.shape}")
                generated_first_frame = used_decoded_sample_images[0].unsqueeze(0).clone()
                if color_match_source is None:
                    # use the first frame of the decoded prefix samples as the color match source
                    color_match_source = used_decoded_sample_images[0].unsqueeze(0).clone()
                
                # Apply color matching to prefix samples based on selected method
                used_decoded_sample_images = self.process_first_frames_sequence(
                    used_decoded_sample_images,
                    color_match_method,
                    color_match_strength,
                    False,
                    20,
                    contrast_stabilization,
                    shadow_threshold,
                    highlight_threshold,
                    shadow_strength,
                    highlight_strength,
                    shadow_anti_banding,
                    highlight_anti_banding
                )
                
                # we need to reencode the decoded prefix samples
                print(f"Re-encoding decoded prefix samples with wanVideoEncode")
                encoded_samples = wanVideoEncode.encode(
                    vae, 
                    used_decoded_sample_images, 
                    encodeTile, 
                    encodeTileX, 
                    encodeTileY, 
                    encodeTileStrideX, 
                    encodeTileStrideY,
                    encodeAugStrength,
                    encodeLatentStrength,
                    encodeMask,
                )
                prefix_samples = {"samples": encoded_samples[0]["samples"]}
            else:
                print(f"Not decoding prefix samples with wanVideoDecode for colormatch, using provided prefix_samples")
            
            prefix_sample_shape = prefix_samples["samples"].shape

        sample_shape = None
        number_of_sample_latents = 0
        if (samples):
            sample_shape = samples["samples"].shape
            number_of_sample_latents = sample_shape[2]

        # Get the total number of frames to generate
        latent_frames = image_embeds["target_shape"][1]
        total_frames = image_embeds["num_frames"]

        initial_batch_length = batch_length
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

        print(f"latent_frames: {latent_frames}, total_frames: {total_frames}, number_of_batches: {number_of_batches}, initial_batch_length: {initial_batch_length}, batch_length: {batch_length}, overlap_length: {overlap_length}, prefix_sample_num_latents: {prefix_sample_num_latents}, prefix_sample_num_frames: {prefix_sample_num_frames}, sample_shape: {sample_shape}, prefix_sample_shape: {prefix_sample_shape}")

        loop_count = 0
        remaining_frames = total_frames
        # Loop through batches
        for start_idx in range(0, total_frames, batch_length - 1):
            # Stop the loop early if remaining_frames <= 0
            if remaining_frames <= 0:
                break

            # Determine the end index for the current batch
            end_idx = min(start_idx + batch_length - 1, total_frames) # note the end_idx is INCLUSIVE
            number_of_frames_for_batch = end_idx - start_idx + 1 # +1 because end_idx is inclusive
            number_of_latents_for_batch = (number_of_frames_for_batch - 1) // vae_stride[0] + 1

            remaining_frames -= number_of_frames_for_batch
            if loop_count < number_of_batches - 1:
                remaining_frames += 1 # we need to add 1 back to the remaining frames because we are goign to process it again in the next loop

            start_latent_index = int(start_idx / 4) # inclusive
            end_latent_index = math.ceil(end_idx / 4) # inclusive

            # adjust the number_of_frames_for_batch to take into account the overlap
            number_of_frames_for_batch_embeds = number_of_frames_for_batch + prefix_sample_num_frames
            number_of_latents_for_batch_embeds = (number_of_frames_for_batch_embeds - 1) // vae_stride[0] + 1

            output_image_height = image_embeds["target_shape"][2] * 8
            output_image_width = image_embeds["target_shape"][3] * 8

            print(f"Processing batch [{loop_count + 1}/{number_of_batches}] = output_image_width {output_image_width}, output_image_height: {output_image_height}")
            print(f"Processing batch [{loop_count + 1}/{number_of_batches}] = frames {start_idx} to {end_idx} inclusive. remaining_frames: {remaining_frames}, start_latent_index: {start_latent_index}, end_latent_index: {end_latent_index}, prefix_sample_num_frames: {prefix_sample_num_frames}")
            print(f"Processing batch [{loop_count + 1}/{number_of_batches}] = number_of_frames_for_batch {number_of_frames_for_batch}, number_of_latents_for_batch: {number_of_latents_for_batch}, number_of_frames_for_batch_embeds: {number_of_frames_for_batch_embeds}, number_of_latents_for_batch_embeds: {number_of_latents_for_batch_embeds}, prefix_sample_num_latents: {prefix_sample_num_latents}, overlap_length: {overlap_length}")

            # Generate batch_image_embeds by slicing the target_shape and control_embeds
            target_shape = (16, 
                            number_of_latents_for_batch_embeds,
                            image_embeds["target_shape"][2],
                            image_embeds["target_shape"][3],)

            batch_image_embeds = {
                "target_shape": target_shape,
                "num_frames": number_of_frames_for_batch_embeds,
            }

            # unfortunately, right nnow these are not used...
            if image_embeds.get("control_embeds") is not None:
                # print(f'image_embeds["control_embeds"].shape: {image_embeds["control_embeds"].shape}, start_latent_index: {start_latent_index}, end_latent_index: {end_latent_index}')
                
                original_control_images = image_embeds["control_embeds"].get("control_images", None)
                control_images = image_embeds["control_embeds"].get("control_images", None)
                original_control_camera_latents = image_embeds["control_embeds"].get("control_camera_latents", None)
                control_camera_latents = image_embeds["control_embeds"].get("control_camera_latents", None)
                
                # Slice control_images if they exist
                if control_images is not None:
                    control_images = control_images[:, start_latent_index:end_latent_index]
                    # log out the provided shape, and the calculated shape
                    #print(f"control_images.shape: {control_images.shape}, original_control_images.shape: {original_control_images.shape}, start_latent_index: {start_latent_index}, end_latent_index: {end_latent_index}")
                
                # Slice control_camera_latents if they exist
                if control_camera_latents is not None:
                    control_camera_latents = control_camera_latents[:, start_latent_index:end_latent_index]
                    # log out the provided shape, and the calculated shape
                    #print(f"control_camera_latents.shape: {control_camera_latents.shape}, original_control_camera_latents.shape: {original_control_camera_latents.shape}, start_latent_index: {start_latent_index}, end_latent_index: {end_latent_index}")
                
                # Add sliced control_embeds to batch_image_embeds
                batch_image_embeds["control_embeds"] = {
                    "control_images": control_images,
                    "control_camera_latents": control_camera_latents,
                    "control_camera_start_percent": image_embeds["control_embeds"].get("control_camera_start_percent", 0.0),
                    "control_camera_end_percent": image_embeds["control_embeds"].get("control_camera_end_percent", 1.0),
                    "start_percent": image_embeds["control_embeds"].get("start_percent", 0.0),
                    "end_percent": image_embeds["control_embeds"].get("end_percent", 1.0),
                }

            batch_latent_frames = batch_image_embeds["target_shape"][1]
            batch_num_frames = batch_image_embeds["num_frames"]
            print(f"Processing batch [{loop_count + 1}/{number_of_batches}] = batch_latent_frames: {batch_latent_frames}, batch_num_frames: {batch_num_frames}, target_shape: {target_shape}")

            # Slice the samples for the current batch if available
            batch_samples = None
            if samples is not None:
                start_sample_latent_index = start_latent_index
                end_sample_latent_index = start_sample_latent_index + number_of_latents_for_batch
                if samples_control == "Repeat":
                    start_sample_latent_index = start_latent_index % number_of_sample_latents
                    end_sample_latent_index = start_sample_latent_index + number_of_latents_for_batch
                    # Adjust indices to include overlap
                    # start_sample_latent_index = max(0, start_latent_index - overlap_number_of_latents)
                    # end_sample_latent_index = min(sample_shape[2], end_latent_index + overlap_number_of_latents)

                # Slice the samples
                sliced_samples = samples["samples"][:, :, start_sample_latent_index:end_sample_latent_index]

                # Check if the sliced samples are fewer than needed
                if samples_control == "Repeat" and sliced_samples.shape[2] < batch_latent_frames:
                    # Calculate how many additional latents are needed
                    additional_latents_needed = batch_latent_frames - sliced_samples.shape[2]

                    # Loop back to the beginning of the samples to fill the gap
                    while additional_latents_needed > 0:
                        looped_samples = samples["samples"][:, :, :min(additional_latents_needed, sample_shape[2])]
                        looped_samples = looped_samples.to(sliced_samples.device)  # Ensure device consistency
                        sliced_samples = torch.cat((sliced_samples, looped_samples), dim=2)
                        additional_latents_needed -= looped_samples.shape[2]

                    # Trim to ensure the final size matches batch_latent_frames
                    sliced_samples = sliced_samples[:, :, :batch_latent_frames]

                # Assign the repeated or sliced samples to batch_samples
                batch_samples = {"samples": sliced_samples}

                # If the sliced samples are still empty, set to None
                if batch_samples["samples"].shape[2] == 0:
                    batch_samples = None
                    batch_samples_shape = None
                else:
                    batch_samples_shape = batch_samples["samples"].shape
                print(f"Processing batch [{loop_count + 1}/{number_of_batches}] = start_sample_latent_index: {start_sample_latent_index}, end_sample_latent_index: {end_sample_latent_index}, batch_samples_shape: {batch_samples_shape}, number_of_sample_latents: {number_of_sample_latents}")

            # only use force_offload if we are on the last loop iteration, otherwise set it to false
            batch_force_offload = force_offload if loop_count == number_of_batches - 1 else False

            batch_cache_args = cache_args
            if loop_count > 0:
                batch_cache_args = cache_args2

            # Create a new instance of WanVideoDiffusionForcingSampler
            sampler = WanVideoDiffusionForcingSampler()
            # Call the process method of WanVideoDiffusionForcingSampler
            batch_result = sampler.process(
                model=model,
                text_embeds=text_embeds,
                image_embeds=batch_image_embeds,
                shift=shift,
                fps=fps,
                steps=steps,
                addnoise_condition=addnoise_condition,
                cfg=cfg,
                seed=seed,
                scheduler=scheduler,
                force_offload=batch_force_offload,
                samples=batch_samples,
                prefix_samples=prefix_samples,
                denoise_strength=denoise_strength,
                slg_args=slg_args,
                rope_function=rope_function,
                cache_args=batch_cache_args,
                experimental_args=experimental_args,
                unianimate_poses=unianimate_poses,
                noise_reduction_factor=noise_reduction_factor,
                denoising_multiplier=denoising_multiplier,
                denoising_multiplier_end=denoising_multiplier_end,
                denoising_skew=denoising_skew,
            )

            noise_reduction_factor = noise_reduction_factor + reduction_factor_change

            if seed_batch_control == "Randomize":
                # Randomize the seed for each batch
                seed = random.randint(0, 0xffffffffffffffff)
            else:
                seed = (seed + seed_adjust) % 0xffffffffffffffff

            batch_result_samples = batch_result[0]
            # Decode the samples if wanVideoDecode is available
            print(f"WanVideoLoopingDiffusionForcingSampler deciding if it should decode")
            if (wanVideoDecode is not None and vae is not None):
                print(f"WanVideoLoopingDiffusionForcingSampler decoding. require number_of_frames_for_batch={number_of_frames_for_batch} frames this loop")

                decoded_samples = wanVideoDecode.decode(vae, batch_result_samples, decodeTile, decodeTileX, decodeTileY, decodeTileStrideX, decodeTileStrideY)
                decoded_sample_images = decoded_samples[0]
                used_decoded_sample_images = decoded_samples[0]
                
                if restore_face and use_restore_face and reactor:
                    start_time = time.time()  # Start timing
                    faceDetectionModel = restore_face["facedetection"]
                    faceRestoreModel = restore_face["model"]
                    faceRestoreVisibility = restore_face["visibility"]
                    faceRestoreCodeformerWeight = restore_face["codeformer_weight"]
                    print(f"WanVideoLoopingDiffusionForcingSampler restoring faces in used_decoded_sample_images with faceRestoreModel: {faceRestoreModel}, faceRestoreVisibility: {faceRestoreVisibility}, faceRestoreCodeformerWeight: {faceRestoreCodeformerWeight}, faceDetectionModel: {faceDetectionModel}")
                    used_decoded_sample_images = reactor.restore_face(self, used_decoded_sample_images, faceRestoreModel, faceRestoreVisibility, faceRestoreCodeformerWeight, faceDetectionModel)
                    reactor.unload_face_restore_model(self)
                    end_time = time.time()  # End timing
                    elapsed_time = end_time - start_time
                    print(f"Execution time for face restoration block: {elapsed_time:.4f} seconds")

                if use_model_upscale and upscale_model is not None:
                    start_time = time.time()  # Start timing
                    print(f"WanVideoLoopingDiffusionForcingSampler upscaling used_decoded_sample_images with upscale_model: {upscale_model}, upscale_model_device_preference: {upscale_model_device_preference}")
                    used_decoded_sample_images = self.upscaleWithModel(upscale_model, used_decoded_sample_images, upscale_model_device_preference)
                    end_time = time.time()  # End timing
                    elapsed_time = end_time - start_time
                    print(f"Execution time for model upscale block: {elapsed_time:.4f} seconds")

                if color_match_strength > 0.0:       
                    start_time = time.time()

                    used_decoded_sample_images = self.process_first_frames_sequence(
                        used_decoded_sample_images,
                        color_match_method,
                        color_match_strength,
                        False,
                        20,
                        contrast_stabilization,
                        shadow_threshold,
                        highlight_threshold,
                        shadow_strength,
                        highlight_strength,
                        shadow_anti_banding,
                        highlight_anti_banding
                    )
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Execution time for frame-by-frame color matching: {elapsed_time:.4f} seconds")

                if wanVideoEncode is not None:
                    start_time = time.time()  # Start timing
                    print(f"WanVideoLoopingDiffusionForcingSampler re-encoding used_decoded_sample_images with wanVideoEncode")
                    # we need to re-encode the decoded samples
                    encoded_samples = wanVideoEncode.encode(
                        vae, 
                        used_decoded_sample_images, 
                        encodeTile, 
                        encodeTileX, 
                        encodeTileY, 
                        encodeTileStrideX, 
                        encodeTileStrideY,
                        encodeAugStrength,
                        encodeLatentStrength,
                        encodeMask,
                    )
                    end_time = time.time()  # End timing
                    elapsed_time = end_time - start_time
                    print(f"Execution time for re-encoding block: {elapsed_time:.4f} seconds")
                    batch_result_samples = encoded_samples[0]

                if remaining_frames > 0:
                    used_decoded_sample_images = used_decoded_sample_images[-(number_of_frames_for_batch):-1]  # Drop the last image as it will be decoded and added in the next loop
                else:
                    used_decoded_sample_images = used_decoded_sample_images[-number_of_frames_for_batch:]

                if generated_images is None:
                    print(f"WanVideoLoopingDiffusionForcingSampler creating generated_images")
                    generated_images = used_decoded_sample_images
                else:
                    print(f"WanVideoLoopingDiffusionForcingSampler extending generated_images")
                    generated_images = torch.cat((generated_images, used_decoded_sample_images), dim=0)
                print(f"Processing batch [{loop_count + 1}/{number_of_batches}] = generated_images shape: {generated_images.shape}, used_decoded_sample_images shape: {used_decoded_sample_images.shape}, decoded_sample_images shape: {decoded_sample_images.shape}")

            number_of_latents_for_this_loop = number_of_latents_for_batch
            
            # if there are remaining frames, we need to drop the last latent, which contains only one frame of data
            if remaining_frames > 0:
                # number_of_latents_for_this_loop = number_of_latents_for_this_loop - 1
                samples_to_add_for_this_loop = {
                        "samples": batch_result_samples["samples"][:, :, -number_of_latents_for_this_loop:-1]
                }
            else:
                samples_to_add_for_this_loop = {
                        "samples": batch_result_samples["samples"][:, :, -number_of_latents_for_this_loop:]
                }
            print(f"!samples_to_add_for_this_loop shape: {samples_to_add_for_this_loop['samples'].shape}")

            if generated_samples_output is None:
                # Initialize final_samples with the first batch
                generated_samples_output = {
                    "samples": samples_to_add_for_this_loop["samples"]
                }
            else:
                # Exclude the overlap region from the previous batch
                try:
                    print(f"!generated_samples_output['samples'] shape before merge: {generated_samples_output['samples'].shape}")
                    merged_samples = torch.cat((generated_samples_output["samples"], samples_to_add_for_this_loop["samples"]), dim=2)
                    generated_samples_output = {
                        "samples": merged_samples
                    }
                    print(f"!generated_samples_output['samples'] shape after merge: {generated_samples_output['samples'].shape}")
                except Exception as e:
                    print(f"!Error concatenating final_samples: {e}")

            # Merge the current batch samples into the final samples
            if final_samples is None:
                # Initialize final_samples with the first batch
                final_samples = {
                    "samples": samples_to_add_for_this_loop["samples"]
                }
            else:
                # Exclude the overlap region from the previous batch
                try:
                    print(f"!final_samples['samples'] shape before merge: {final_samples['samples'].shape}")
                    merged_samples = torch.cat((final_samples["samples"], samples_to_add_for_this_loop["samples"]), dim=2)
                    final_samples = {
                        "samples": merged_samples
                    }
                    print(f"!final_samples['samples'] shape after merge: {final_samples['samples'].shape}")
                except Exception as e:
                    print(f"!Error concatenating final_samples: {e}")
                    # print(f"!final_samples['samples'] shape: {final_samples['samples'].shape}, non_overlapping_samples shape: {non_overlapping_samples.shape}")
            print(f"Processing batch [{loop_count + 1}/{number_of_batches}] = final_samples['samples'] shape: {final_samples['samples'].shape}")
                    
            prefix_sample_num_frames = overlap_length
            prefix_sample_num_latents = (prefix_sample_num_frames) // vae_stride[0]
            if (prefix_sample_num_latents > 0 and overlap_length > 0):
                # Set the prefix_samples for the next iteration to the last overlap_length samples
                prefix_samples = {"samples": final_samples["samples"][:,  :,  -prefix_sample_num_latents:]}
            else:
                prefix_samples = None
                prefix_sample_num_latents = 0
            
            if prefix_samples is not None and prefix_samples["samples"] is not None:
                if wanVideoDecode is not None and wanVideoEncode is not None and vae is not None:
                    decoded_samples = None
                    if generated_images is not None:
                        # no need to decode again, we already have the images. so just get the needed images
                        # tmp_num_frames = int((prefix_sample_num_latents * vae_stride[0]) - 3)
                        tmp_num_frames = prefix_sample_num_frames
                        print(f"re-using {tmp_num_frames} images already decoded samples from: {generated_images.shape}")
                        decoded_samples = generated_images[-tmp_num_frames:]
                    else:
                        # decoding the prefix samples and the re-encoding them can improve quality
                        print(f"Decoding prefix samples shape: {prefix_samples['samples'].shape}")
                        # Decode the prefix samples
                        decoded_samples = wanVideoDecode.decode(vae, prefix_samples, decodeTile, decodeTileX, decodeTileY, decodeTileStrideX, decodeTileStrideY)[0]
                    
                    # ensure the decoded_samples are the correct size
                    print(f"Decoded prefix samples shape: {decoded_samples.shape}, prefix_sample_num_latents: {prefix_sample_num_latents}, prefix_sample_num_frames: {prefix_sample_num_frames}, overlap_length: {overlap_length}")
                    decoded_samples = self.scale(decoded_samples, output_image_width, output_image_height, simple_scale_method, simple_crop_method)

                    # Encode the decoded samples
                    print(f"Re-encoding decoded prefix samples shape: {decoded_samples.shape}")
                    encoded_samples = wanVideoEncode.encode(vae, decoded_samples, encodeTile, encodeTileX, encodeTileY, encodeTileStrideX, encodeTileStrideY, encodeAugStrength, encodeLatentStrength, encodeMask)
                    print(f"Assigning re-encoded prefix samples shape: {encoded_samples[0]['samples'].shape}")
                    prefix_samples = {"samples": encoded_samples[0]["samples"]}

                if prefix_samples_output is None:
                    # Initialize final_samples with the first batch
                    prefix_samples_output = {
                        "samples": prefix_samples["samples"]
                    }
                else:
                    # Exclude the overlap region from the previous batch
                    try:
                        print(f"!prefix_samples_output['samples'] shape before merge: {prefix_samples_output['samples'].shape}")
                        merged_samples = torch.cat((prefix_samples_output["samples"], prefix_samples["samples"]), dim=2)
                        prefix_samples_output = {
                            "samples": merged_samples
                        }
                        print(f"!final_samples['samples'] shape after merge: {final_samples['samples'].shape}")
                    except Exception as e:
                        print(f"!Error concatenating final_samples: {e}")

            if prefix_samples is not None and prefix_samples["samples"] is not None:
                print(f"Processing batch [{loop_count + 1}/{number_of_batches}] = prefix_samples shape: {prefix_samples['samples'].shape}, prefix_sample_num_latents: {prefix_sample_num_latents}, prefix_sample_num_frames: {prefix_sample_num_frames}, overlap_length: {overlap_length}")

            print(f"Processed batch [{loop_count + 1}/{number_of_batches}]. calculated next loop = prefix_sample_num_frames: {prefix_sample_num_frames}, prefix_sample_num_latents: {prefix_sample_num_latents}")

            # Increment the loop count
            loop_count += 1

        # return ({"samples": final_samples}),
        return ({
            "samples": final_samples['samples'],
            }, {
            "samples": prefix_samples_output['samples'] if prefix_samples_output else None,
            },{
            "samples": generated_prefix_samples['samples'] if generated_prefix_samples else None,
            }, {
            "samples": generated_samples_output['samples'] if generated_samples_output else None,
            }, 
            generated_images,
            color_match_source,
            generated_first_frame,)


    def calculate_brightness_signature(self, frame):
        """Calculate comprehensive brightness signature for frame matching"""
        lum = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        
        # Histogram-based signature (most important)
        hist = torch.histc(lum.flatten(), bins=16, min=0, max=1)
        hist_norm = hist / (hist.sum() + 1e-8)
        
        # Statistical measures
        mean_lum = lum.mean().item()
        std_lum = lum.std().item()
        
        # Percentiles for distribution shape
        percentiles = torch.quantile(lum.flatten(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        
        return {
            'histogram': hist_norm,
            'mean': mean_lum,
            'std': std_lum,
            'percentiles': percentiles
        }

    def find_best_brightness_match(self, target_signature, brightness_lookup):
        """Find the best matching frame from brightness lookup"""
        best_distance = float('inf')
        best_frame_data = None
        
        for frame_data in brightness_lookup:
            ref_sig = frame_data['signature']
            
            # Primary: Histogram distance
            hist_distance = torch.sum(torch.abs(target_signature['histogram'] - ref_sig['histogram'])).item()
            
            # Secondary: Statistical distances
            mean_distance = abs(target_signature['mean'] - ref_sig['mean'])
            std_distance = abs(target_signature['std'] - ref_sig['std'])
            perc_distance = torch.mean(torch.abs(target_signature['percentiles'] - ref_sig['percentiles'])).item()
            
            # Weighted combination
            total_distance = (0.6 * hist_distance + 
                             0.2 * mean_distance + 
                             0.1 * std_distance + 
                             0.1 * perc_distance)
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_frame_data = frame_data
        
        return best_frame_data

    def apply_contrast_stabilization(self, current_frame, shadow_strength=0.8, highlight_strength=0.8, 
                                   shadow_threshold=0.3, highlight_threshold=0.7, 
                                   shadow_anti_banding=0.3, highlight_anti_banding=0.2):
        """Apply contrast stabilization to prevent shadow/highlight drift without using reference frames"""
        curr_lum = 0.299 * current_frame[..., 0] + 0.587 * current_frame[..., 1] + 0.114 * current_frame[..., 2]
        
        result = current_frame.clone()
        
        # Create smooth transition masks instead of hard thresholds
        # Shadow mask: strong at 0, fades to 0 at shadow_threshold, extends slightly beyond
        shadow_fade_end = shadow_threshold + 0.1
        shadow_mask = torch.clamp((shadow_fade_end - curr_lum) / (shadow_fade_end - 0.0), 0.0, 1.0)
        shadow_mask = shadow_mask * (curr_lum < shadow_fade_end).float()
        
        # Highlight mask: starts at highlight_threshold, strong at 1.0, with smooth transition
        highlight_fade_start = highlight_threshold - 0.1
        highlight_mask = torch.clamp((curr_lum - highlight_fade_start) / (1.0 - highlight_fade_start), 0.0, 1.0)
        highlight_mask = highlight_mask * (curr_lum > highlight_fade_start).float()
        
        # Shadow stabilization: prevent shadows from getting darker and less saturated
        if shadow_mask.sum() > 0 and shadow_strength > 0:
            # Calculate per-pixel lift factor based on how dark the pixel is
            lift_factor = torch.ones_like(curr_lum)
            shadow_areas = shadow_mask > 0.1
            
            if shadow_areas.any():
                # Stronger lift for darker pixels, gentler for lighter shadows
                darkness_factor = (shadow_threshold - curr_lum[shadow_areas]) / shadow_threshold
                lift_amount = 1.0 + (darkness_factor * 0.2 * shadow_strength)  # Max 20% lift
                lift_factor[shadow_areas] = lift_amount
            
            # Apply luminance lift and saturation restoration
            for c in range(3):
                # Lift shadows
                shadow_correction = (current_frame[..., c] * lift_factor - current_frame[..., c]) * shadow_mask
                
                # Restore saturation by pushing towards channel mean in shadow areas
                if shadow_areas.any():
                    channel_mean = current_frame[..., c][shadow_mask > 0.1].mean()
                    saturation_restore = (channel_mean - current_frame[..., c]) * 0.1 * shadow_strength * shadow_mask
                    result[..., c] = result[..., c] + shadow_correction + saturation_restore
                else:
                    result[..., c] = result[..., c] + shadow_correction
        
        # Highlight stabilization: prevent highlights from getting lighter and blown out
        if highlight_mask.sum() > 0 and highlight_strength > 0:
            # Calculate per-pixel compression factor based on how bright the pixel is
            compress_factor = torch.ones_like(curr_lum)
            highlight_areas = highlight_mask > 0.1
            
            if highlight_areas.any():
                # Stronger compression for brighter pixels
                brightness_factor = (curr_lum[highlight_areas] - highlight_threshold) / (1.0 - highlight_threshold)
                compress_amount = 1.0 - (brightness_factor * 0.15 * highlight_strength)  # Max 15% compression
                compress_factor[highlight_areas] = torch.clamp(compress_amount, 0.7, 1.0)
            
            # Apply luminance compression and detail restoration
            for c in range(3):
                # Compress highlights
                highlight_correction = (current_frame[..., c] * compress_factor - current_frame[..., c]) * highlight_mask
                
                # Restore detail by preserving local contrast
                if highlight_areas.any():
                    local_mean = torch.nn.functional.avg_pool2d(
                        current_frame[..., c].unsqueeze(0).unsqueeze(0), 
                        kernel_size=5, stride=1, padding=2
                    ).squeeze()
                    detail_preserve = (current_frame[..., c] - local_mean) * 0.8 * highlight_mask
                    result[..., c] = result[..., c] + highlight_correction + detail_preserve
                else:
                    result[..., c] = result[..., c] + highlight_correction
        
        # Apply anti-banding smoothing
        if shadow_anti_banding > 0 and shadow_mask.sum() > 0:
            result = self.apply_zone_smoothing(result, current_frame, shadow_mask, shadow_anti_banding, "shadow")
        
        if highlight_anti_banding > 0 and highlight_mask.sum() > 0:
            result = self.apply_zone_smoothing(result, current_frame, highlight_mask, highlight_anti_banding, "highlight")
        
        result = torch.clamp(result, 0.0, 1.0)
        return result

    def apply_zone_smoothing(self, result, original, mask, smoothing_strength, zone_type):
        """Fast edge-preserving smoothing with optimized operations"""
        if smoothing_strength <= 0:
            return result
        
        # Reduced kernel sizes for speed
        kernel_size = 3  # Same size for both shadow and highlight for simplicity
        padding = kernel_size // 2
        
        # Simple edge detection (faster than full gradient calculation)
        orig_gray = 0.299 * original[..., 0] + 0.587 * original[..., 1] + 0.114 * original[..., 2]
        
        # Faster edge detection using built-in conv2d
        edge_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                   dtype=torch.float32, device=original.device).unsqueeze(0).unsqueeze(0)
        
        orig_padded = orig_gray.unsqueeze(0).unsqueeze(0)
        orig_padded = torch.nn.functional.pad(orig_padded, (1, 1, 1, 1), mode='reflect')
        edge_response = torch.nn.functional.conv2d(orig_padded, edge_kernel).squeeze()
        edge_strength = torch.abs(edge_response)
        
        # Simplified edge weighting
        edge_threshold = 0.1
        smooth_weights = torch.exp(-edge_strength / edge_threshold)
        zone_smooth_weights = mask * smooth_weights * smoothing_strength
        
        # Fast box filter using separable convolution
        smoothed = result.clone()
        box_kernel = torch.ones(1, 1, kernel_size, 1, device=result.device) / kernel_size
        
        for c in range(3):
            channel = result[..., c].unsqueeze(0).unsqueeze(0)
            # Horizontal pass
            channel_h = torch.nn.functional.pad(channel, (0, 0, padding, padding), mode='reflect')
            channel_h = torch.nn.functional.conv2d(channel_h, box_kernel)
            # Vertical pass  
            channel_v = torch.nn.functional.pad(channel_h, (padding, padding, 0, 0), mode='reflect')
            channel_smooth = torch.nn.functional.conv2d(channel_v, box_kernel.transpose(-1, -2))
            
            # Blend with original
            blend_factor = zone_smooth_weights
            smoothed[..., c] = (1 - blend_factor) * result[..., c] + blend_factor * channel_smooth.squeeze()
        
        return smoothed

    def process_first_frames_sequence(self, decoded_frames, color_match_method, color_match_strength, editInPlace, gc_interval,
                                     contrast_stabilization=False, shadow_threshold=0.3, highlight_threshold=0.7, 
                                     shadow_strength=0.8, highlight_strength=0.8, shadow_anti_banding=0.3, highlight_anti_banding=0.2):
        """
        Enhanced processing with optional contrast stabilization
        """
        if color_match_strength <= 0.0 and not contrast_stabilization:
            return decoded_frames
        
        processed_frames = []
        
        for i, current_frame in enumerate(decoded_frames):
            current_frame_batch = current_frame.unsqueeze(0)  # Add batch dimension
            
            if self.brightness_lookup is None or len(self.brightness_lookup) < self.numberOfFirstFrames:
                # Build brightness lookup from first frames (no color matching applied)
                signature = self.calculate_brightness_signature(current_frame)
                self.brightness_lookup.append({
                    'frame': current_frame.clone(),
                    'signature': signature,
                    'frame_index': i
                })
                print(f"Frame {i}: Added to brightness lookup (mean luminance: {signature['mean']:.3f}, std: {signature['std']:.3f})")
                
                # Apply contrast stabilization to reference frames if enabled
                if contrast_stabilization:
                    processed_frame = self.apply_contrast_stabilization(
                        current_frame, shadow_strength, highlight_strength,
                        shadow_threshold, highlight_threshold,
                        shadow_anti_banding, highlight_anti_banding
                    )
                else:
                    processed_frame = current_frame
                    
                processed_frames.append(processed_frame)
            else:
                # Use brightness lookup to find best matching reference frame
                target_signature = self.calculate_brightness_signature(current_frame)
                best_match = self.find_best_brightness_match(target_signature, self.brightness_lookup)
                
                if best_match is not None:
                    reference_frame = best_match['frame'].unsqueeze(0)
                    print(f"Frame {i}: Matched with reference frame {best_match['frame_index']} (target mean lum: {target_signature['mean']:.3f})")
                    
                    # Apply color matching using the matched reference frame
                    if color_match_strength > 0.0:
                        color_match_result = colormatch(reference_frame, current_frame_batch, color_match_method, color_match_strength, editInPlace, gc_interval)
                        processed_frame = color_match_result[0][0]  # Remove batch dimension
                    else:
                        processed_frame = current_frame
                else:
                    print(f"Frame {i}: No suitable match found in brightness lookup, using original frame")
                    processed_frame = current_frame
                
                # Apply contrast stabilization if enabled
                if contrast_stabilization:
                    processed_frame = self.apply_contrast_stabilization(
                        processed_frame, shadow_strength, highlight_strength,
                        shadow_threshold, highlight_threshold,
                        shadow_anti_banding, highlight_anti_banding
                    )
                
                processed_frames.append(processed_frame)
        
        return torch.stack(processed_frames, dim=0)

    def scale(self, image, width, height, upscale_method, crop):
        # Get the dimensions of the image
        original_height, original_width = image.shape[1], image.shape[2]
        # if we are not actually scaling, just return the original image
        if width == original_width and height == original_height:
            # print(f"no scaling needed from {original_width}x{original_height} to {width}x{height}")
            return image

        # print(f"scaling from {original_width}x{original_height} to {width}x{height}")

        # Move dimensions for processing
        samples = image.movedim(-1, 1)
        # Perform the upscale
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1, -1)
        
        return s


    def upscaleWithModel(
        self,
        upscale_model: ImageModelDescriptor,
        image: torch.Tensor,
        device_preference: str = "auto",
    ):
        upscale_amount = upscale_model.scale

        # Determine the device based on device_preference
        if device_preference == "auto":
            device = model_management.get_torch_device()
        elif device_preference == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = model_management.get_torch_device()  # Fallback to auto
        elif device_preference == "cpu":
            device = "cpu"
        else:
            raise ValueError(f"Invalid device preference: {device_preference}")

        logging.info(f"VTSImageUpscaleWithModel - upscale_model.scale = {upscale_model.scale}, size_requirements= {upscale_model.size_requirements}, device = {device}")

        # Memory management
        if device != "cpu":
            memory_required = model_management.module_size(upscale_model.model)
            memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_amount, 1.0) * 384.0  # Estimate
            memory_required += image.nelement() * image.element_size()
            model_management.free_memory(memory_required, device)

        # Move model to the selected device
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                # Calculate the number of steps for tiling
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)

                # Perform tiled scaling
                s = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_amount,  # Pass the correct upscale amount
                    pbar=pbar,
                )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        # Move model back to CPU and clamp output
        if upscale_model.device != "cpu":
            upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return s

NODE_CLASS_MAPPINGS = {
    "WanVideoDiffusionForcingSampler": WanVideoDiffusionForcingSampler,
    "WanVideoLoopingDiffusionForcingSampler": WanVideoLoopingDiffusionForcingSampler,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoDiffusionForcingSampler": "WanVideo Diffusion Forcing Sampler",
    "WanVideoLoopingDiffusionForcingSampler": "WanVideo Looping Diffusion Forcing Sampler",
    }
