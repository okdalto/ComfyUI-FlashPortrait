import os
import gc
import sys
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# ComfyUI imports
import folder_paths
import comfy.model_management as mm
import comfy.utils

# FlashPortrait / Wan modules
from diffusers import FlowMatchEulerDiscreteScheduler
from .wan.models.face_align import FaceAlignment
from .wan.models.face_model import FaceModel
from .wan.models.pdf import FanEncoder, det_landmarks, get_drive_expression_pd_fgc
from .wan.models.portrait_encoder import PortraitEncoder
from .wan.pipeline.pipeline_wan_long import WanI2VLongPipeline
from .wan.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
from .wan.utils.utils import filter_kwargs
from .wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from .wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .wan.utils.fp8_optimization import (
    replace_parameters_by_name, 
    convert_model_weight_to_float8, 
    convert_weight_dtype_wrapper
)

from comfy.utils import ProgressBar

# Utils
def find_replacement(a):
    while a > 0:
        if (a - 1) % 4 == 0: return a
        a -= 1
    return 0

class FlashPortraitLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "GPU_memory_mode": (["default", "sequential_cpu_offload", "model_cpu_offload_and_qfloat8"], {"default": "default"}),
                "download_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLASH_PORTRAIT_PIPE", "FACE_ALIGN_MODELS")
    RETURN_NAMES = ("pipe", "face_align_models")
    FUNCTION = "load_models"
    CATEGORY = "FlashPortrait"

    def load_models(self, precision, GPU_memory_mode, download_missing):
        device = mm.get_torch_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Define base path in ComfyUI/models/flash_portrait
        base_path = os.path.join(folder_paths.models_dir, "flash_portrait")
        
        # Define sub-paths
        wan_path = os.path.join(base_path, "Wan2.1-I2V-14B-720P")
        fp_path = os.path.join(base_path, "FlashPortrait")

        # Download if requested and missing
        if download_missing:
            if not os.path.exists(wan_path) or not os.listdir(wan_path):
                print(f"Downloading Wan2.1-I2V-14B-720P to {wan_path}...")
                snapshot_download(repo_id="Wan-AI/Wan2.1-I2V-14B-720P", local_dir=wan_path)
            
            if not os.path.exists(fp_path) or not os.listdir(fp_path):
                print(f"Downloading FlashPortrait to {fp_path}...")
                snapshot_download(repo_id="FrancisRing/FlashPortrait", local_dir=fp_path)

        # Config paths
        config_path = os.path.join(os.path.dirname(__file__), "config/wan2.1/wan_civitai.yaml")
        config = OmegaConf.load(config_path)

        # Load Heavy Models (Transformer, VAE, Encoders)
        print(">>> Loading Transformer...")
        transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(wan_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True, torch_dtype=dtype
        )
        
        # Load FlashPortrait transformer weights
        transformer_fp_path = os.path.join(fp_path, "transformer.pt")
        st = load_file(transformer_fp_path) if transformer_fp_path.endswith("safetensors") else torch.load(transformer_fp_path, map_location="cpu", weights_only=True)
        transformer.load_state_dict(st.get("state_dict", st), strict=False)
        del st
        gc.collect()
        transformer.to(device) # Keep on device or let Comfy manage? For now, manual management as per infer.py structure.

        print(">>> Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(wan_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])
        ).to(device=device, dtype=dtype)

        print(">>> Loading Text Encoder & Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(wan_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(wan_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True, torch_dtype=dtype
        ).to(device).eval()

        print(">>> Loading CLIP Image Encoder...")
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(wan_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'))
        ).to(device=device, dtype=dtype).eval()

        print(">>> Loading Portrait Encoder...")
        pe_path = os.path.join(fp_path, "portrait_encoder.pt")
        pe_st = torch.load(pe_path, map_location="cpu", weights_only=True)
        portrait_encoder = PortraitEncoder(adapter_in_dim=768, adapter_proj_dim=2048)
        for p, sub_m in [("proj_model.", portrait_encoder.proj_model), 
                         ("mouth_proj_model.", portrait_encoder.mouth_proj_model), 
                         ("emo_proj_model.", portrait_encoder.emo_proj_model)]:
            sub_m.load_state_dict({k[len(p):]: v for k, v in pe_st.items() if k.startswith(p)})
        del pe_st
        gc.collect()
        portrait_encoder.to(device).eval()

        # Pack pipeline components
        # Scheduler setup (default to Flow)
        sc_cls = FlowMatchEulerDiscreteScheduler
        scheduler = sc_cls(**filter_kwargs(sc_cls, OmegaConf.to_container(config['scheduler_kwargs'])))

        pipeline = WanI2VLongPipeline(
            transformer=transformer, vae=vae, tokenizer=tokenizer, 
            text_encoder=text_encoder, scheduler=scheduler, 
            clip_image_encoder=clip_image_encoder, portrait_encoder=portrait_encoder
        )

        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation",], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, dtype)

        pipe = {
            "pipeline": pipeline,
            "transformer": transformer,
            "vae": vae,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "clip_image_encoder": clip_image_encoder,
            "portrait_encoder": portrait_encoder,
            "config": config,
            "device": device,
            "dtype": dtype
        }

        # Load Face Alignment Models (CPU friendly until used)
        print(">>> Loading Face Alignment Models...")
        det_path = os.path.join(fp_path, "face_det.onnx")
        align_path = os.path.join(fp_path, "face_landmark.onnx")
        pd_fpg_path = os.path.join(fp_path, "pd_fpg.pth")

        # Initialize these later in extractor to save memory? 
        # But we need paths or loaded models. infer.py loads them in Phase 1 and deletes.
        # We'll pass the paths or re-load them in extractor.
        # Let's verify paths exist.
        if not all(os.path.exists(p) for p in [det_path, align_path, pd_fpg_path]):
             raise FileNotFoundError("Some face alignment models are missing.")

        face_align_models = {
            "det_path": det_path,
            "align_path": align_path,
            "pd_fpg_path": pd_fpg_path
        }

        return (pipe, face_align_models)


class FlashPortraitFeatureExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_align_models": ("FACE_ALIGN_MODELS",),
                "images": ("IMAGE",), # [B, H, W, 3]
                "source_fps": ("FLOAT", {"default": 25.0}),
                "context_size": ("INT", {"default": 51}),
                "context_overlap": ("INT", {"default": 30}),
            }
        }

    RETURN_TYPES = ("HEAD_EMO_FEAT", "FLOAT", "INT")
    RETURN_NAMES = ("head_emo_features", "fps", "total_frames")
    FUNCTION = "extract_features"
    CATEGORY = "FlashPortrait"

    def extract_features(self, face_align_models, images, source_fps, context_size, context_overlap):
        device = mm.get_torch_device()
        
        # Prepare Feature Extractor Models
        print(">>> Initializing Face Models for extraction...")
        fa = FaceModel(FaceAlignment(None, face_align_models["align_path"], face_align_models["det_path"]), reset=False)
        fm = FanEncoder().to(device).eval()
        fm.load_state_dict(torch.load(face_align_models["pd_fpg_path"], map_location=device), strict=False)

        # Convert IMAGE tensor (B,H,W,C) 0-1 to List of numpy (H,W,C) 0-255 BGR (opencv format)
        # Comfy IMAGE is RGB. cv2 prefers BGR.
        # batch_images = (images.cpu().numpy() * 255).astype(np.uint8)
        # frame_list = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in batch_images]
        
        # Actually infer.py uses: frame_list.append(frame.copy()) where frame is from cap.read() (BGR)
        # But let's check det_landmarks implementation. It usually expects BGR if using cv2.
        # If I look at infer.py -> det_landmarks, it takes frame_list.
        
        # Let's convert to BGR just to be safe if underlying code uses cv2.
        batch = (images.cpu().numpy() * 255).astype(np.uint8)
        frame_list = []
        for i in range(batch.shape[0]):
             # RGB to BGR
             frame = batch[i][..., ::-1].copy() 
             frame_list.append(frame)

        num_frames = find_replacement(len(frame_list))
        frame_list = frame_list[:num_frames]
        
        print(f">>> Extracting features from {len(frame_list)} frames...")
        # Landmark & Emotion Extraction
        with torch.no_grad():
             landmark_list = det_landmarks(fa, frame_list)[1]
             emo_list = get_drive_expression_pd_fgc(fm, frame_list, landmark_list, device)

        ef_list, hef_list = [], []
        for emo in emo_list:
            ef = torch.cat([emo["eye_embed"], emo["emo_embed"], emo["mouth_feat"]], dim=1)
            hef = torch.cat([emo["headpose_emb"], ef], dim=1)
            ef_list.append(ef)
            hef_list.append(hef)
        
        head_emo = torch.cat(hef_list, dim=0) # [NumFrames, FeatureDim]

        # Stride compensation from infer.py
        raw_num = num_frames
        stride = context_size - context_overlap
        if raw_num > context_size:
             final_num_frames = context_size + ((raw_num - context_size) // stride) * stride
        else:
             final_num_frames = context_size
        
        # Pad or slice? infer.py does: head_emo = head_emo[:num_frames] (where num_frames matches calculation)
        # But wait, raw_num is find_replacement(len).
        # infer.py lines 182-184:
        # stride = cfg.context_size - cfg.context_overlap
        # num_frames = cfg.context_size + ((raw_num - cfg.context_size) // stride) * stride if raw_num > cfg.context_size else cfg.context_size
        # head_emo = head_emo[:num_frames].unsqueeze(0).to(device)

        # If actual frames < required frames, we might need to be careful. 
        # But usually raw_num >= num_frames (calculated) unless raw_num < context_size?
        # If raw_num < context_size, num_frames=context_size. head_emo length is raw_num.
        # So we might need to repeat?
        # infer.py doesn't seem to repeat explicitly there, but head_emo[:num_frames] implies it has enough?
        # No, if raw_num < calculated, slicing will fail or return shorter.
        # Let's implement safe logic.
        
        if head_emo.shape[0] < final_num_frames:
             # Repeat last frame to fill
             diff = final_num_frames - head_emo.shape[0]
             last = head_emo[-1:]
             head_emo = torch.cat([head_emo, last.repeat(diff, 1)], dim=0)
        
        head_emo = head_emo[:final_num_frames].unsqueeze(0) # [1, T, D]

        # Cleanup
        del fa, fm, frame_list, landmark_list, emo_list
        torch.cuda.empty_cache()
        gc.collect()

        return (head_emo, source_fps, final_num_frames)


class FlashPortraitSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("FLASH_PORTRAIT_PIPE",),
                "head_emo_features": ("HEAD_EMO_FEAT",),
                "image": ("IMAGE",), # Reference image
                "prompt": ("STRING", {"multiline": True, "default": "The man is singing"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0}), # Text CFG ?? infer.py says text_cfg_scale default 1.0, guidance 4.0?
                # infer.py: guidance_scale=4.0 (for uncoditional?), text_cfg_scale=1.0, emo_cfg_scale=4.0
                "guidance_scale": ("FLOAT", {"default": 4.0}), 
                "text_cfg_scale": ("FLOAT", {"default": 1.0}),
                "emo_cfg_scale": ("FLOAT", {"default": 4.0}),
                "max_size": ("INT", {"default": 720}),
                "shift": ("FLOAT", {"default": 5.0}),
                "context_size": ("INT", {"default": 51}),
                "context_overlap": ("INT", {"default": 30}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "FlashPortrait"

    def sample(self, pipe, head_emo_features, image, prompt, negative_prompt, seed, steps, 
               cfg_scale, guidance_scale, text_cfg_scale, emo_cfg_scale, max_size, shift,
               context_size, context_overlap):
        
        transformer = pipe["transformer"]
        vae = pipe["vae"]
        tokenizer = pipe["tokenizer"]
        text_encoder = pipe["text_encoder"]
        clip_image_encoder = pipe["clip_image_encoder"]
        portrait_encoder = pipe["portrait_encoder"]
        config = pipe["config"]
        device = pipe["device"]
        dtype = pipe["dtype"]

        if "pipeline" in pipe:
            pipeline = pipe["pipeline"]
        else:
            # Fallback for older workflows or if pipeline wasn't created in loader
            # Scheduler setup (default to Flow)
            sc_cls = FlowMatchEulerDiscreteScheduler
            scheduler = sc_cls(**filter_kwargs(sc_cls, OmegaConf.to_container(config['scheduler_kwargs'])))

            pipeline = WanI2VLongPipeline(
                transformer=transformer, vae=vae, tokenizer=tokenizer, 
                text_encoder=text_encoder, scheduler=scheduler, 
                clip_image_encoder=clip_image_encoder, portrait_encoder=portrait_encoder
            )
        
        # Prepare Input Image
        # Comfy Image is [B,H,W,C] RGB Tensor 0-1
        # Pipeline expects PIL Image for resize logic, then Tensor.
        # Let's assume batch size 1 for reference image for now.
        ref_img_tensor = image[0] 
        ref_img_np = (ref_img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(ref_img_np).convert("RGB")

        # Resize logic from infer.py
        num_frames = head_emo_features.shape[1]
        # Adjust num_frames to satisfy (N-1)%4 == 0 constraint
        if (num_frames - 1) % 4 != 0:
            num_frames = (num_frames - 1) // 4 * 4 + 1
        
        # Prepare head_emo
        head_emo = head_emo_features[:, :num_frames].to(device)

        sub_num_frames = num_frames
        
        # Calculate latent frames similar to pipeline
        vae_temporal_compression = 4 # Hardcoded or get from config if possible (pipe['config']['vae_kwargs']['temporal_compression_ratio'])
        latents_num_frames = (num_frames - 1) // vae_temporal_compression + 1
        infer_length = latents_num_frames

        # Resize logic from infer.py
        scale = max_size / max(img.size)
        w, h = (int(img.size[0] * scale) // 16 * 16, int(img.size[1] * scale) // 16 * 16)
        img = img.resize((w, h), Image.LANCZOS)

        # Create input_video with correct length
        input_video = torch.tile(torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(1).unsqueeze(0), [1, 1, sub_num_frames, 1, 1]).to(dtype) / 255.0
        
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, 1:] = 255
        
        # If video is shorter than context window, reduce context size to match
        # Ensure effective_context_size is positive and safe
        effective_context_size = max(1, min(context_size, infer_length))
        
        # ComfyUI Progress Bar
        pbar = ProgressBar(steps)
        def callback(pipe, step, timestep, callback_kwargs):
            pbar.update(1)

        with torch.no_grad():
            sample = pipeline(
                prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames,
                height=h, width=w, generator=torch.Generator(device=device).manual_seed(seed),
                video=input_video, mask_video=input_video_mask, clip_image=img, shift=shift,
                context_size=effective_context_size, context_overlap=context_overlap,
                latents_num_frames=latents_num_frames, sub_num_frames=sub_num_frames,
                head_emo_feat_all=head_emo, guidance_scale=guidance_scale, num_inference_steps=steps,
                text_cfg_scale=text_cfg_scale, emo_cfg_scale=emo_cfg_scale,
                callback_on_step_end=callback
            ).videos
            
            # sample result is [B, C, F, H, W]??
            # Pipeline output .videos usually is tensor.
            # infer.py: sample = sample[:, :, 1:] (removing first frame?)
            
            sample = sample[:, :, 1:] 
            
            # Convert to Comfy output [F, H, W, C]
            # sample shape: [B, C, F, H, W] -> [B, F, H, W, C]
            output = sample.permute(0, 2, 3, 4, 1)
            output = output[0] # Take first batch
            output = output.cpu()

        return (output,)

NODE_CLASS_MAPPINGS = {
    "FlashPortraitLoader": FlashPortraitLoader,
    "FlashPortraitFeatureExtractor": FlashPortraitFeatureExtractor,
    "FlashPortraitSampler": FlashPortraitSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashPortraitLoader": "FlashProtrait Model Loader",
    "FlashPortraitFeatureExtractor": "FlashPortrait Feature Extractor",
    "FlashPortraitSampler": "FlashPortrait Sampler"
}
