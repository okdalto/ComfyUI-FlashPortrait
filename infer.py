import os
import sys
import cv2
import gc
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from safetensors.torch import load_file

# Wan & FlashPortrait modules
from diffusers import FlowMatchEulerDiscreteScheduler
from wan.models.face_align import FaceAlignment
from wan.models.face_model import FaceModel
from wan.models.pdf import FanEncoder, det_landmarks, get_drive_expression_pd_fgc
from wan.models.portrait_encoder import PortraitEncoder
from wan.pipeline.pipeline_wan_long import WanI2VLongPipeline
from wan.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
from wan.utils.utils import simple_save_videos_grid, filter_kwargs
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from xfuser.core.distributed import (get_world_group, init_distributed_environment, 
                                     initialize_model_parallel, get_sp_group)

# ==========================================================================================
# 1. Configuration
# ==========================================================================================
class GlobalConfig:
    wan_model_name = "./checkpoints/Wan2.1-I2V-14B-720P"
    transformer_path = "./checkpoints/FlashPortrait/transformer.pt"
    portrait_encoder_path = "./checkpoints/FlashPortrait/portrait_encoder.pt"
    det_model_path = "./checkpoints/FlashPortrait/face_det.onnx"
    alignment_model_path = "./checkpoints/FlashPortrait/face_landmark.onnx"
    pd_fpg_model_path = "./checkpoints/FlashPortrait/pd_fpg.pth"
    config_path = "config/wan2.1/wan_civitai.yaml"
    
    ulysses_degree = 1
    ring_degree = 1
    weight_dtype = torch.bfloat16
    sampler_name = "Flow"
    shift = 5
    max_size = 720
    sub_num_frames = 201
    latents_num_frames = 51
    context_overlap = 30
    context_size = 51
    num_inference_steps = 30
    seed = 42
    
    prompt = "The man is singing"
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    validation_image_start = "./examples/doecii/chim.png"
    validation_driven_video_path = "./examples/doecii/doecii.mp4"
    save_path = "samples/wan-videos-i2v"

# ==========================================================================================
# 2. Utility & Helper Functions
# ==========================================================================================

def setup_distributed(cfg):
    """분산 처리 환경 설정 함수"""
    if cfg.ulysses_degree > 1 or cfg.ring_degree > 1:
        if get_sp_group is None:
            raise RuntimeError("xfuser is not installed.")
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(), 
            ring_degree=cfg.ring_degree, 
            ulysses_degree=cfg.ulysses_degree
        )
        return torch.device(f"cuda:{get_world_group().local_rank}")
    return torch.device("cuda")

def find_replacement(a):
    """프레임 수를 4의 배수로 맞추는 함수 (원본 로직)"""
    while a > 0:
        if (a - 1) % 4 == 0: return a
        a -= 1
    return 0

def get_emo_feature(video_path, face_aligner, pd_fpg_motion, device):
    """드라이빙 비디오에서 감정 및 포즈 특징 추출"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_list.append(frame.copy())
    cap.release()
    
    num_frames = find_replacement(len(frame_list))
    frame_list = frame_list[:num_frames]
    
    # 랜드마크 및 감정 추출
    landmark_list = det_landmarks(face_aligner, frame_list)[1]
    emo_list = get_drive_expression_pd_fgc(pd_fpg_motion, frame_list, landmark_list, device)
    
    ef_list, hef_list = [], []
    for emo in emo_list:
        ef = torch.cat([emo["eye_embed"], emo["emo_embed"], emo["mouth_feat"]], dim=1)
        hef = torch.cat([emo["headpose_emb"], ef], dim=1)
        ef_list.append(ef)
        hef_list.append(hef)
        
    return torch.cat(ef_list, dim=0), torch.cat(hef_list, dim=0), fps, num_frames

# ==========================================================================================
# 3. Memory-Safe Loading Functions
# ==========================================================================================

def load_wan_pipeline_models(cfg, config, device):
    """메인 생성 모델들을 순차적으로 로드하여 VRAM 안착"""
    print(">>> Loading Transformer...")
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(cfg.wan_model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=cfg.weight_dtype
    )
    st = load_file(cfg.transformer_path) if cfg.transformer_path.endswith("safetensors") else torch.load(cfg.transformer_path, map_location="cpu", weights_only=True)
    transformer.load_state_dict(st.get("state_dict", st), strict=False)
    del st
    gc.collect()
    transformer.to(device)

    print(">>> Loading VAE & Encoders...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(cfg.wan_model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])
    ).to(device=device, dtype=cfg.weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(cfg.wan_model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(cfg.wan_model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=cfg.weight_dtype
    ).to(device).eval()

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(cfg.wan_model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'))
    ).to(device=device, dtype=cfg.weight_dtype).eval()

    print(">>> Loading Portrait Encoder...")
    pe_st = torch.load(cfg.portrait_encoder_path, map_location="cpu", weights_only=True)
    portrait_encoder = PortraitEncoder(adapter_in_dim=768, adapter_proj_dim=2048)
    for p, sub_m in [("proj_model.", portrait_encoder.proj_model), 
                     ("mouth_proj_model.", portrait_encoder.mouth_proj_model), 
                     ("emo_proj_model.", portrait_encoder.emo_proj_model)]:
        sub_m.load_state_dict({k[len(p):]: v for k, v in pe_st.items() if k.startswith(p)})
    del pe_st
    gc.collect()
    portrait_encoder.to(device).eval()

    return transformer, vae, tokenizer, text_encoder, clip_image_encoder, portrait_encoder

# ==========================================================================================
# 4. Main Execution Logic
# ==========================================================================================

def main():
    cfg = GlobalConfig()
    device = setup_distributed(cfg)
    config = OmegaConf.load(cfg.config_path)

    # --- Phase 1: Pre-processing (전처리 후 즉시 메모리 해제) ---
    print(">>> Phase 1: Video Pre-processing...")
    fa = FaceModel(FaceAlignment(None, cfg.alignment_model_path, cfg.det_model_path), reset=False)
    fm = FanEncoder().to(device).eval()
    fm.load_state_dict(torch.load(cfg.pd_fpg_model_path, map_location=device), strict=False)

    with torch.no_grad():
        _, head_emo, fps, raw_num = get_emo_feature(cfg.validation_driven_video_path, fa, fm, device)
    
    # Stride(21) 보정
    stride = cfg.context_size - cfg.context_overlap
    num_frames = cfg.context_size + ((raw_num - cfg.context_size) // stride) * stride if raw_num > cfg.context_size else cfg.context_size
    head_emo = head_emo[:num_frames].unsqueeze(0).to(device)

    del fa, fm
    torch.cuda.empty_cache()
    gc.collect()

    # --- Phase 2: Load Main Pipeline ---
    print(">>> Phase 2: Loading Heavy Models...")
    transformer, vae, tokenizer, text_encoder, clip_image_encoder, portrait_encoder = load_wan_pipeline_models(cfg, config, device)
    
    sc_cls = {"Flow": FlowMatchEulerDiscreteScheduler, "Flow_Unipc": FlowUniPCMultistepScheduler, "Flow_DPM++": FlowDPMSolverMultistepScheduler}[cfg.sampler_name]
    scheduler = sc_cls(**filter_kwargs(sc_cls, OmegaConf.to_container(config['scheduler_kwargs'])))

    pipeline = WanI2VLongPipeline(
        transformer=transformer, vae=vae, tokenizer=tokenizer, 
        text_encoder=text_encoder, scheduler=scheduler, 
        clip_image_encoder=clip_image_encoder, portrait_encoder=portrait_encoder
    )

    # --- Phase 3: Inference ---
    print(">>> Phase 3: Starting Generation...")
    img = Image.open(cfg.validation_image_start).convert("RGB")
    scale = cfg.max_size / max(img.size)
    w, h = (int(img.size[0] * scale) // 16 * 16, int(img.size[1] * scale) // 16 * 16)
    img = img.resize((w, h), Image.LANCZOS)
    
    input_video = torch.tile(torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(1).unsqueeze(0), [1, 1, cfg.sub_num_frames, 1, 1]).to(cfg.weight_dtype) / 255.0
    input_video_mask = torch.zeros_like(input_video[:, :1])
    input_video_mask[:, :, 1:] = 255

    with torch.no_grad():
        sample = pipeline(
            prompt=cfg.prompt, negative_prompt=cfg.negative_prompt, num_frames=num_frames,
            height=h, width=w, generator=torch.Generator(device=device).manual_seed(cfg.seed),
            video=input_video, mask_video=input_video_mask, clip_image=img, shift=cfg.shift,
            context_size=cfg.context_size, context_overlap=cfg.context_overlap,
            latents_num_frames=cfg.latents_num_frames, sub_num_frames=cfg.sub_num_frames,
            head_emo_feat_all=head_emo, guidance_scale=4.0, num_inference_steps=30
        ).videos
        sample = sample[:, :, 1:]

    # --- Phase 4: Save Result ---
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(cfg.save_path, exist_ok=True)
        out_path = os.path.join(cfg.save_path, f"{len(os.listdir(cfg.save_path))+1:08d}.mp4")
        simple_save_videos_grid(sample, out_path, fps=fps)
        print(f">>> Process Completed. Saved to: {out_path}")

if __name__ == "__main__":
    main()