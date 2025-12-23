import torch
from wan.models.wan_vae_tiny import WanVAE_tiny, Wan2_2_VAE_tiny


class DecoderOutput:
    """Simple wrapper to match the output format expected by pipeline"""
    def __init__(self, sample):
        self.sample = sample


def setup_tiny_vae(pipeline, model_type="wan2.1", tiny_vae_path=None, parallel_decode=False, need_scaled=False):
    """
    Replace pipeline VAE with tiny VAE for faster decoding.
    
    Args:
        pipeline: WanI2VLongPipeline instance
        model_type: "wan2.1" or "wan2.2"
        tiny_vae_path: path to tiny VAE checkpoint (.pth file)
        parallel_decode: if True, use parallel mode (faster but more memory)
        need_scaled: if True, apply latent normalization
        
    Returns:
        Modified pipeline with tiny VAE
    """
    import os
    
    if model_type == "wan2.2":
        tiny_vae_cls = Wan2_2_VAE_tiny
        default_path = "taew2_2.pth"
    else:
        tiny_vae_cls = WanVAE_tiny
        default_path = "taew2_1.pth"
    
    vae_path = tiny_vae_path if tiny_vae_path is not None else default_path
    
    if not os.path.exists(vae_path):
        raise FileNotFoundError(
            f"Tiny VAE checkpoint not found: {vae_path}\n"
            f"Please download the checkpoint or provide correct path via tiny_vae_path parameter.\n"
            f"Expected location: {os.path.abspath(vae_path)}"
        )
    
    original_vae = pipeline.vae
    device = original_vae.device if hasattr(original_vae, 'device') else torch.device("cuda")
    dtype = original_vae.dtype if hasattr(original_vae, 'dtype') else torch.bfloat16
    
    tiny_vae = tiny_vae_cls(
        vae_path=vae_path,
        dtype=dtype,
        device=device,
        need_scaled=need_scaled
    ).to(device)
    
    tiny_vae.config = original_vae.config
    tiny_vae.encode = original_vae.encode
    
    tiny_vae.temporal_compression_ratio = getattr(original_vae, 'temporal_compression_ratio', 4)
    tiny_vae.spatial_compression_ratio = getattr(original_vae, 'spatial_compression_ratio', 8)
    tiny_vae.latent_channels = getattr(original_vae.config, 'latent_channels', 16)
    
    original_decode = tiny_vae.decode
    def patched_decode(latents, **kwargs):
        result = original_decode(latents, parallel=parallel_decode)
        return DecoderOutput(sample=result)
    tiny_vae.decode = patched_decode
    
    pipeline.vae = tiny_vae
    
    return pipeline
