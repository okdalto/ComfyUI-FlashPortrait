"""
Tiled VAE Implementation for FlashPortrait

This implementation follows LightX2V's tiled_encode/tiled_decode exactly:
- Uses low-level encoder/decoder instead of high-level encode/decode API
- Temporal chunking: first frame alone, then 4 frames at a time for encoding
- Frame-by-frame decoding with feature cache
- Scale parameter for latent normalization
- Feature map caching for causal convolutions

Reference: LightX2V/lightx2v/models/video_encoders/hf/wan/vae.py
"""

import torch
import torch.nn.functional as F


CACHE_T = 2


def count_conv3d(model):
    """Count CausalConv3d layers for cache initialization"""
    count = 0
    for m in model.modules():
        if hasattr(m, '_padding'):  # CausalConv3d has _padding attribute
            count += 1
    return count


class TiledVAEWrapper:
    """
    Tile VAE Wrapper for processing high-resolution videos
    
    This implementation matches LightX2V exactly:
    1. Uses low-level encoder/decoder with feature caching
    2. Temporal chunking: 1 frame first, then 4 frames at a time (encode)
    3. Frame-by-frame processing (decode)
    4. Scale parameter for latent normalization
    
    Usage:
    - Only needed for standard VAE + high resolution (720P+)
    - Not needed for Tiny VAE (already lightweight) or low resolution
    """
    
    def __init__(
        self,
        vae,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_stride_height=192,
        tile_sample_stride_width=192
    ):
        self.vae = vae
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width
        
        self.spatial_compression_ratio = getattr(vae, 'spatial_compression_ratio', 8)
        self.temporal_compression_ratio = getattr(vae, 'temporal_compression_ratio', 4)
        
        # Get internal model (AutoencoderKLWan_ inside AutoencoderKLWan)
        self._internal_model = getattr(vae, 'model', vae)
        
        # Get scale parameters
        if hasattr(vae, 'scale'):
            self.scale = vae.scale
        elif hasattr(vae, 'mean') and hasattr(vae, 'std'):
            self.scale = [vae.mean.to(vae.dtype if hasattr(vae, 'dtype') else torch.float32), 
                         1.0 / vae.std.to(vae.dtype if hasattr(vae, 'dtype') else torch.float32)]
        else:
            self.scale = [0, 1]
        
        # z_dim for scale reshape
        self.z_dim = getattr(vae.config, 'latent_channels', 16) if hasattr(vae, 'config') else 16
        
    def clear_cache(self):
        """Initialize feature caches for causal convolutions (matches LightX2V)"""
        model = self._internal_model
        self._conv_num = count_conv3d(model.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(model.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
        
    def blend_v(self, a, b, blend_extent):
        """Vertical blending (matches LightX2V exactly)"""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b
    
    def blend_h(self, a, b, blend_extent):
        """Horizontal blending (matches LightX2V exactly)"""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b
    
    def tiled_encode(self, x):
        """
        Encode video with tiling for high resolution (matches LightX2V exactly)
        
        Key differences from simple implementation:
        1. Uses low-level encoder + conv1 instead of high-level encode
        2. Temporal chunking: first frame alone, then 4 frames at a time
        3. Feature map caching for causal convolutions
        4. Scale parameter normalization
        
        Args:
            x: [B, C, T, H, W] video tensor (pixel values in [-1, 1] or [0, 1])
            
        Returns:
            Encoded latents [B, C_latent, T_latent, H_latent, W_latent]
        """
        model = self._internal_model
        scale = self.scale
        
        # Get model dtype for consistency
        model_dtype = next(model.parameters()).dtype
        x = x.to(model_dtype)
        
        _, _, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self.clear_cache()
                time = []
                # Temporal chunking: 1 frame first, then 4 frames at a time
                frame_range = 1 + (num_frames - 1) // 4
                for k in range(frame_range):
                    self._enc_conv_idx = [0]
                    if k == 0:
                        tile = x[:, :, :1, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + 4 * (k - 1) : 1 + 4 * k,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                    # Use low-level encoder with cache
                    tile = model.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    mu, log_var = model.conv1(tile).chunk(2, dim=1)
                    
                    # Apply scale normalization
                    if isinstance(scale[0], torch.Tensor):
                        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1).to(mu.device)) * scale[1].view(1, self.z_dim, 1, 1, 1).to(mu.device)
                    else:
                        mu = (mu - scale[0]) * scale[1]

                    time.append(mu)

                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc
    
    def tiled_decode(self, z):
        """
        Decode latents with tiling for high resolution (matches LightX2V exactly)
        
        Key differences from simple implementation:
        1. Uses low-level conv2 + decoder instead of high-level decode
        2. Frame-by-frame processing (each frame independently)
        3. Feature map caching for causal convolutions
        4. Scale parameter de-normalization
        
        Args:
            z: [B, C_latent, T_latent, H_latent, W_latent] latent tensor
            
        Returns:
            Decoded video [B, C, T, H, W]
        """
        model = self._internal_model
        scale = self.scale
        
        # Get model dtype for consistency
        model_dtype = next(model.parameters()).dtype
        z = z.to(model_dtype)
        
        # Apply inverse scale normalization
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1).to(z.device, z.dtype) + scale[0].view(1, self.z_dim, 1, 1, 1).to(z.device, z.dtype)
        else:
            z = z / scale[1] + scale[0]

        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                # Frame-by-frame decoding
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    tile = model.conv2(tile)
                    decoded = model.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        return dec
    
    def encode(self, x, return_dict=True):
        """High-level encode with tiling support"""
        if x.shape[-2] >= self.tile_sample_min_height and x.shape[-1] >= self.tile_sample_min_width:
            mu = self.tiled_encode(x)
            # DiagonalGaussianDistribution expects [mu, logvar] concatenated along channel dim
            # It will split by channel, so we need to provide 2*z_dim channels
            # Use zeros for logvar since we're using deterministic encoding (mode)
            logvar = torch.zeros_like(mu)
            latent = torch.cat([mu, logvar], dim=1)
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
            from diffusers.models.modeling_outputs import AutoencoderKLOutput
            posterior = DiagonalGaussianDistribution(latent)
            if not return_dict:
                return (posterior,)
            return AutoencoderKLOutput(latent_dist=posterior)
        else:
            return self.vae.encode(x, return_dict=return_dict)
    
    def decode(self, z, return_dict=True):
        """High-level decode with tiling support"""
        expected_height = z.shape[-2] * self.spatial_compression_ratio
        expected_width = z.shape[-1] * self.spatial_compression_ratio
        
        if expected_height >= self.tile_sample_min_height and expected_width >= self.tile_sample_min_width:
            decoded = self.tiled_decode(z).clamp_(-1, 1)
            from diffusers.models.autoencoders.vae import DecoderOutput
            if not return_dict:
                return (decoded,)
            return DecoderOutput(sample=decoded)
        else:
            return self.vae.decode(z, return_dict=return_dict)
    
    def __getattr__(self, name):
        if name in ('vae', 'tile_sample_min_height', 'tile_sample_min_width', 
                    'tile_sample_stride_height', 'tile_sample_stride_width',
                    'spatial_compression_ratio', 'temporal_compression_ratio',
                    '_internal_model', 'scale', 'z_dim',
                    '_conv_num', '_conv_idx', '_feat_map',
                    '_enc_conv_num', '_enc_conv_idx', '_enc_feat_map'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.vae, name)


def setup_tiled_vae(
    vae,
    tile_sample_min_height=256,
    tile_sample_min_width=256,
    tile_sample_stride_height=192,
    tile_sample_stride_width=192
):
    """
    Wrap VAE with tiling support for high resolution processing
    
    This implementation matches LightX2V's tiled_encode/tiled_decode exactly:
    - Low-level encoder/decoder access
    - Temporal chunking for encoding (1 frame + 4 frames groups)
    - Frame-by-frame decoding
    - Feature map caching for causal convolutions
    
    Args:
        vae: Original VAE instance (AutoencoderKLWan)
        tile_sample_min_height: Height of each tile (default 256)
        tile_sample_min_width: Width of each tile (default 256)
        tile_sample_stride_height: Vertical stride between tiles (default 192, overlap=64)
        tile_sample_stride_width: Horizontal stride between tiles (default 192, overlap=64)
        
    Returns:
        Wrapped VAE with tiling support
    """
    return TiledVAEWrapper(
        vae=vae,
        tile_sample_min_height=tile_sample_min_height,
        tile_sample_min_width=tile_sample_min_width,
        tile_sample_stride_height=tile_sample_stride_height,
        tile_sample_stride_width=tile_sample_stride_width
    )

