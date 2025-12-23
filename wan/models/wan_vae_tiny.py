import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import os
from tqdm.auto import tqdm


DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), act_func, conv(n_out, n_out), act_func, conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N * T, C, H, W)
        for b in tqdm(model, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        out = []
        work_queue = [TWorkItem(xt, 0) for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))]
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        mem = [None] * len(model)
        while work_queue:
            xt, i = work_queue.pop(0)
            if i == 0:
                progress_bar.update(1)
            if i == len(model):
                out.append(xt)
            else:
                b = model[i]
                if isinstance(b, MemBlock):
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(xt)
                    work_queue.insert(0, TWorkItem(xt_new, i + 1))
                elif isinstance(b, TPool):
                    if mem[i] is None:
                        mem[i] = []
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        raise ValueError("???")
                    elif len(mem[i]) < b.stride:
                        pass
                    else:
                        N, C, H, W = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(N * b.stride, C, H, W))
                        mem[i] = []
                        work_queue.insert(0, TWorkItem(xt, i + 1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C, H, W = xt.shape
                    for xt_next in reversed(xt.view(N, b.stride * C, H, W).chunk(b.stride, 1)):
                        work_queue.insert(0, TWorkItem(xt_next, i + 1))
                else:
                    xt = b(xt)
                    work_queue.insert(0, TWorkItem(xt, i + 1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x


class TAEHV(nn.Module):
    def __init__(self, checkpoint_path="taehv.pth", decoder_time_upscale=(True, True), decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16, model_type="wan21"):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.is_cogvideox = checkpoint_path is not None and "taecvx" in checkpoint_path
        self.model_type = model_type
        if model_type == "wan22":
            self.patch_size, self.latent_channels = 2, 48
        if model_type == "hy15":
            act_func = nn.LeakyReLU(0.2, inplace=True)
        else:
            act_func = nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size**2, 64),
            act_func,
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            act_func,
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            act_func,
            conv(n_f[3], self.image_channels * self.patch_size**2),
        )
        if checkpoint_path is not None:
            ext = os.path.splitext(checkpoint_path)[1].lower()

            if ext == ".pth":
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            elif ext == ".safetensors":
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(checkpoint_path, device="cpu")
                except ImportError:
                    raise ImportError("safetensors is required to load .safetensors files. Install with: pip install safetensors")
            else:
                raise ValueError(f"Unsupported checkpoint format: {ext}. Supported formats: .pth, .safetensors")

            self.load_state_dict(self.patch_tgrow_layers(state_dict))

    def patch_tgrow_layers(self, sd):
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        if x.shape[1] % 4 != 0:
            n_pad = 4 - x.shape[1] % 4
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=True):
        skip_trim = self.is_cogvideox and x.shape[1] % 2 == 0
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        if self.model_type == "hy15":
            x = x.clamp_(-1, 1)
        else:
            x = x.clamp_(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        if skip_trim:
            return x
        return x[:, self.frames_to_trim :]


class WanVAE_tiny(nn.Module):
    def __init__(self, vae_path="taew2_1.pth", dtype=torch.bfloat16, device="cuda", need_scaled=False):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.taehv = TAEHV(vae_path).to(self.dtype)
        self.temperal_downsample = [True, True, False]
        self.need_scaled = need_scaled

        if self.need_scaled:
            self.latents_mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ]

            self.latents_std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
            ]

            self.z_dim = 16

    @torch.no_grad()
    def decode(self, latents, parallel=False):
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)

        if self.need_scaled:
            latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

        return self.taehv.decode_video(latents.transpose(1, 2).to(self.dtype), parallel=parallel, show_progress_bar=False).transpose(1, 2).mul_(2).sub_(1)

    @torch.no_grad()
    def encode_video(self, vid):
        return self.taehv.encode_video(vid)

    @torch.no_grad()
    def decode_video(self, vid_enc):
        return self.taehv.decode_video(vid_enc)


class Wan2_2_VAE_tiny(nn.Module):
    def __init__(self, vae_path="taew2_2.pth", dtype=torch.bfloat16, device="cuda", need_scaled=False):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.taehv = TAEHV(vae_path, model_type="wan22").to(self.dtype)
        self.need_scaled = need_scaled
        if self.need_scaled:
            self.latents_mean = [
                -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
                -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
                -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
                -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
                -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
                0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
            ]

            self.latents_std = [
                0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
                0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
                0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
            ]

            self.z_dim = 48

    @torch.no_grad()
    def decode(self, latents, parallel=False):
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)

        if self.need_scaled:
            latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

        return self.taehv.decode_video(latents.transpose(1, 2).to(self.dtype), parallel=parallel, show_progress_bar=False).transpose(1, 2).mul_(2).sub_(1)

    @torch.no_grad()
    def encode_video(self, vid):
        return self.taehv.encode_video(vid)

    @torch.no_grad()
    def decode_video(self, vid_enc):
        return self.taehv.decode_video(vid_enc)
