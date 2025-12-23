# FlashPortrait (ComfyUI Version)

<a href='https://francis-rings.github.io/FlashPortrait'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='http://arxiv.org/abs/2512.16900'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/FrancisRing/FlashPortrait/tree/main'><img src='https://img.shields.io/badge/HuggingFace-Model-orange'></a> 
<a href='https://github.com/comfyanonymous/ComfyUI'><img src='https://img.shields.io/badge/ComfyUI-Custom_Node-blue'></a>

This repository provides [**FlashPortrait**](https://github.com/Francis-Rings/FlashPortrait) custom nodes for ComfyUI. 
It allows you to generate **infinite-length portrait animations** driven by a video, directly within your ComfyUI workflow.

*Original Paper*: **FlashPortrait: 6$\times$ Faster Infinite Portrait Animation with Adaptive Latent Prediction**

---

## ✨ Features
*   **Automatic Model Downloading**: No manual weight placement required. The loader fetches `Wan2.1` and `FlashPortrait` models from Hugging Face automatically.
*   **Infinite Length Support**: Uses FlashPortrait's sliding window mechanism to process long driving videos.
*   **Flexible Inputs**:
    *   **Reference Image**: Defines the identity (ID).
    *   **Driving Video**: Defines the motion and expression.
    *   **Initial Video Frames** (Optional): Allows pre-defining the canvas content if distinct from the reference tiling.

## �️ Installation

1.  **Clone the Repository**:
    Go to your ComfyUI `custom_nodes` folder and run:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/Francis-Rings/FlashPortrait
    cd FlashPortrait
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Requires approx. 40GB VRAM for full model loading (BF16).*

3.  **Restart ComfyUI**.

## � Nodes Usage

### 1. FlashPortrait Loader
Loads the heavy models (Transformer, VAE) and face alignment tools.
*   **precision**: `bf16` (recommended), `fp16`, `fp32`.
*   **download_missing**: Enable this to automatically download models to `ComfyUI/models/flash_portrait/` on first run.

### 2. FlashPortrait Feature Extractor
Extracts motion and expression features from the driving video. 
*   **images**: Connect your driving video frames here (use `Load Video` or similar nodes).
*   **source_fps**: Frame rate of the source video (default 25.0).
*   **context_size** / **context_overlap**: Controls the sliding window size for feature extraction.

### 3. FlashPortrait Sampler
The main generation node.
*   **pipe**: Connect from Loader.
*   **head_emo_features**: Connect from Feature Extractor.
*   **image**: The **Reference Image** (Identity). Only the first frame is used.
*   **initial_video_frames** (Optional): If provided, these frames define the starting canvas/background instead of simply tiling the reference image.
*   **prompt** / **negative_prompt**: Text guidance.
*   **guidance_scale** / **text_cfg_scale** / **emo_cfg_scale**: Control the influence of unconditional, text, and emotion guidance.
*   **steps**: Inference steps (default 30).
*   **max_size**: Output resolution height (e.g., 720).

## 🧱 Acknowledgments
This implementation is based on the amazing work by the original FlashPortrait team.
**Huge congratulations and thanks to:**
**Shuyuan Tu, Yueming Pan, Yinming Huang, Xintong Han, Zhen Xing, Qi Dai, Kai Qiu, Chong Luo, Zuxuan Wu**

And to the open-source projects that made this possible:
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [PD-FGC](https://github.com/Dorniwang/PD-FGC-inference)
- [FantasyPortrait](https://github.com/Fantasy-AMAP/fantasy-portrait)
- [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)

## 📄 Citation
```bib
@article{tu2025flashportrait,
  title={FlashPortrait: 6$\times$ Faster Infinite Portrait Animation with Adaptive Latent Prediction},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Qiu, Kai and Luo, Chong and Wu, Zuxuan},
  journal={arXiv preprint arXiv:2512.16900},
  year={2025}
}
```
