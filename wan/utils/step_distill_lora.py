"""
Step Distillation LoRA Utilities for FlashPortrait

This module provides utilities for loading and applying step distillation LoRA
weights to the transformer model. Step distillation LoRA enables 4-step inference
instead of the standard 30+ steps.

The LoRA weights are trained to make the model learn larger "jumps" in the
denoising process, effectively compressing multiple steps into one.

Reference: LightX2V/lightx2v/models/networks/wan/lora_adapter.py
    )
"""

import os
from typing import Dict, Any

import torch
from safetensors.torch import load_file
from loguru import logger


def load_step_distill_lora(
    lora_path: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Load step distillation LoRA weights from file.
    
    Args:
        lora_path: Path to the LoRA safetensors file
        dtype: Data type for weights
        
    Returns:
        Dictionary of LoRA weights
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"Step distillation LoRA not found: {lora_path}")
    
    logger.info(f"Loading step distillation LoRA from: {lora_path}")
    
    if lora_path.endswith(".safetensors"):
        state_dict = load_file(lora_path)
    else:
        state_dict = torch.load(lora_path, map_location="cpu", weights_only=True)
    
    # Convert to target dtype
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    
    logger.info(f"Loaded {len(state_dict)} LoRA weight tensors")
    
    return state_dict


def apply_step_distill_lora(
    pipeline,
    lora_path: str,
    strength: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> Any:
    """
    Apply step distillation LoRA to the pipeline's transformer.
    
    This function follows LightX2V's WanLoraWrapper implementation:
    1. Parse LoRA pairs: lora_down.weight + lora_up.weight -> target.weight
    2. Parse diff values: diff -> target.weight, diff_b -> target.bias
    3. Apply: weight += strength * (lora_B @ lora_A) or weight += strength * diff
    
    Args:
        pipeline: The inference pipeline with transformer
        lora_path: Path to the step distillation LoRA file
        strength: LoRA strength/multiplier (default: 1.0)
        dtype: Data type for computation
        
    Returns:
        Modified pipeline with LoRA applied
    """
    logger.info(f"Applying step distillation LoRA with strength={strength}")
    
    # Load LoRA weights
    lora_weights = load_step_distill_lora(lora_path, dtype)
    
    # Get transformer state dict (reference to actual parameters)
    transformer = pipeline.transformer
    weight_dict = dict(transformer.named_parameters())
    
    # Parse LoRA pairs and diffs following LightX2V's logic
    lora_pairs = {}  # target_name -> (lora_A_key, lora_B_key)
    lora_diffs = {}  # target_name -> diff_key
    
    # LoRA keys are in format: diffusion_model.{layer}.lora_down.weight
    # Transformer keys are in format: {layer}.weight
    # So we need to remove "diffusion_model." prefix
    
    for key in lora_weights.keys():
        # Remove diffusion_model. prefix if present
        if key.startswith("diffusion_model."):
            base_key = key[len("diffusion_model."):]
        else:
            base_key = key
        
        # Try LoRA pair: lora_down.weight + lora_up.weight -> weight
        if key.endswith("lora_down.weight"):
            # Get target weight name (remove .lora_down.weight, add .weight back is implicit)
            target_name = base_key.replace(".lora_down.weight", ".weight")
            pair_key = key.replace("lora_down.weight", "lora_up.weight")
            if pair_key in lora_weights:
                lora_pairs[target_name] = (key, pair_key)
        
        # Alternative LoRA format: lora_A.weight + lora_B.weight
        elif key.endswith("lora_A.weight"):
            target_name = base_key.replace(".lora_A.weight", ".weight")
            pair_key = key.replace("lora_A.weight", "lora_B.weight")
            if pair_key in lora_weights:
                lora_pairs[target_name] = (key, pair_key)
        
        # Diff for weights (e.g., norm, modulation)
        elif key.endswith(".diff"):
            target_name = base_key.replace(".diff", ".weight")
            # Also try without .weight for modulation
            if target_name not in weight_dict:
                target_name = base_key.replace(".diff", "")
            lora_diffs[target_name] = key
        
        # Diff for bias
        elif key.endswith(".diff_b"):
            target_name = base_key.replace(".diff_b", ".bias")
            lora_diffs[target_name] = key
        
        # Diff for modulation (special case)
        elif key.endswith(".diff_m"):
            target_name = base_key.replace(".diff_m", ".modulation")
            lora_diffs[target_name] = key
    
    logger.info(f"Found {len(lora_pairs)} LoRA pairs and {len(lora_diffs)} diff values")
    
    # Apply LoRA weights
    applied_lora_count = 0
    applied_diff_count = 0
    skipped_keys = []
    
    with torch.no_grad():
        # Apply LoRA pairs: weight += strength * (lora_B @ lora_A)
        for target_name, (lora_A_key, lora_B_key) in lora_pairs.items():
            if target_name not in weight_dict:
                skipped_keys.append(f"LoRA pair -> {target_name}")
                continue
            
            param = weight_dict[target_name]
            lora_A = lora_weights[lora_A_key].to(param.device, param.dtype)  # lora_down
            lora_B = lora_weights[lora_B_key].to(param.device, param.dtype)  # lora_up
            
            # Check shape compatibility: W' = W + B @ A
            # lora_A: (rank, in_features), lora_B: (out_features, rank)
            # Result: (out_features, in_features)
            expected_shape = (lora_B.shape[0], lora_A.shape[1])
            
            if param.shape == expected_shape:
                param.data += strength * torch.matmul(lora_B, lora_A)
                applied_lora_count += 1
            elif len(param.shape) == 4 and param.shape[:2] == expected_shape[:2]:
                # Conv2d case
                delta = torch.matmul(
                    lora_B.squeeze(-1).squeeze(-1) if len(lora_B.shape) == 4 else lora_B,
                    lora_A.squeeze(-1).squeeze(-1) if len(lora_A.shape) == 4 else lora_A
                )
                if len(param.shape) == 4:
                    delta = delta.unsqueeze(-1).unsqueeze(-1)
                param.data += strength * delta
                applied_lora_count += 1
            else:
                skipped_keys.append(f"LoRA shape mismatch: {target_name} param={param.shape} vs expected={expected_shape}")
        
        # Apply diff values: weight += strength * diff
        for target_name, diff_key in lora_diffs.items():
            if target_name not in weight_dict:
                skipped_keys.append(f"Diff -> {target_name}")
                continue
            
            param = weight_dict[target_name]
            diff_value = lora_weights[diff_key].to(param.device, param.dtype)
            
            if param.shape == diff_value.shape:
                param.data += strength * diff_value
                applied_diff_count += 1
            else:
                skipped_keys.append(f"Diff shape mismatch: {target_name} param={param.shape} vs diff={diff_value.shape}")
    
    logger.info(f"Applied {applied_lora_count} LoRA weight adjustments")
    logger.info(f"Applied {applied_diff_count} diff value adjustments")
    logger.info(f"Total applied: {applied_lora_count + applied_diff_count}")
    
    if skipped_keys:
        logger.warning(f"Skipped {len(skipped_keys)} keys (first 10):")
        for key in skipped_keys[:10]:
            logger.debug(f"  - {key}")
    
    if applied_lora_count + applied_diff_count == 0:
        logger.error("No LoRA weights were applied! Check key naming conventions.")
        logger.info("Expected LoRA keys: diffusion_model.<layer>.lora_down.weight / lora_up.weight")
        logger.info("Expected diff keys: diffusion_model.<layer>.diff / diff_b")
        
        # Debug: print first few weight_dict keys and lora keys
        logger.debug("First 10 transformer parameter names:")
        for i, k in enumerate(list(weight_dict.keys())[:10]):
            logger.debug(f"  {k}")
        logger.debug("First 10 LoRA keys:")
        for i, k in enumerate(list(lora_weights.keys())[:10]):
            logger.debug(f"  {k}")
    
    return pipeline


def remove_step_distill_lora(
    pipeline,
    lora_path: str,
    strength: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> Any:
    """
    Remove step distillation LoRA from the pipeline's transformer.
    
    This function subtracts the LoRA weights from the transformer,
    restoring the original model behavior.
    
    Args:
        pipeline: The inference pipeline with transformer
        lora_path: Path to the step distillation LoRA file
        strength: LoRA strength that was used when applying
        dtype: Data type for computation
        
    Returns:
        Modified pipeline with LoRA removed
    """
    logger.info(f"Removing step distillation LoRA with strength={strength}")
    
    # Load LoRA weights
    lora_weights = load_step_distill_lora(lora_path, dtype)
    
    # Get transformer state dict
    transformer = pipeline.transformer
    weight_dict = dict(transformer.named_parameters())
    
    # Parse LoRA pairs and diffs (same logic as apply)
    lora_pairs = {}
    lora_diffs = {}
    
    for key in lora_weights.keys():
        # Remove diffusion_model. prefix if present
        if key.startswith("diffusion_model."):
            base_key = key[len("diffusion_model."):]
        else:
            base_key = key
        
        if key.endswith("lora_down.weight"):
            target_name = base_key.replace(".lora_down.weight", ".weight")
            pair_key = key.replace("lora_down.weight", "lora_up.weight")
            if pair_key in lora_weights:
                lora_pairs[target_name] = (key, pair_key)
        elif key.endswith("lora_A.weight"):
            target_name = base_key.replace(".lora_A.weight", ".weight")
            pair_key = key.replace("lora_A.weight", "lora_B.weight")
            if pair_key in lora_weights:
                lora_pairs[target_name] = (key, pair_key)
        elif key.endswith(".diff"):
            target_name = base_key.replace(".diff", ".weight")
            if target_name not in weight_dict:
                target_name = base_key.replace(".diff", "")
            lora_diffs[target_name] = key
        elif key.endswith(".diff_b"):
            target_name = base_key.replace(".diff_b", ".bias")
            lora_diffs[target_name] = key
        elif key.endswith(".diff_m"):
            target_name = base_key.replace(".diff_m", ".modulation")
            lora_diffs[target_name] = key
    
    # Remove LoRA weights (subtract instead of add)
    removed_count = 0
    
    with torch.no_grad():
        for target_name, (lora_A_key, lora_B_key) in lora_pairs.items():
            if target_name not in weight_dict:
                continue
            
            param = weight_dict[target_name]
            lora_A = lora_weights[lora_A_key].to(param.device, param.dtype)
            lora_B = lora_weights[lora_B_key].to(param.device, param.dtype)
            
            expected_shape = (lora_B.shape[0], lora_A.shape[1])
            
            if param.shape == expected_shape:
                param.data -= strength * torch.matmul(lora_B, lora_A)
                removed_count += 1
            elif len(param.shape) == 4 and param.shape[:2] == expected_shape[:2]:
                delta = torch.matmul(
                    lora_B.squeeze(-1).squeeze(-1) if len(lora_B.shape) == 4 else lora_B,
                    lora_A.squeeze(-1).squeeze(-1) if len(lora_A.shape) == 4 else lora_A
                )
                if len(param.shape) == 4:
                    delta = delta.unsqueeze(-1).unsqueeze(-1)
                param.data -= strength * delta
                removed_count += 1
        
        for target_name, diff_key in lora_diffs.items():
            if target_name not in weight_dict:
                continue
            
            param = weight_dict[target_name]
            diff_value = lora_weights[diff_key].to(param.device, param.dtype)
            
            if param.shape == diff_value.shape:
                param.data -= strength * diff_value
                removed_count += 1
    
    logger.info(f"Removed LoRA from {removed_count} layers")
    
    return pipeline
