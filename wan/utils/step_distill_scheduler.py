"""
Step Distillation Scheduler for FlashPortrait

This scheduler implements the step distillation inference logic,
allowing the model to generate videos in just 4 steps instead of 30+ steps.

The key idea is to use a pre-trained LoRA that learns to "skip" intermediate
denoising steps, directly predicting the clean video from specific noise levels.

Reference: LightX2V step distillation implementation LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py
"""

import math
import numpy as np
from typing import Union, List, Optional

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class StepDistillScheduler(SchedulerMixin, ConfigMixin):
    """
    Step Distillation Scheduler for accelerated inference.
    
    This scheduler works with step-distilled LoRA models to enable
    4-step inference instead of the standard 30+ steps.
    
    Key differences from standard Flow Matching scheduler:
    1. Uses fixed denoising step list (e.g., [1000, 750, 500, 250])
    2. Each step predicts a larger "jump" in the denoising process
    3. Works with distilled LoRA weights that learned these larger jumps
    
    Args:
        num_train_timesteps: Number of training timesteps (default: 1000)
        shift: Noise schedule shift parameter (default: 5.0 for 720p)
        denoising_step_list: List of timesteps to denoise at
    """
    
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        denoising_step_list: List[int] = None,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ):
        if denoising_step_list is None:
            # Default 4-step distillation schedule
            denoising_step_list = [1000, 750, 500, 250]
        
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.denoising_step_list = denoising_step_list
        self.infer_steps = len(denoising_step_list)
        
        self.sigma_max = 1.0
        self.sigma_min = 0.0
        
        # Initialize timesteps and sigmas
        self.timesteps = None
        self.sigmas = None
        self._step_index = None  # Use _step_index for compatibility with pipeline
        
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.base_shift = base_shift
        self.max_shift = max_shift
        
        # Required by diffusers pipeline
        self.order = 1  # First-order method
    
    @property
    def step_index(self):
        """The current step index."""
        return self._step_index
    
    @step_index.setter
    def step_index(self, value):
        self._step_index = value
        
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        mu: float = None,
    ):
        """
        Set the timesteps for step distillation.
        
        Note: num_inference_steps is ignored as we use the fixed denoising_step_list.
        """
        # For step distillation, we use fixed timesteps from denoising_step_list
        # num_inference_steps is overridden by len(denoising_step_list)
        self.infer_steps = len(self.denoising_step_list)
        
        # Calculate sigma schedule with shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        
        # Apply shift to sigmas
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        
        # Get timesteps from sigmas
        timesteps = sigmas * self.num_train_timesteps
        
        # Select only the timesteps in our denoising_step_list
        denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        
        self.timesteps = timesteps[denoising_step_index].to(device)
        self.sigmas = sigmas[denoising_step_index].to("cpu")
        
        # Add final sigma of 0
        self.sigmas = torch.cat([self.sigmas, torch.tensor([0.0])])
        
        self.step_index = 0
        
    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor = None) -> torch.Tensor:
        """
        Scale the model input (no scaling needed for flow matching).
        """
        return sample
    
    def _index_for_timestep(self, timestep: torch.Tensor) -> int:
        """
        Find the step index corresponding to the given timestep.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item() if timestep.numel() == 1 else timestep[0].item()
        
        # Find matching index in timesteps
        for i, ts in enumerate(self.timesteps):
            ts_val = ts.item() if isinstance(ts, torch.Tensor) else ts
            if abs(ts_val - timestep) < 1e-3:
                return i
        
        # If not found, find closest
        timesteps_np = self.timesteps.cpu().numpy() if isinstance(self.timesteps, torch.Tensor) else self.timesteps
        closest_idx = int(np.abs(timesteps_np - timestep).argmin())
        return closest_idx
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ):
        """
        Perform one step of the step distillation denoising process.
        
        The step distillation uses flow matching formulation:
        - model_output is the predicted flow (velocity)
        - We step from current sigma to next sigma
        
        Args:
            model_output: Predicted flow from the model
            timestep: Current timestep
            sample: Current noisy sample
            return_dict: Whether to return as dict
            
        Returns:
            Denoised sample for next step
        """
        # Determine step index from timestep if not set
        if self._step_index is None:
            self._step_index = self._index_for_timestep(timestep)
        
        current_step_index = self._step_index
        
        # Get current and next sigma
        sigma = self.sigmas[current_step_index].item()
        
        # Convert to float32 for precision
        flow_pred = model_output.to(torch.float32)
        latents = sample.to(torch.float32)
        
        # Flow matching step: x_0 = x_t - sigma * v
        # where v is the predicted flow/velocity
        denoised = latents - sigma * flow_pred
        
        # Add noise for next step if not the last step
        if current_step_index < self.infer_steps - 1:
            sigma_next = self.sigmas[current_step_index + 1].item()
            # x_{t-1} = x_0 + sigma_next * v
            sample_next = denoised + flow_pred * sigma_next
        else:
            sample_next = denoised
        
        # Increment step index
        self._step_index += 1
        
        # Convert back to original dtype
        sample_next = sample_next.to(model_output.dtype)
        
        if return_dict:
            from diffusers.schedulers.scheduling_utils import SchedulerOutput
            return SchedulerOutput(prev_sample=sample_next)
        
        return (sample_next,)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to the flow matching schedule.
        
        x_t = (1 - sigma) * x_0 + sigma * noise
        """
        # Get sigma from timestep
        sigmas = timesteps.float() / self.num_train_timesteps
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        
        # Reshape for broadcasting
        while len(sigmas.shape) < len(original_samples.shape):
            sigmas = sigmas.unsqueeze(-1)
        
        noisy_samples = (1 - sigmas) * original_samples + sigmas * noise
        return noisy_samples.type_as(noise)
    
    def __len__(self):
        return self.infer_steps


class StepDistillSchedulerWrapper:
    """
    Wrapper to make StepDistillScheduler compatible with existing pipeline code.
    
    This wrapper intercepts the scheduler calls and adapts them for step distillation.
    """
    
    def __init__(
        self,
        original_scheduler,
        denoising_step_list: List[int] = None,
        shift: float = 5.0,
    ):
        """
        Args:
            original_scheduler: The original scheduler from the pipeline
            denoising_step_list: List of timesteps for step distillation
            shift: Noise schedule shift parameter
        """
        self.original_scheduler = original_scheduler
        
        if denoising_step_list is None:
            denoising_step_list = [1000, 750, 500, 250]
        
        self.denoising_step_list = denoising_step_list
        self.shift = shift
        self.infer_steps = len(denoising_step_list)
        
        # Initialize step distillation scheduler
        self.distill_scheduler = StepDistillScheduler(
            num_train_timesteps=1000,
            shift=shift,
            denoising_step_list=denoising_step_list,
        )
        
        self._use_distill = True
        
    def enable_distill(self):
        """Enable step distillation mode."""
        self._use_distill = True
        
    def disable_distill(self):
        """Disable step distillation mode (use original scheduler)."""
        self._use_distill = False
    
    def set_timesteps(self, num_inference_steps: int, device=None, **kwargs):
        """Set timesteps - routes to distill or original scheduler."""
        if self._use_distill:
            self.distill_scheduler.set_timesteps(num_inference_steps, device, **kwargs)
            self.timesteps = self.distill_scheduler.timesteps
            self.sigmas = self.distill_scheduler.sigmas
        else:
            self.original_scheduler.set_timesteps(num_inference_steps, device, **kwargs)
            self.timesteps = self.original_scheduler.timesteps
            if hasattr(self.original_scheduler, 'sigmas'):
                self.sigmas = self.original_scheduler.sigmas
    
    def step(self, *args, **kwargs):
        """Perform scheduler step."""
        if self._use_distill:
            return self.distill_scheduler.step(*args, **kwargs)
        else:
            return self.original_scheduler.step(*args, **kwargs)
    
    def scale_model_input(self, *args, **kwargs):
        """Scale model input."""
        if self._use_distill:
            return self.distill_scheduler.scale_model_input(*args, **kwargs)
        else:
            return self.original_scheduler.scale_model_input(*args, **kwargs)
    
    def add_noise(self, *args, **kwargs):
        """Add noise to samples."""
        if self._use_distill:
            return self.distill_scheduler.add_noise(*args, **kwargs)
        else:
            return self.original_scheduler.add_noise(*args, **kwargs)
    
    def __len__(self):
        if self._use_distill:
            return len(self.distill_scheduler)
        else:
            return len(self.original_scheduler)
    
    def __getattr__(self, name):
        """Forward attribute access to appropriate scheduler."""
        if name.startswith('_') or name in ['original_scheduler', 'distill_scheduler', 
                                              'denoising_step_list', 'shift', 'infer_steps',
                                              'enable_distill', 'disable_distill', 'set_timesteps',
                                              'step', 'scale_model_input', 'add_noise', 'timesteps', 'sigmas']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if self._use_distill:
            return getattr(self.distill_scheduler, name)
        else:
            return getattr(self.original_scheduler, name)
