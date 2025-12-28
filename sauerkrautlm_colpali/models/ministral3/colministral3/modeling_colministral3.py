"""
ColMinistral3: Late Interaction Model für Ministral-3 (3B/8B/14B)

Based on Mistral3ForConditionalGeneration with Pixtral Vision Encoder.
Similar architecture to ColLightOnOCR.
"""

import torch
import torch.nn as nn
from typing import ClassVar, List, Optional
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel
from transformers.utils import logging

from sauerkrautlm_colpali.models.ministral3.colministral3.processing_colministral3 import ColMinistral3Processor
from sauerkrautlm_colpali.utils.torch_utils import get_torch_device


logger = logging.get_logger(__name__)


class ColMinistral3(nn.Module):
    """
    ColMinistral3 model implementation for late interaction retrieval.
    
    This model wraps the Ministral-3 architecture and adds a custom
    text projection layer for generating embeddings suitable for late interaction.
    
    Architecture:
    - Base: Mistral3ForConditionalGeneration (Ministral-3 language model + Pixtral vision encoder)
    - Custom text projection: Linear layer to project hidden states to 128 dimensions
    - Supports LoRA fine-tuning
    
    Similar to ColLightOnOCR but for Ministral-3 models.
    """
    
    main_input_name: ClassVar[str] = "doc_input_ids"  # For PeftModel compatibility
    
    def __init__(
        self,
        config: str | PreTrainedModel = "mistralai/Ministral-3-3B-Instruct-2512",
        dim: int = 128,
        mask_non_image_embeddings: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize ColMinistral3 model.
        
        Args:
            config: Path or name of the base Ministral-3 model, or a PreTrainedModel instance
            dim: Dimension of the output embeddings (default: 128)
            mask_non_image_embeddings: Whether to mask non-image embeddings (default: False)
            torch_dtype: Data type for model weights (default: torch.bfloat16)
        """
        super().__init__()
        
        self._torch_dtype = torch_dtype
        
        # Load base model
        if isinstance(config, str):
            try:
                from transformers import Mistral3ForConditionalGeneration
                self.base_model = Mistral3ForConditionalGeneration.from_pretrained(
                    config,
                    torch_dtype=torch_dtype,
                )
            except ImportError:
                # Fallback to Auto
                self.base_model = AutoModelForImageTextToText.from_pretrained(
                    config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
        else:
            # config is already a loaded model
            self.base_model = config
        
        self.dim = dim
        self.mask_non_image_embeddings = mask_non_image_embeddings
        
        # Get hidden size from config
        # Ministral-3 uses text_config for hidden_size
        if hasattr(self.base_model, 'config'):
            model_config = self.base_model.config
        else:
            model_config = self.base_model
        
        hidden_size = model_config.text_config.hidden_size  # 3072 for 3B
        
        # Custom text projection layer
        # Projects from hidden_size to dim (128)
        self.custom_text_proj = nn.Linear(hidden_size, dim, bias=False)
        
        # Initialize with small random weights
        with torch.no_grad():
            nn.init.normal_(self.custom_text_proj.weight, mean=0.0, std=0.02)
        
        # Convert to specified dtype to match base model
        self.custom_text_proj = self.custom_text_proj.to(torch_dtype)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for ColMinistral3.
        
        Takes all arguments and passes them to base model.
        """
        # Save attention_mask before popping anything
        attention_mask = kwargs.get("attention_mask", None)
        
        # Remove arguments that shouldn't go to base model forward
        kwargs.pop("return_dict", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("output_hidden_states", None)
        
        # Call base model forward with hidden states
        outputs = self.base_model(
            *args,
            **kwargs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden state
        last_hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        
        # Project to embedding dimension
        proj = self.custom_text_proj(last_hidden_states)  # [batch_size, seq_len, dim]
        
        # L2 normalization - ESSENTIAL for ColBERT-style retrieval!
        # Add epsilon to avoid division by zero
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Apply attention mask AFTER normalization
        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)
        
        return proj
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "ColMinistral3":
        """
        Load a pretrained ColMinistral3 model.
        
        Args:
            pretrained_model_name_or_path: Path to model directory or HuggingFace model ID
            torch_dtype: Data type for model weights (default: torch.bfloat16)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded ColMinistral3 model
        """
        from pathlib import Path
        
        model_path = Path(pretrained_model_name_or_path)
        
        # Check if it's a local ColMinistral3 model
        if model_path.exists() and (model_path / "model.safetensors").exists() and (model_path / "config.json").exists():
            # Load config to check for ColMinistral3 fields
            import json
            with open(model_path / "config.json") as f:
                config_dict = json.load(f)
            
            # Check if it's a ColMinistral3 model (has "dim" and "base_model" fields)
            if "dim" in config_dict and "base_model" in config_dict:
                print(f"   Loading ColMinistral3 from {pretrained_model_name_or_path}...")
                
                # Get ColMinistral3-specific params
                dim = config_dict.get("dim", 128)
                mask_non_image = config_dict.get("mask_non_image_embeddings", False)
                
                # Load base model
                try:
                    from transformers import Mistral3ForConditionalGeneration
                    base_model = Mistral3ForConditionalGeneration.from_pretrained(
                        str(model_path),
                        torch_dtype=torch_dtype,
                        device_map=None,
                        local_files_only=True,
                    )
                    print(f"   ✅ Base model loaded with Mistral3ForConditionalGeneration")
                except ImportError:
                    from transformers import AutoModelForImageTextToText
                    base_model = AutoModelForImageTextToText.from_pretrained(
                        str(model_path),
                        torch_dtype=torch_dtype,
                        device_map=None,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    print(f"   ✅ Base model loaded with AutoModelForImageTextToText")
                
                # Create our wrapper
                model = cls(config=base_model, dim=dim, mask_non_image_embeddings=mask_non_image)
                print(f"   ✅ ColMinistral3 wrapper created")
                
                # CRITICAL: Load custom_text_proj weights from safetensors!
                from safetensors.torch import load_file
                safetensors_path = model_path / "model.safetensors"
                if safetensors_path.exists():
                    state_dict = load_file(str(safetensors_path))
                    if "custom_text_proj.weight" in state_dict:
                        model.custom_text_proj.weight.data = state_dict["custom_text_proj.weight"].to(model.custom_text_proj.weight.dtype)
                        print(f"   ✅ Loaded custom_text_proj weights from safetensors")
                    else:
                        print(f"   ⚠️ custom_text_proj.weight not found in safetensors - using initialized weights")
                
                return model
        
        # Otherwise, load as base model
        return cls(config=pretrained_model_name_or_path, **kwargs)
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        from pathlib import Path
        from safetensors.torch import save_file
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        state_dict = self.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                save_dict[k] = v.detach().clone()
            else:
                save_dict[k] = v
        
        save_file(save_dict, save_path / 'model.safetensors')
    
    def get_output_embeddings(self):
        """Get output embeddings (required for PeftModel)."""
        return self.base_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (required for PeftModel)."""
        self.base_model.set_output_embeddings(new_embeddings)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing (required for training with PEFT)."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs is not None:
                self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


# Export for convenience
__all__ = ["ColMinistral3", "ColMinistral3Processor"]

