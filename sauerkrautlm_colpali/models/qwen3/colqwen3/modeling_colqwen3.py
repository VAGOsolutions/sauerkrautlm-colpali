from typing import ClassVar

import torch
from torch import nn
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel



class ColQwen3(Qwen3VLModel):
    """
    ColQwen3 model implementation following ColPali paper architecture.
    
    Key insights from ColPali paper:
    1. Use ALL tokens (text + vision) as multi-vector embeddings
    2. Vision tokens carry visual information for retrieval
    3. Handle DeepStack features and image_grid_thw properly
    4. Use late interaction (MaxSim) for scoring
    
    Inherits from Qwen3VLModel to get proper vision processing with DeepStack.
    """
    
    main_input_name: ClassVar[str] = "doc_input_ids"
    
    def __init__(self, config: Qwen3VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        
        # Read dim from config if available, otherwise default to 128
        # This allows loading models with different projection dimensions (e.g., 512)
        self.dim = getattr(config, 'dim', 128)
        
        # Qwen3-VL stores hidden_size in text_config, Qwen2.5-VL in root config
        # But both should work with config.hidden_size after super().__init__
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        # EXACTLY like ColQwen2.5 - just call post_init()
        self.post_init()
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load pretrained model.
        
        CRITICAL: Immer config als Qwen3VLConfig laden, auch bei HF Models!
        """
        from transformers import AutoConfig
        from pathlib import Path
        import json
        
        # Check if config is already provided
        if 'config' not in kwargs:
            # Load config manually (lokal ODER HF!)
            try:
                if Path(pretrained_model_name_or_path).is_dir():
                    # Lokaler Pfad
                    config_path = Path(pretrained_model_name_or_path) / "config.json"
                    with open(config_path) as f:
                        config_dict = json.load(f)
                else:
                    # HF Model - download config
                    from huggingface_hub import hf_hub_download
                    config_file = hf_hub_download(pretrained_model_name_or_path, "config.json")
                    with open(config_file) as f:
                        config_dict = json.load(f)
                
                # Force model_type to qwen3_vl für korrektes Parsing
                original_model_type = config_dict.get("model_type")
                config_dict["model_type"] = "qwen3_vl"
                
                # Create Qwen3VLConfig from dict (parsed rope_scaling korrekt!)
                config = Qwen3VLConfig.from_dict(config_dict)
                
                # Restore original model_type
                config.model_type = original_model_type if original_model_type else "colqwen3"
                
                # CRITICAL: Read dim from config.json if present (for upgraded projection models)
                if "dim" in config_dict:
                    config.dim = config_dict["dim"]
                    print(f"   📐 Using custom projection dim from config: {config.dim}")
                
                # Add config to kwargs
                kwargs['config'] = config
            except Exception as e:
                print(f"⚠️ Could not pre-load config: {e}")
                # Fallback to default loading
                pass
        
        # Load with config (if successfully parsed) or default
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass following ColPali paper architecture.
        
        MUST handle pixel_values EXACTLY like ColQwen2.5!
        """
        
        # CRITICAL FIX: Handle pixel_values EXACTLY like ColQwen2.5
        # The processor pads them, we must unpad them here!
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )
        
        # Remove arguments that shouldn't go to parent forward
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        
        # CRITICAL: Call the parent forward to get full model output with hidden states
        # Qwen3VLModel.forward returns Qwen3VLCausalLMOutputWithPast which has hidden_states
        output = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
        )
        
        # Extract hidden states from the output object
        last_hidden_states = output.hidden_states[-1] if hasattr(output, 'hidden_states') else output.last_hidden_state
        
        # Project to ColPali dimension
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)
        
        # L2 normalization - ESSENTIAL for ColBERT-style retrieval!
        proj = proj / proj.norm(dim=-1, keepdim=True)  # EXACTLY like ColQwen2.5 - NO epsilon!
        
        # Apply attention mask AFTER normalization (EXACTLY like ColQwen2.5)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        
        # Optionally mask non-image embeddings
        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Qwen3-VL specific image token ID (from documentation)
            image_token_id = getattr(self.config, 'image_token_id', 151655)
            # CRITICAL: Ensure input_ids is on same device (for Multi-GPU)
            input_ids = kwargs["input_ids"].to(proj.device)
            image_mask = (input_ids == image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        
        return proj
    
    @property
    def patch_size(self) -> int:
        """Get patch size - following ColQwen2.5 pattern"""
        # Qwen3-VL might structure this differently
        if hasattr(self, 'visual') and hasattr(self.visual, 'config'):
            return getattr(self.visual.config, 'patch_size', 16)
        return 16  # Default for Qwen3-VL
    
    @property
    def spatial_merge_size(self) -> int:
        """Get spatial merge size - following ColQwen2.5 pattern"""
        if hasattr(self, 'visual') and hasattr(self.visual, 'config'):
            return getattr(self.visual.config, 'spatial_merge_size', 2)
        return 2  # Default for Qwen3-VL
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(super(), 'gradient_checkpointing_enable'):
            super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(super(), 'gradient_checkpointing_disable'):
            super().gradient_checkpointing_disable()