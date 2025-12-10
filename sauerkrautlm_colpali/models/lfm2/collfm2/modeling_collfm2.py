"""
ColLFM2 - ColPali-style adaptation of LFM2-VL models
"""
from typing import ClassVar, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoConfig


class ColLFM2(nn.Module):
    """
    ColLFM2 model - adapting LFM2-VL for multi-vector document retrieval
    
    Based on LiquidAI's LFM2-VL-450M with:
    - SigLIP2 vision encoder (86M params)
    - Hybrid conv+attention backbone
    - Dynamic token mapping
    - Native 512x512 resolution
    """
    
    main_input_name: ClassVar[str] = "input_ids"
    
    def __init__(self, config=None, mask_non_image_embeddings: bool = False, device_map=None, **kwargs):
        super().__init__()
        
        # Determine model name
        if isinstance(config, str):
            model_name = config
        elif config is None:
            model_name = "LiquidAI/LFM2-VL-450M"
        else:
            # config is already a config object
            model_name = getattr(config, "_name_or_path", "LiquidAI/LFM2-VL-450M")
            
        # Load base LFM2-VL model MIT device_map!
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map=device_map  # WICHTIG: device_map weitergeben!
        )
        
        # Get config from loaded model
        if isinstance(config, str) or config is None:
            config = self.base_model.config
        
        # Get hidden size from config
        if hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        elif hasattr(config, 'd_model'):
            hidden_size = config.d_model
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            hidden_size = config.text_config.hidden_size
        else:
            # Default for LFM2-VL-450M
            hidden_size = 1024
            
        # ColPali projection layer
        self.dim = 128
        self.custom_text_proj = nn.Linear(hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        
        # Store config
        self.config = config
        
        # Initialize projection
        self._init_weights()
        
        # Convert to bfloat16 to match base model
        self.custom_text_proj = self.custom_text_proj.to(torch.bfloat16)
        
        # Move to device if device_map specified
        if device_map is not None:
            self.custom_text_proj = self.custom_text_proj.to(device_map)
        
    def _init_weights(self):
        """Initialize weights following ColPali"""
        with torch.no_grad():
            nn.init.normal_(self.custom_text_proj.weight, std=0.02)
            if self.custom_text_proj.bias is not None:
                nn.init.zeros_(self.custom_text_proj.bias)
                
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load pretrained model"""
        mask_non_image = kwargs.pop("mask_non_image_embeddings", False)
        
        import os
        import json
        
        # Check if it's a HuggingFace model (not local path)
        if not os.path.exists(pretrained_model_name_or_path):
            # It's a HF model - download it first!
            from huggingface_hub import snapshot_download
            
            print(f"   Downloading from HuggingFace: {pretrained_model_name_or_path}...")
            try:
                local_path = snapshot_download(
                    pretrained_model_name_or_path,
                    allow_patterns=["*.json", "*.safetensors", "*.pt", "*.model", "*.txt"]
                )
                pretrained_model_name_or_path = local_path
                print(f"   ✅ Downloaded to {local_path}")
            except Exception as e:
                print(f"   ⚠️ Download failed: {e}")
                # Continue with original path and hope it works
        
        # Check if this is a saved ColLFM2 model or base LFM2
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # If it's our ColLFM2 model, load the base model from config
            if config.get("model_type") == "collfm2":
                base_model = config.get("base_model", "LiquidAI/LFM2-VL-450M")
                
                # Create model with base LFM2
                model = cls(
                    config=base_model,
                    mask_non_image_embeddings=mask_non_image
                )
                
                # Load FULL model from model.safetensors (wie ColQwen3!)
                safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
                collfm2_projection_path = os.path.join(pretrained_model_name_or_path, "collfm2_projection.pt")
                
                if os.path.exists(safetensors_path):
                    # Load COMPLETE model from model.safetensors (base + custom_text_proj)
                    print(f"   Loading FULL model from model.safetensors...")
                    from safetensors.torch import load_file
                    state_dict = load_file(safetensors_path)
                    
                    # Map keys: model.* → base_model.model.*, custom_text_proj.* bleibt
                    fixed_state = {}
                    for k, v in state_dict.items():
                        if 'custom_text_proj' in k:
                            # custom_text_proj.* bleibt wie es ist
                            if k.startswith('model.custom_text_proj.'):
                                new_key = k.replace('model.custom_text_proj.', 'custom_text_proj.')
                            elif k.startswith('base_model.model.custom_text_proj.'):
                                new_key = k.replace('base_model.model.custom_text_proj.', 'custom_text_proj.')
                            else:
                                new_key = k  # custom_text_proj.* ohne prefix
                            fixed_state[new_key] = v
                        elif not k.startswith('base_model.'):
                            # model.* → base_model.model.*
                            new_key = 'base_model.' + k
                            fixed_state[new_key] = v
                        else:
                            # Schon base_model.* prefix
                            fixed_state[k] = v
                    
                    # Load mit gemappten keys
                    missing, unexpected = model.load_state_dict(fixed_state, strict=False)
                    print(f"   ✅ Loaded {len(state_dict)} weights from model.safetensors")
                    if missing and len(missing) > 1:  # lm_head ist OK
                        print(f"      ⚠️ Missing: {len(missing)} keys: {list(missing)[:3]}")
                    if unexpected:
                        print(f"      ⚠️ Unexpected: {len(unexpected)} keys: {list(unexpected)[:3]}")
                        
                elif os.path.exists(collfm2_projection_path):
                    # Fallback: Load from collfm2_projection.pt (nur custom_text_proj)
                    print(f"   Loading custom_text_proj from collfm2_projection.pt...")
                    state_dict = torch.load(collfm2_projection_path, map_location="cpu")
                    model.custom_text_proj.load_state_dict(state_dict['custom_text_proj'])
                    print(f"   ✅ Loaded custom_text_proj from collfm2_projection.pt")
                else:
                    print(f"   ⚠️ No weights found, using random initialization")
            else:
                # It's a base LFM2 model
                model = cls(
                    config=pretrained_model_name_or_path,
                    mask_non_image_embeddings=mask_non_image
                )
        else:
            # Direct model name (e.g., "LiquidAI/LFM2-VL-450M")
            model = cls(
                config=pretrained_model_name_or_path,
                mask_non_image_embeddings=mask_non_image
            )
                    
        return model
    
    def forward(self, **kwargs):
        """
        Forward pass for multi-vector embeddings
        """
        # Remove conflicting kwargs
        output_hidden_states = kwargs.pop("output_hidden_states", True)
        return_dict = kwargs.pop("return_dict", True)
        kwargs.pop("use_cache", None)
        
        # CRITICAL FIX: For LFM2, we need to ensure output_hidden_states is passed correctly
        # Set it on the config if needed
        if hasattr(self.base_model.model.config, 'output_hidden_states'):
            original_output_hidden_states = self.base_model.model.config.output_hidden_states
            self.base_model.model.config.output_hidden_states = True
        
        # Call the inner model's forward with output_hidden_states=True
        outputs = self.base_model.model(
            **kwargs,
            output_hidden_states=True,
            return_dict=return_dict
        )
        
        # Restore original config
        if hasattr(self.base_model.model.config, 'output_hidden_states'):
            self.base_model.model.config.output_hidden_states = original_output_hidden_states
        
        # Extract hidden states from inner model
        # If still None, try to use last output
        if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            last_hidden_states = outputs.hidden_states[-1]
        else:
            # Fallback: LFM2 might not support hidden_states, try to get from embeddings or logits
            raise ValueError(f"LFM2 hidden_states is None. Available keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else type(outputs)}")
        
        # Project to ColPali dimension
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, seq_len, dim)
        
        # L2 normalization - EXACTLY like ColQwen2.5 - NO EPSILON!
        proj = proj / proj.norm(dim=-1, keepdim=True)
        
        # Apply attention mask if present
        if "attention_mask" in kwargs:
            proj = proj * kwargs["attention_mask"].unsqueeze(-1)
            
        # Optionally mask non-image embeddings
        if "input_ids" in kwargs and self.mask_non_image_embeddings:
            # LFM2 uses <image> sentinel token
            # We need to identify image tokens (this is model-specific)
            pass  # TODO: Implement if needed
            
        return proj
    
    def save_pretrained(self, save_directory):
        """Save model"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(save_directory)
        
        # Save custom projection
        torch.save({
            'custom_text_proj': self.custom_text_proj.state_dict(),
            'config': {
                'dim': self.dim,
                'mask_non_image_embeddings': self.mask_non_image_embeddings
            }
        }, os.path.join(save_directory, 'collfm2_projection.pt'))
        
    @property
    def device(self):
        return next(self.parameters()).device
