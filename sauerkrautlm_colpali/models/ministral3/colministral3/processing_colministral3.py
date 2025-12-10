"""
ColMinistral3Processor: Processing utilities for ColMinistral3

Uses PixtralProcessor for image processing (same as Ministral-3).
"""

import torch
from typing import ClassVar, List, Union, Optional, Tuple
from PIL import Image

from transformers import AutoProcessor, BatchEncoding, BatchFeature
from transformers.utils import logging

from sauerkrautlm_colpali.utils.processing_utils import BaseVisualRetrieverProcessor


logger = logging.get_logger(__name__)


class ColMinistral3Processor(BaseVisualRetrieverProcessor):
    """
    Processor for ColMinistral3 - wraps Mistral3/Pixtral processor.
    
    Ministral-3 uses PixtralProcessor with chat templates.
    """
    
    image_token: ClassVar[str] = "[IMG]"
    visual_prompt_prefix: ClassVar[str] = ""  # Will use chat template instead
    query_augmentation_token: ClassVar[str] = "</s>"  # Ministral EOS token
    
    def __init__(self, tokenizer, image_processor):
        """Initialize with tokenizer and image_processor"""
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # Set left padding
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ):
        """Load processor from pretrained model."""
        kwargs.pop("trust_remote_code", None)
        
        from pathlib import Path
        import json
        
        # Load the processor
        try:
            from transformers import PixtralProcessor
            processor = PixtralProcessor.from_pretrained(pretrained_model_name_or_path)
        except:
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=True
            )
        
        # Check for local processor_config.json with custom settings
        local_path = Path(pretrained_model_name_or_path)
        if local_path.exists():
            processor_config_path = local_path / "processor_config.json"
            if processor_config_path.exists():
                with open(processor_config_path) as f:
                    config = json.load(f)
                # Apply image processor settings if present
                if "image_processor" in config and hasattr(processor, "image_processor"):
                    img_config = config["image_processor"]
                    if "size" in img_config and hasattr(processor.image_processor, "size"):
                        processor.image_processor.size = img_config["size"]
                        print(f"   ✅ Applied processor_config: size={img_config['size']}")
        
        # Wrap it
        instance = cls(tokenizer=processor.tokenizer, image_processor=processor.image_processor)
        instance._base_processor = processor
        return instance
    
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """Process images - optimized batch processing"""
        images = [image.convert("RGB") for image in images]
        
        # Use the base processor if available
        if hasattr(self, '_base_processor') and self._base_processor is not None:
            # Chat template for image
            messages = [{"role": "user", "content": [{"type": "image"}]}]
            text = self._base_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Process ALL images at once
            batch = self._base_processor(
                text=[text] * len(images),
                images=images,
                return_tensors="pt",
                padding=True,
            )
        else:
            # Fallback
            image_inputs = self.image_processor(images, return_tensors="pt")
            messages = [{"role": "user", "content": [{"type": "image"}]}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text_inputs = self.tokenizer([text] * len(images), return_tensors="pt", padding="longest")
            batch = BatchFeature({**image_inputs, **text_inputs})
        
        # Convert to bfloat16
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        
        batch.pop("token_type_ids", None)
        return batch
    
    def process_texts(
        self,
        texts: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """Process queries - text only"""
        # Add augmentation tokens
        texts_aug = [text + self.query_augmentation_token * 10 for text in texts]
        
        # Tokenize
        batch = self.tokenizer(
            texts_aug,
            return_tensors="pt",
            padding="longest",
        )
        
        batch.pop("token_type_ids", None)
        return batch
    
    def process_queries(self, queries: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """Alias for process_texts"""
        return self.process_texts(queries)
    
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """MaxSim scoring"""
        return self.score_multi_vector(qs, ps, device=device, **kwargs)
    
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 14,
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int]:
        """Calculate patches"""
        height, width = image_size
        n_patches_x = width // patch_size
        n_patches_y = height // patch_size
        return n_patches_x, n_patches_y
    
    def save_pretrained(self, save_directory: str):
        """Save processor components"""
        self.tokenizer.save_pretrained(save_directory)
        self.image_processor.save_pretrained(save_directory)


__all__ = ["ColMinistral3Processor"]

