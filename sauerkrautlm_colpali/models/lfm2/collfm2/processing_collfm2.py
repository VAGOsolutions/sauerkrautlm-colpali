from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoProcessor, BatchEncoding, BatchFeature

from sauerkrautlm_colpali.utils.processing_utils import BaseVisualRetrieverProcessor


class ColLFM2Processor(BaseVisualRetrieverProcessor):
    """
    Processor for ColLFM2, adapting LFM2-VL's processing for ColBERT-style retrieval.
    
    Key features of LFM2-VL processing:
    - ChatML-like template
    - Dynamic image tokens (64-256)
    - Native 512x512 resolution
    """
    
    # LFM2 specific tokens - using ColQwen2.5 style prompts
    image_token: ClassVar[str] = "<image>"
    # Use the EXACT same prompt structure as ColQwen2.5 (just with LFM2's image token)
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<image>Describe the image.<|im_end|>"
    )
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"  # Query augmentation for ColBERT-style retrieval
    
    def __init__(self, processor, max_image_tokens: int = 256, min_image_tokens: int = 64):
        """
        Initialize processor
        """
        self.processor = processor
        self.max_image_tokens = max_image_tokens
        self.min_image_tokens = min_image_tokens
        
        # Ensure left padding for ColBERT
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_image_tokens: Optional[int] = 256,
        min_image_tokens: Optional[int] = 64,
        **kwargs
    ):
        """Load processor from pretrained"""
        # Remove trust_remote_code if already in kwargs to avoid duplicate
        kwargs.pop("trust_remote_code", None)
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **kwargs
        )
        return cls(
            processor=processor,
            max_image_tokens=max_image_tokens,
            min_image_tokens=min_image_tokens
        )

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images using LFM2-VL's processor.
        Process individually then batch the results.
        """
        images = [image.convert("RGB") for image in images]

        # Process each image individually with its text prompt
        all_batches = []
        for image in images:
            # Process single image with single text prompt
            batch = self.processor(
                text=self.visual_prompt_prefix,
                images=image,
                return_tensors="pt",
                padding="longest",  # Don't force max_length - images have dynamic tokens
            )
            all_batches.append(batch)
        
        # Collate all batches - pad to longest in this batch
        if all_batches:
            # Find max length for each key
            from transformers import BatchFeature
            collated = {}
            
            # First, concatenate all tensors
            for key in all_batches[0].keys():
                tensors = [b[key] for b in all_batches]
                # Find max length for this batch
                max_len = max(t.shape[-1] if t.dim() > 1 else 1 for t in tensors)
                
                # Pad all to max_len
                padded = []
                for t in tensors:
                    if t.dim() > 1 and t.shape[-1] < max_len:
                        # Pad last dimension
                        pad_size = max_len - t.shape[-1]
                        t = torch.nn.functional.pad(t, (0, pad_size))
                    padded.append(t)
                
                collated[key] = torch.cat(padded, dim=0)
            batch = BatchFeature(collated)
        else:
            batch = BatchFeature({})
        
        return batch

    def process_texts(
        self,
        texts: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process query texts - for text-only queries
        """
        # For text queries, use direct tokenization
        # LFM2's tokenizer should be accessible
        tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        
        # Direct tokenization - simple text tokenization
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
            max_length=2048  # Set reasonable max_length for LFM2-VL
        )
        
        # Ensure it's a BatchFeature
        if not isinstance(batch, (BatchFeature, BatchEncoding)):
            batch = BatchFeature(batch)
            
        return batch

    def process_queries(
        self,
        queries: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """Alias for process_texts for compatibility"""
        return self.process_texts(queries)

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """MaxSim scoring for ColBERT-style retrieval"""
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 16,
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int]:
        """Calculate number of patches - LFM2 uses dynamic tiling, this is a placeholder"""
        # LFM2's tiling strategy is complex, this might not be directly applicable
        # For now, a placeholder based on common ViT patch sizes
        height, width = image_size
        n_patches_x = width // patch_size
        n_patches_y = height // patch_size
        return n_patches_x, n_patches_y

    @property
    def tokenizer(self):
        """Access tokenizer"""
        return self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor

    def __getattr__(self, name):
        """Forward to processor"""
        return getattr(self.processor, name)