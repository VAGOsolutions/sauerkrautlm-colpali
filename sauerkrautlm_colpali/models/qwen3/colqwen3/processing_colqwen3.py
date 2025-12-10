from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers import Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from sauerkrautlm_colpali.utils.processing_utils import BaseVisualRetrieverProcessor


class ColQwen3Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):
    """
    Processor for ColQwen3 - using Qwen2VLProcessor which handles images correctly
    """
    
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    # Use same augmentation token as ColQwen2.5 - this is CRITICAL for training!
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"
    image_token_id: ClassVar[int] = 151655
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"
    
    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs
        )
        
        if "max_num_visual_tokens" in kwargs:
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * 28 * 28
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels
        
        return instance
    
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """Process images - EXACTLY like ColQwen2.5"""
        images = [image.convert("RGB") for image in images]
        
        # Use self directly since we inherit from Qwen2VLProcessor
        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        
        # CRITICAL: Remove token_type_ids as per Qwen3VL documentation
        batch_doc.pop("token_type_ids", None)
        
        # CRITICAL FIX: Apply same pixel_values padding as ColQwen2.5!
        # This ensures consistent behavior with DDP on multiple GPUs
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]  # (batch_size,)
        
        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]
        
        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)
        
        return batch_doc
    
    def process_texts(
        self, 
        texts: List[str]
    ) -> Union[BatchFeature, BatchEncoding]:
        """Process texts - EXACTLY like ColQwen2.5"""
        batch = self(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )
        
        # CRITICAL: Remove token_type_ids as per Qwen3VL documentation
        batch.pop("token_type_ids", None)
        
        return batch
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.score_multi_vector(qs, ps, device=device, **kwargs)
    
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.
        
        EXACTLY like ColQwen2.5 - using smart_resize!
        """
        patch_size = self.image_processor.patch_size
        
        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.image_processor.merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )
        
        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size
        
        return n_patches_x, n_patches_y
    
    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
