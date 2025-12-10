"""
SauerkrautLM Visual Document Retrieval - MTEB Integration

This package provides MTEB model wrappers and metadata for SauerkrautLM ColPali-style models:
- ColQwen3 (Qwen3-VL backbone)
- ColLFM2 (LFM2 backbone)
- ColMinistral3 (Ministral3 backbone)

Usage in MTEB:
    import mteb
    model = mteb.get_model("VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1")
"""

from .slm_models import (
    # Base
    SLMBaseWrapper,
    # ColQwen3
    SLMColQwen3Wrapper,
    slm_colqwen3_loader,
    slm_colqwen3_1_7b_turbo,
    slm_colqwen3_2b,
    slm_colqwen3_4b,
    slm_colqwen3_8b,
    # ColLFM2
    SLMColLFM2Wrapper,
    slm_collfm2_loader,
    slm_collfm2_450m,
    # ColMinistral3
    SLMColMinistral3Wrapper,
    slm_colministral3_loader,
    slm_colministral3_3b,
    # Citations & Languages
    SAUERKRAUTLM_CITATION,
    COLPALI_CITATION,
    SUPPORTED_LANGUAGES,
)

__all__ = [
    # Base
    "SLMBaseWrapper",
    # ColQwen3
    "SLMColQwen3Wrapper",
    "slm_colqwen3_loader",
    "slm_colqwen3_1_7b_turbo",
    "slm_colqwen3_2b",
    "slm_colqwen3_4b",
    "slm_colqwen3_8b",
    # ColLFM2
    "SLMColLFM2Wrapper",
    "slm_collfm2_loader",
    "slm_collfm2_450m",
    # ColMinistral3
    "SLMColMinistral3Wrapper",
    "slm_colministral3_loader",
    "slm_colministral3_3b",
    # Citations & Languages
    "SAUERKRAUTLM_CITATION",
    "COLPALI_CITATION",
    "SUPPORTED_LANGUAGES",
]
