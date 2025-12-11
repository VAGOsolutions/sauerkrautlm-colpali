# SauerkrautLM-ColPali: Multi-vector vision retrieval models
# Fork of colpali-engine with additional model architectures

__all__ = []

# Import available models with try/except for graceful fallback

# Qwen2.5 models
try:
    from .models import BiQwen2_5, BiQwen2_5_Processor, ColQwen2_5, ColQwen2_5_Processor
    __all__.extend(["BiQwen2_5", "BiQwen2_5_Processor", "ColQwen2_5", "ColQwen2_5_Processor"])
except ImportError:
    pass

# Qwen3 models
try:
    from .models import ColQwen3, ColQwen3Processor
    __all__.extend(["ColQwen3", "ColQwen3Processor"])
except ImportError:
    pass

# LFM2 models
try:
    from .models import ColLFM2, ColLFM2Processor
    __all__.extend(["ColLFM2", "ColLFM2Processor"])
except ImportError:
    pass

# Ministral3 models
try:
    from .models import ColMinistral3, ColMinistral3Processor
    __all__.extend(["ColMinistral3", "ColMinistral3Processor"])
except ImportError:
    pass
