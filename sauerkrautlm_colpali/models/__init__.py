try:
    from .qwen2_5 import BiQwen2_5, BiQwen2_5_Processor, ColQwen2_5, ColQwen2_5_Processor
except ImportError:
    pass  # Qwen2.5 not available

try:
    from .qwen3 import ColQwen3, ColQwen3Processor
except ImportError:
    pass  # Qwen3 not available

try:
    from .qwen_omni import ColQwen2_5Omni, ColQwen2_5OmniProcessor
except ImportError:
    pass  # Qwen Omni not available


try:
    from .ministral3 import ColMinistral3, ColMinistral3Processor
except ImportError:
    pass  # Ministral3 not available
