# Import only available models
__all__ = []

from .models import (
    BiPali,
    BiPaliProj,
    ColIdefics3,
    ColIdefics3Processor,
    ColPali,
    ColPaliProcessor,
)

__all__.extend([
    "BiPali",
    "BiPaliProj", 
    "ColIdefics3",
    "ColIdefics3Processor",
    "ColPali",
    "ColPaliProcessor",
])

# Try to import Qwen models if available
try:
    from .models import (
        BiQwen2,
        BiQwen2Processor,
        ColQwen2,
        ColQwen2Processor,
    )
    __all__.extend(["BiQwen2", "BiQwen2Processor", "ColQwen2", "ColQwen2Processor"])
except ImportError:
    pass

try:
    from .models import (
        BiQwen2_5,
        BiQwen2_5_Processor,
        ColQwen2_5,
        ColQwen2_5_Processor,
    )
    __all__.extend(["BiQwen2_5", "BiQwen2_5_Processor", "ColQwen2_5", "ColQwen2_5_Processor"])
except ImportError:
    pass

try:
    from .models import (
        ColQwen2_5Omni,
        ColQwen2_5OmniProcessor,
    )
    __all__.extend(["ColQwen2_5Omni", "ColQwen2_5OmniProcessor"])
except ImportError:
    pass

try:
    from .models import (
        BiQwen3,
        BiQwen3Processor,
        ColQwen3,
        ColQwen3Processor,
    )
    __all__.extend(["BiQwen3", "BiQwen3Processor", "ColQwen3", "ColQwen3Processor"])
except ImportError:
    pass

try:
    from .models import (
        ColLightOnOCR,
        ColLightOnOCRProcessor,
    )
    __all__.extend(["ColLightOnOCR", "ColLightOnOCRProcessor"])
except ImportError:
    pass
