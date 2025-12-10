from .modeling_colqwen3 import ColQwen3
from .processing_colqwen3 import ColQwen3Processor

# Try importing MTEB model if available
try:
    from .mteb_model import ColQwen3ForMTEB, load_colqwen3_for_mteb
    MTEB_AVAILABLE = True
except ImportError:
    MTEB_AVAILABLE = False
    ColQwen3ForMTEB = None
    load_colqwen3_for_mteb = None

__all__ = ["ColQwen3", "ColQwen3Processor"]

if MTEB_AVAILABLE:
    __all__.extend(["ColQwen3ForMTEB", "load_colqwen3_for_mteb"])