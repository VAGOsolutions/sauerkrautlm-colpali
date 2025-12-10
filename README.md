# SauerkrautLM-ColPali: Multi-Vector Vision Retrieval Models

[![GitHub](https://img.shields.io/badge/VAGO_Solutions-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/VAGOsolutions)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/VAGOsolutions)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

---

> **Fork Notice**: This repository is a fork of [colpali-engine](https://github.com/illuin-tech/colpali) by Illuin Technology. We extend the original codebase with additional model architectures for document retrieval using vision language models.

## Overview

**SauerkrautLM-ColPali** provides model implementations and processors for multi-vector vision retrieval based on the ColPali architecture. This package includes support for several VLM backbones:

- **ColQwen3** - Based on Qwen3-VL (2B, 4B, 8B)
- **ColLFM2** - Based on LargeFlamingoModel 2 (~450M parameters)
- **ColMinistral3** - Based on Ministral-3B-Instruct with Pixtral vision encoder

## Models

| Model | Parameters | VRAM (bf16) | Max Tokens | Base Model | License |
|-------|------------|-------------|------------|------------|---------|
| [SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1) | 1.7B | ~3.4 GB | 262K | Qwen3-VL-1.7B | Apache 2.0 |
| [SauerkrautLM-ColQwen3-2b-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1) | 2.2B | ~4.4 GB | 262K | Qwen3-VL-2B | Apache 2.0 |
| [SauerkrautLM-ColQwen3-4b-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1) | 4B | ~8 GB | 262K | Qwen3-VL-4B | Apache 2.0 |
| [SauerkrautLM-ColQwen3-8b-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-8b-v0.1) | 8B | ~16 GB | 262K | Qwen3-VL-8B | Apache 2.0 |
| [SauerkrautLM-ColLFM2-450M-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1) | 450M | ~0.9 GB | 32K | LFM2 | Apache 2.0 |
| [SauerkrautLM-ColMinistral3-3b-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1) | 3B | ~6 GB | 262K | Ministral-3B | Apache 2.0 |

**Supported Languages:** English, German, French, Spanish, Italian, Portuguese

## 🎯 Why Visual Document Retrieval?

Traditional document retrieval relies on **OCR + Text Search**, which has significant limitations:

| Approach | Limitations |
|----------|-------------|
| **OCR-based** | ❌ Loses layout information, tables, charts, images |
| **OCR-based** | ❌ OCR errors compound in downstream tasks |
| **OCR-based** | ❌ Struggles with handwriting, low-quality scans |
| **OCR-based** | ❌ Cannot understand visual elements (logos, diagrams) |

**Visual Document Retrieval** solves these problems by:
- ✅ **Direct visual understanding** - No OCR errors, preserves full document context
- ✅ **Layout-aware** - Understands tables, forms, multi-column layouts
- ✅ **Multimodal** - Combines text and visual elements naturally
- ✅ **End-to-end** - Single model for retrieval, no pipeline complexity

<p align="center">
  <img src="assets/benchmark_vidore_v1_vs_highdim.png" alt="ViDoRe v1 Benchmark - 128-dim vs High-dim" width="50%"/>
</p>

## 🏆 Benchmark Results

Our models achieve **state-of-the-art** performance on the ViDoRe (Visual Document Retrieval) benchmarks while maintaining a **compact 128-dimensional embedding space** for efficient retrieval.

### Key Highlights

| Achievement | Model | Score | Comparison |
|------------|-------|-------|------------|
| 🥇 **#1 ViDoRe v1 (128-dim)** | ColQwen3-8b | **91.08** | Beats all 128-dim models |
| 🥇 **#1 ViDoRe v3 (128-dim)** | ColQwen3-8b | **58.55** | Best 128-dim model |
| 🥇 **#1 Small Model (<1B)** | ColLFM2-450M | **83.56** | Beats colSmol-500M with fewer params |
| 🥇 **#1 Medium (1-3B, 128-dim)** | ColQwen3-2b | **90.24** | Best 128-dim in 1-3B class |
| ⚡ **Most Efficient** | All models | **128 dim** | Same dim as ColPali, 2.5-24x smaller than high-dim competitors |

### 128-dim Models Comparison (Fair Comparison)

When comparing only models with the same 128-dimensional embedding space:

| Model | Params | ViDoRe v1 | MTEB v1+v2 | ViDoRe v3 |
|-------|--------|-----------|------------|-----------|
| **SauerkrautLM-ColQwen3-8b-v0.1** ⭐ | 8.0B | **91.08 (#1)** | 82.91 (#2) | **58.55 (#1)** |
| EvoQwen2.5-VL-Retriever-7B-v1 | 7.0B | 90.68 (#3) | **83.41 (#1)** | - |
| **SauerkrautLM-ColQwen3-4b-v0.1** | 4.0B | 90.80 (#2) | 81.97 (#4) | 56.03 (#4) |
| EvoQwen2.5-VL-Retriever-3B-v1 | 3.0B | 90.67 (#4) | 82.76 (#3) | - |
| **SauerkrautLM-ColQwen3-2b-v0.1** | 2.2B | 90.24 (#5) | 81.02 (#6) | 54.32 (#5) |
| colnomic-embed-multimodal-7b | 7.0B | 89.72 (#7) | 81.30 (#5) | 57.64 (#2) |
| colnomic-embed-multimodal-3b | 3.0B | 89.86 (#6) | 80.09 (#7) | 56.40 (#3) |
| colqwen2-v1.0 | 2.2B | 89.23 (#8) | 79.74 (#8) | 44.18 (#7) |
| **SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1** | 1.7B | 88.89 (#9) | 77.94 (#9) | 48.76 (#6) |
| colpali-v1.3 | 2.9B | 84.75 (#10) | 76.17 (#10) | 42.95 (#9) |
| **SauerkrautLM-ColLFM2-450M-v0.1** | 450M | 83.56 | 74.33 | 43.32 (#8) |

*Rankings among 128-dim models only. ⭐ = Best in category. Bold = our models.*

### Size Category Comparison (128-dim Models)

**Small Models (<1B):**
| Model | Params | ViDoRe v1 | MTEB v1+v2 | ViDoRe v3 |
|-------|--------|-----------|------------|-----------|
| **SauerkrautLM-ColLFM2-450M-v0.1** ⭐ | 450M | **83.56** | **74.33** | **43.32** |
| colSmol-500M | 500M | 82.49 | 71.17 | - |
| colSmol-256M | 256M | 79.74 | 66.90 | 20.73 |

**Medium Models (1-3B):**
| Model | Params | ViDoRe v1 | MTEB v1+v2 | ViDoRe v3 |
|-------|--------|-----------|------------|-----------|
| **SauerkrautLM-ColQwen3-2b-v0.1** ⭐ | 2.2B | **90.24** | **81.02** | **54.32** |
| colqwen2-v1.0 | 2.2B | 89.23 | 79.74 | 44.18 |
| **SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1** | 1.7B | 88.89 | 77.94 | 48.76 |

**Large Models (3-5B):**
| Model | Params | ViDoRe v1 | MTEB v1+v2 | ViDoRe v3 |
|-------|--------|-----------|------------|-----------|
| **SauerkrautLM-ColQwen3-4b-v0.1** ⭐ | 4.0B | **90.80** | 81.97 | 56.03 |
| EvoQwen2.5-VL-Retriever-3B-v1 | 3.0B | 90.67 | **82.76** | - |
| colnomic-embed-multimodal-3b | 3.0B | 89.86 | 80.09 | **56.40** |

**XLarge Models (5-10B):**
| Model | Params | ViDoRe v1 | MTEB v1+v2 | ViDoRe v3 |
|-------|--------|-----------|------------|-----------|
| **SauerkrautLM-ColQwen3-8b-v0.1** ⭐ | 8.0B | **91.08** | 82.91 | **58.55** |
| EvoQwen2.5-VL-Retriever-7B-v1 | 7.0B | 90.68 | **83.41** | - |
| colnomic-embed-multimodal-7b | 7.0B | 89.72 | 81.30 | 57.64 |

*⭐ = Best 128-dim model in category*

### Why Choose Our Models?

1. **🏆 #1 in 128-dim Class**: Our ColQwen3-8b beats ALL other 128-dim models on ViDoRe v1 and v3

2. **⚡ Compact Embeddings**: All our models use 128 dimensions - same as ColPali/ColQwen2/colSmol
   - No storage overhead compared to standard ColPali models
   - 2.5x smaller than tomoro (320 dim)
   - 16-24x smaller than llama-nemoretriever (2048-3072 dim)

3. **💰 Best-in-Class for Every Size**:
   - **Small (<1B)**: ColLFM2-450M beats colSmol-500M with 10% fewer parameters
   - **Medium (1-3B)**: ColQwen3-2b beats colqwen2-v1.0 by +1.01 points
   - **Large (3-5B)**: ColQwen3-4b achieves 90.80, only -0.20 behind much larger llama-nemo
   - **XLarge (5-10B)**: ColQwen3-8b achieves the highest 128-dim score ever

4. **🌍 Multilingual**: Trained on 6 languages (EN, DE, FR, ES, IT, PT)

5. **🔧 Easy Integration**: MTEB-compatible for standardized evaluation

## Training

### Hardware

| Model Size | GPUs | Effective Batch Size |
|------------|------|---------------------|
| 450M - 4B | 4x NVIDIA RTX 6000 Ada (48GB) | 256 |
| 8B | 4x NVIDIA A100 SXM (80GB) | 256 |

### Training Datasets

Our models were trained on a diverse mix of public and proprietary datasets:

| Dataset | Type | Description |
|---------|------|-------------|
| [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) | Public | Original ColPali training data with document-query pairs |
| [openbmb/VisRAG-Ret-Train-In-domain-data](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) | Public | Visual RAG retrieval training data |
| [llamaindex/vdr-multilingual-train](https://huggingface.co/datasets/llamaindex/vdr-multilingual-train) | Public | Multilingual document retrieval data |
| [unicamp-dl/mmarco](https://huggingface.co/datasets/unicamp-dl/mmarco) | Public | Multilingual MS MARCO (used for recovery training) |
| VAGO Multilingual Dataset 1 | **In-house** | Proprietary multilingual document-query pairs |
| VAGO Multilingual Dataset 2 | **In-house** | Proprietary multilingual document-query pairs |

### Special Training Techniques

| Model | Technique | Description |
|-------|-----------|-------------|
| **ColLFM2-450M** | Curriculum Learning | Progressive difficulty training across 4 stages |
| **ColLFM2-450M** | Hierarchical Merge | Combined mMARCO specialist with retrieval model |
| **ColQwen3-1.7b-Turbo** | Structured Pruning | Layer + intermediate size pruning (-23% params) |
| **ColQwen3-1.7b-Turbo** | mMARCO Recovery | Pre-training to heal pruned model |

## Installation

```bash
# From source (recommended)
pip install git+https://github.com/VAGOsolutions/sauerkrautlm-colpali

# For ColMinistral3 models (requires transformers 5.0.0rc0)
pip install "sauerkrautlm-colpali[ministral]"
```

> **Note**: ColMinistral3 requires `transformers>=5.0.0rc0`. Install with `pip install "sauerkrautlm-colpali[ministral]"` or manually install the RC version.

## Quick Start

### ColQwen3 Example

```python
import torch
from PIL import Image
from sauerkrautlm_colpali.models import ColQwen3, ColQwen3Processor

model_name = "VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1"

# Load model and processor
model = ColQwen3.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()

processor = ColQwen3Processor.from_pretrained(model_name)

# Prepare inputs
images = [Image.open("document.png")]
queries = ["What is the main topic of this document?"]

# Process inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Get embeddings
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

# Calculate similarity scores
scores = processor.score(query_embeddings, image_embeddings)
print(f"Similarity scores: {scores}")
```

### ColMinistral3 Example

```python
import torch
from PIL import Image
from sauerkrautlm_colpali.models import ColMinistral3, ColMinistral3Processor

model_name = "VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1"

model = ColMinistral3.from_pretrained(model_name)
model = model.to(dtype=torch.bfloat16, device="cuda:0").eval()

processor = ColMinistral3Processor.from_pretrained(model_name)

# Same usage pattern as ColQwen3...
```

### ColLFM2 Example

```python
import torch
from PIL import Image
from sauerkrautlm_colpali.models import ColLFM2, ColLFM2Processor

model_name = "VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1"

model = ColLFM2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()

processor = ColLFM2Processor.from_pretrained(model_name)

# Same usage pattern as ColQwen3...
```

## MTEB Integration

This package includes MTEB (Massive Text Embedding Benchmark) integration for standardized evaluation on the ViDoRe benchmark. See the `mteb_integration/` folder for details.

```python
# Example: Run MTEB evaluation
import mteb

model = mteb.get_model("VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1")
tasks = mteb.get_tasks(tasks=["VidoreArxivQARetrieval"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/")
```

## Architecture

All models in this package follow the ColPali architecture:
1. **Vision Encoder**: Extracts patch embeddings from document images
2. **Language Model**: Processes visual tokens alongside text tokens
3. **Projection Layer**: Maps hidden states to 128-dimensional embedding space
4. **Late Interaction**: MaxSim scoring between query and document embeddings

```
Document Image → Vision Encoder → Visual Tokens → LLM → Projection → Multi-Vector Embeddings
                                                                              ↓
Query Text → Tokenizer → LLM → Projection → Multi-Vector Embeddings → MaxSim Score
```

## Original ColPali

This package is based on the excellent work by Illuin Technology. For the original ColPali models (ColPali, ColQwen2, ColQwen2.5, ColSmol), please use the original [colpali-engine](https://github.com/illuin-tech/colpali) package.

## Citation

If you use this package, please cite both the original ColPali paper and our work:

```bibtex
@misc{sauerkrautlm-colpali-2025,
  title={SauerkrautLM-ColPali: Multi-Vector Vision Retrieval Models},
  author={David Golchinfar},
  organization={VAGO Solutions},
  year={2025},
  url={https://github.com/VAGOsolutions/sauerkrautlm-colpali}
}

@misc{faysse2024colpaliefficientdocumentretrieval,
  title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
  author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
  year={2024},
  eprint={2407.01449},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2407.01449}, 
}
```

## 📊 Benchmark Visualizations

### ViDoRe v1 Benchmark (128-dim vs High-dim)
<p align="center">
  <img src="assets/benchmark_vidore_v1_vs_highdim.png" alt="ViDoRe v1 Benchmark" width="50%"/>
</p>

### MTEB v1+v2 Benchmark (128-dim vs High-dim)
<p align="center">
  <img src="assets/benchmark_mteb_v1v2_vs_highdim.png" alt="MTEB v1+v2 Benchmark" width="50%"/>
</p>

### ViDoRe v3 Benchmark (128-dim vs High-dim)
<p align="center">
  <img src="assets/benchmark_vidore_v3_vs_highdim.png" alt="ViDoRe v3 Benchmark" width="50%"/>
</p>

## 📋 Summary Tables

### 128-dim Models Comparison
<p align="center">
  <img src="assets/table_summary_128dim.png" alt="128-dim Models Summary" width="50%"/>
</p>

### Comparison vs High-dim Models
<p align="center">
  <img src="assets/table_summary_highdim_comparison.png" alt="High-dim Comparison" width="50%"/>
</p>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **VAGO Solutions**: [https://vago-solutions.ai](https://vago-solutions.ai)
- **GitHub**: [https://github.com/VAGOsolutions](https://github.com/VAGOsolutions)
- **Email**: info@vago-solutions.ai
