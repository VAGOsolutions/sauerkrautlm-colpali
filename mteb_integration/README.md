# SauerkrautLM-ColPali MTEB Integration

This guide describes how to integrate the SauerkrautLM-ColPali models with MTEB.

## Step 1: Clone MTEB Repository

```bash
cd /workspace
git clone https://github.com/embeddings-benchmark/mteb.git
cd mteb
git checkout -b feat/sauerkrautlm-colpali
pip install -e ".[dev]"
```

## Step 2: Copy Integration Files

Copy the wrapper file into the MTEB repository:

```bash
cp slm_models.py /workspace/mteb/mteb/models/model_implementations/slm_models.py
```

## Step 3: Test

```bash
# Test that the model can be loaded
python -c "
import mteb
model = mteb.get_model('VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1')
print(f'Model loaded: {model}')
"

# Test on a task
python -c "
import mteb

model = mteb.get_model('VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1')
tasks = mteb.get_tasks(tasks=['VidoreArxivQARetrieval'])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder='results/slm-colqwen3-2b')
print(results)
"
```

## Available Models

| Model Name | Parameters | VRAM (bf16) | Max Tokens |
|------------|------------|-------------|------------|
| `VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1` | 1.7B | ~3.4 GB | 262K |
| `VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1` | 2.2B | ~4.4 GB | 262K |
| `VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1` | 4B | ~8 GB | 262K |
| `VAGOsolutions/SauerkrautLM-ColQwen3-8b-v0.1` | 8B | ~16 GB | 262K |
| `VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1` | 450M | ~0.9 GB | 32K |
| `VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1` | 3B | ~6 GB | 262K |

## Important Notes

1. **Model Names**: The `name` in `ModelMeta` must exactly match the HuggingFace repository name.

2. **ColMinistral3**: Requires `transformers>=5.0.0rc0`. Install with:
   ```bash
   pip install "sauerkrautlm-colpali[ministral]"
   ```

3. **Multi-Vector Embeddings**: For ViDoRe tasks, MTEB uses `get_image_embeddings()` and `get_text_embeddings()` methods with MaxSim scoring.

4. **Supported Languages**: English, German, French, Spanish, Italian, Portuguese
