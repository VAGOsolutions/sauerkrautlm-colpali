# SauerkrautLM-ColPali MTEB Integration

This guide describes how to integrate the SauerkrautLM-ColPali models into MTEB.

## Step 1: Clone MTEB Repo

```bash
git clone https://github.com/embeddings-benchmark/mteb.git
cd mteb
git checkout -b feat/sauerkrautlm-colpali
pip install -e ".[dev]"
```

## Step 2: Copy Wrapper File

Copy the wrapper file into the MTEB repo:

```bash
cp mteb_integration/slm_models.py mteb/mteb/models/model_implementations/slm_models.py
```

## Step 3: Add Dependencies to pyproject.toml

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
sauerkrautlm-colpali = [
    "transformers>=4.47.0",
    "torch>=2.0.0",
    "sauerkrautlm-colpali @ git+https://github.com/VAGOsolutions/sauerkrautlm-colpali.git"
]
```

And under `conflicts` (if present):

```toml
[{ extra = "sauerkrautlm-colpali" }, { extra = "pylate" }],
[{ extra = "sauerkrautlm-colpali" }, { extra = "llm2vec" }],
```

## Step 4: Test

```bash
# Install dependencies
pip install -e ".[sauerkrautlm-colpali]"

# Test that the model can be loaded
python -c "
import mteb
model = mteb.get_model('VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1')
print(f'Model loaded: {model}')
"

# Test on a ViDoRe task
python -c "
import mteb

model = mteb.get_model('VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1')
tasks = mteb.get_tasks(tasks=['VidoreArxivQARetrieval'])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder='results/colqwen3-2b')
print(results)
"
```

## Step 5: Create PR

```bash
git add mteb/models/model_implementations/slm_models.py
git add pyproject.toml
git commit -m "model: Add SauerkrautLM-ColPali embedding models"
git push origin feat/sauerkrautlm-colpali
```

Then create a PR on GitHub against `embeddings-benchmark/mteb:main`.

## Available Models

| Model | HuggingFace ID | Dim | Parameters |
|-------|----------------|-----|------------|
| ColQwen3-2b | `VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1` | 128 | 2.2B |
| ColQwen3-4b | `VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1` | 128 | 4.0B |
| ColQwen3-8b | `VAGOsolutions/SauerkrautLM-ColQwen3-8b-v0.1` | 128 | 8.3B |
| ColQwen3-1.7b-Turbo | `VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1` | 128 | 1.7B |
| ColLFM2-450M | `VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1` | 128 | 450M |
| ColMinistral3-3b | `VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1` | 128 | 3.0B |

## PR Checklist

```markdown
- [x] I have filled out the ModelMeta object to the extent possible
- [x] I have ensured that my model can be loaded using
  - [x] `mteb.get_model(model_name, revision)` and
  - [x] `mteb.get_model_meta(model_name, revision)`
- [x] I have tested the implementation works on a representative set of tasks.
- [x] The model is public, i.e., is available either as an API or the weights are publicly available to download
```

## Important Notes

1. **Installation for Users**: After the PR is merged, users can simply install with:
   ```bash
   pip install mteb[sauerkrautlm-colpali]
   ```

2. **Multi-Vector Embeddings**: The models use ColBERT-style multi-vector embeddings with MaxSim scoring for ViDoRe tasks.

3. **GPU Recommended**: For optimal performance, a GPU with at least 8GB VRAM is recommended (depending on model size).
