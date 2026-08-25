# FLAN-T5 Base — Function Calling Experiments

Fine-tuning and evaluation experiments on [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1).

## Data Splits

Fixed, reproducible splits are stored in `data/splits/`:

| Split | File | Examples | Share |
|-------|------|----------|-------|
| Train | `data/splits/train.csv` | 8,628 | 80% |
| Validation | `data/splits/val.csv` | 1,078 | 10% |
| Test | `data/splits/test.csv` | 1,079 | 10% |

**Total:** 10,785 examples (deduplicated by `id` across all five dataset configs).

### Split procedure

1. Load all configs from `NousResearch/hermes-function-calling-v1` (`func_calling_singleturn`, `func_calling`, `glaive_func_calling`, `json_mode_agentic`, `json_mode_singleturn`).
2. Deduplicate by `id` (keeps first occurrence) to prevent leakage across splits.
3. Stratify by `subcategory` (null → `Unknown`; subcategories with fewer than 10 examples are grouped into `__rare__` so both split stages remain valid).
4. Split 80% / 20%, then divide the 20% evenly into validation and test.
5. **Random seed:** `42`

To regenerate the splits:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/create_splits.py
```

## Evaluation Methodology

All experiments are evaluated on the held-out **test split** (`data/splits/test.csv`). Validation is used for checkpoint selection and hyperparameter tuning only.

Predictions are compared to reference outputs using the metrics below. For function-calling tasks, a prediction is **correct** when it exactly matches the reference output string (exact-match accuracy).

### Metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **Exact-match accuracy** | Fraction of predictions identical to the reference | Primary score for overall model quality |
| **Precision** | TP / (TP + FP) | Penalizes incorrect or hallucinated function calls |
| **Recall** | TP / (TP + FN) | Penalizes missed function calls |
| **F1 score** | Harmonic mean of precision and recall | Balanced summary when both errors matter |
| **Per-subcategory accuracy** | Exact-match accuracy computed separately for each `subcategory` value | Surfaces strengths and weaknesses by task type |
| **Overall metrics** | Exact-match accuracy, precision, recall, and F1 aggregated across the full test set | Single-number comparison across experiments |

### Comparing experiments

Experiments are ranked primarily by **test-set exact-match accuracy**. When two runs are close, **F1 score** breaks ties. **Per-subcategory accuracy** identifies whether gains are broad or limited to a few task types, and the **validation split** confirms that test improvements are not due to overfitting. All experiments must use these fixed splits so results are directly comparable.

## Experiment 1: Pretrained Baseline (No Training)

**Goal:** Measure zero-shot / out-of-the-box performance of `google/flan-t5-base` on the fixed test split before any fine-tuning.

| Setting | Value |
|---------|-------|
| Model | [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) |
| Training | None (pretrained weights only) |
| Evaluation split | `data/splits/test.csv` (1,079 examples) |
| Prompt format | Conversation turns before the first `gpt` response, ending with `gpt:` |
| Target | First assistant (`gpt`) turn (tool call, JSON, or text) |
| Inference | Greedy decoding, `max_new_tokens=512` |

### How to run

**Google Colab (recommended):** open [`notebooks/experiment_1_pretrained.ipynb`](notebooks/experiment_1_pretrained.ipynb), enable a GPU runtime, and run all cells. The notebook:

1. Clones this repo and installs `requirements.txt`
2. Loads the saved test split via `src/data.py`
3. Runs inference with `src/inference.py` on Colab GPU
4. Computes metrics with `src/metrics.py` through `scripts/run_experiment.py`
5. Saves outputs to `experiments/experiment_1_pretrained/`
6. Provides Git commands to commit and push results

**Local CLI (optional):**

```bash
PYTHONPATH=. python scripts/run_experiment.py \
  --experiment-name experiment_1_pretrained \
  --model-name google/flan-t5-base \
  --split test
```

### Outputs

| File | Description |
|------|-------------|
| `experiments/experiment_1_pretrained/predictions.csv` | Per-example prompts, references, predictions, exact-match flag |
| `experiments/experiment_1_pretrained/metrics.json` | Overall and per-subcategory metrics |

### Results (test set)

> Run the Colab notebook above to generate metrics. After execution, paste the printed README block from the notebook's final inspection cell, or read values from `metrics.json`.

| Metric | Value |
|--------|-------|
| Exact-match accuracy | _pending Colab run_ |
| Precision | _pending Colab run_ |
| Recall | _pending Colab run_ |
| F1 | _pending Colab run_ |
| Examples | 1,079 |

### Findings (expected baseline)

- Without task-specific training, FLAN-T5 Base is expected to achieve **low exact-match accuracy** on Hermes function-calling outputs, which require precise JSON or `<tool_call>` formatting.
- **Tool-call precision/recall** should remain near zero for most subcategories because the pretrained model was not aligned to the Hermes prompt schema.
- **Per-subcategory accuracy** will likely be highest on short free-text first turns and near zero on structured JSON / tool-call subcategories (e.g., `Json Schema`, `Get Stock Price`).
- This experiment defines the **no-training baseline** that later fine-tuning runs must beat on the same test split.

