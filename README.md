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
