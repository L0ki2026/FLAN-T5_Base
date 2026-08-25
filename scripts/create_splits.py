"""Create reproducible train/val/test splits for hermes-function-calling-v1."""

import json
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from sklearn.model_selection import train_test_split

DATASET = "NousResearch/hermes-function-calling-v1"
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"


def load_combined() -> pd.DataFrame:
    configs = get_dataset_config_names(DATASET)
    parts = []
    for config in configs:
        ds = load_dataset(DATASET, config, split="train")
        ds = ds.add_column("source_config", [config] * len(ds))
        parts.append(ds)
    df = concatenate_datasets(parts).to_pandas()
    # Deduplicate by id so the same example cannot appear in multiple splits.
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    return df


def stratify_key(subcategories: pd.Series, min_count: int = 10) -> pd.Series:
    """Bucket rare subcategories so both split stages can stratify."""
    filled = subcategories.fillna("Unknown").astype(str)
    counts = filled.value_counts()
    return filled.map(lambda s: s if counts[s] >= min_count else "__rare__")


def serialize_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and pd.isna(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (ValueError, TypeError):
        pass
    return value


def main() -> None:
    df = load_combined()
    strat = stratify_key(df["subcategory"])

    train_df, temp_df, train_strat, temp_strat = train_test_split(
        df,
        strat,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED,
        stratify=strat,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=temp_strat,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = split_df.copy()
        for col in out.columns:
            out[col] = out[col].map(serialize_value)
        out.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    total = len(df)
    print(f"Total examples: {total}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pct = 100 * len(split_df) / total
        print(f"  {name}: {len(split_df)} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
