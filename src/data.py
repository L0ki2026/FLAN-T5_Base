"""Load fixed train/val/test splits from the repository."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
DATASET = "NousResearch/hermes-function-calling-v1"


def load_split_csv(name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_examples_by_id() -> dict[str, dict]:
    """Resolve full examples from Hugging Face using split ids."""
    examples: dict[str, dict] = {}
    for config in get_dataset_config_names(DATASET):
        ds = load_dataset(DATASET, config, split="train")
        for row in ds:
            examples[row["id"]] = dict(row)
    return examples


def load_split(name: str) -> pd.DataFrame:
    """Return split metadata joined with full conversation content."""
    meta = load_split_csv(name)
    by_id = _load_examples_by_id()
    missing = set(meta["id"]) - set(by_id)
    if missing:
        raise KeyError(f"{len(missing)} ids from {name} split missing in dataset")

    records = []
    for _, row in meta.iterrows():
        example = by_id[row["id"]]
        records.append(
            {
                "id": row["id"],
                "conversations": example["conversations"],
                "tools": example.get("tools"),
                "category": row.get("category", example.get("category")),
                "subcategory": row.get("subcategory", example.get("subcategory")),
                "task": row.get("task", example.get("task")),
                "source_config": row["source_config"],
                "schema": example.get("schema"),
            }
        )
    return pd.DataFrame(records)


def get_input_target(conversations: list[dict]) -> tuple[str, str]:
    """First assistant turn is the prediction target; prior turns form the prompt."""
    gpt_idx = next(
        i for i, turn in enumerate(conversations) if turn["from"] in ("gpt", "assistant")
    )
    prompt = format_prompt(conversations[:gpt_idx])
    target = conversations[gpt_idx]["value"]
    return prompt, target


def format_prompt(turns: list[dict]) -> str:
    lines = [f"{turn['from']}: {turn['value']}" for turn in turns]
    lines.append("gpt:")
    return "\n".join(lines)
