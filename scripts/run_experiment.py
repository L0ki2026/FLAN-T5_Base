#!/usr/bin/env python3
"""Run a function-calling evaluation experiment on a saved split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import get_input_target, load_split
from src.inference import generate_predictions, load_model
from src.metrics import compute_all_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_experiment(
    *,
    experiment_name: str,
    model_name: str = "google/flan-t5-base",
    split: str = "test",
    batch_size: int = 8,
    max_new_tokens: int = 512,
    output_dir: Path | None = None,
    device: str | None = None,
) -> dict:
    """Run inference and evaluation; save predictions and metrics."""
    output_dir = output_dir or REPO_ROOT / "experiments" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = load_split(split)
    prompts: list[str] = []
    references: list[str] = []
    subcategories: list[str] = []

    for _, row in split_df.iterrows():
        prompt, target = get_input_target(row["conversations"])
        prompts.append(prompt)
        references.append(target)
        subcat = row["subcategory"]
        subcategories.append(subcat if isinstance(subcat, str) and subcat else "Unknown")

    model, tokenizer, resolved_device = load_model(model_name, device=device)
    predictions = generate_predictions(
        model,
        tokenizer,
        prompts,
        device=resolved_device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    metrics = compute_all_metrics(predictions, references, subcategories)
    metrics["experiment"] = {
        "name": experiment_name,
        "model_name": model_name,
        "split": split,
        "training": "none",
        "device": resolved_device,
        "batch_size": batch_size,
    }

    predictions_df = pd.DataFrame(
        {
            "id": split_df["id"],
            "subcategory": subcategories,
            "source_config": split_df["source_config"],
            "prompt": prompts,
            "reference": references,
            "prediction": predictions,
            "exact_match": [
                p.strip() == r.strip() for p, r in zip(predictions, references)
            ],
        }
    )

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def print_metrics_summary(metrics: dict) -> None:
    overall = metrics["overall"]
    print(f"Experiment: {metrics['experiment']['name']}")
    print(f"Model: {metrics['experiment']['model_name']}")
    print(f"Split: {metrics['experiment']['split']} ({overall['num_examples']} examples)")
    print(f"Exact-match accuracy: {overall['exact_match_accuracy']:.4f}")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall: {overall['recall']:.4f}")
    print(f"F1: {overall['f1']:.4f}")


def format_readme_results(metrics: dict, top_k: int = 5) -> str:
    """Return a markdown snippet for README Experiment 1 results."""
    overall = metrics["overall"]
    per_sub = metrics["per_subcategory"]
    ranked = sorted(
        per_sub.items(),
        key=lambda item: (item[1]["accuracy"], item[1]["count"]),
        reverse=True,
    )
    top = [x for x in ranked if x[1]["count"] >= 5][:top_k]
    bottom = sorted(
        [x for x in per_sub.items() if x[1]["count"] >= 5],
        key=lambda item: item[1]["accuracy"],
    )[:top_k]

    lines = [
        "### Results (test set)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Exact-match accuracy | {overall['exact_match_accuracy']:.4f} |",
        f"| Precision | {overall['precision']:.4f} |",
        f"| Recall | {overall['recall']:.4f} |",
        f"| F1 | {overall['f1']:.4f} |",
        f"| Examples | {overall['num_examples']} |",
        "",
        "**Top subcategories (accuracy, n ≥ 5):**",
    ]
    for name, stats in top:
        lines.append(f"- {name}: {stats['accuracy']:.3f} ({stats['count']} examples)")
    lines.extend(["", "**Weakest subcategories (accuracy, n ≥ 5):**"])
    for name, stats in bottom:
        lines.append(f"- {name}: {stats['accuracy']:.3f} ({stats['count']} examples)")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to experiments/<experiment-name>/",
    )
    parser.add_argument("--device", default=None, help="Force device (cuda, cpu, mps)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        device=args.device,
    )
    print_metrics_summary(metrics)
    output_dir = args.output_dir or REPO_ROOT / "experiments" / args.experiment_name
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
