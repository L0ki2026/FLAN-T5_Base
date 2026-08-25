"""Evaluation metrics for function-calling experiments."""

from __future__ import annotations

import json
import re
from collections import defaultdict

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _normalize_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
    except json.JSONDecodeError:
        return _normalize_text(text)


def extract_units(text: str) -> set[str]:
    """Extract comparable prediction units for precision/recall."""
    units: set[str] = set()
    for match in TOOL_CALL_PATTERN.findall(text):
        try:
            payload = json.loads(match)
            name = payload.get("name", "")
            args = json.dumps(payload.get("arguments", {}), sort_keys=True)
            units.add(f"tool:{name}:{args}")
        except json.JSONDecodeError:
            units.add(f"tool_raw:{_normalize_text(match)}")

    stripped = text.strip()
    if units:
        return units
    if stripped.startswith("{"):
        return {f"json:{_normalize_json(stripped)}"}
    if stripped:
        return {f"text:{_normalize_text(stripped)}"}
    return set()


def exact_match(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(
        _normalize_text(p) == _normalize_text(r) for p, r in zip(predictions, references)
    )
    return matches / len(predictions)


def precision_recall_f1(
    predictions: list[str], references: list[str]
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        pred_units = extract_units(pred)
        ref_units = extract_units(ref)
        tp += len(pred_units & ref_units)
        fp += len(pred_units - ref_units)
        fn += len(ref_units - pred_units)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def per_subcategory_accuracy(
    predictions: list[str],
    references: list[str],
    subcategories: list[str],
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for pred, ref, subcat in zip(predictions, references, subcategories):
        label = subcat if isinstance(subcat, str) and subcat else "Unknown"
        buckets[label].append(_normalize_text(pred) == _normalize_text(ref))

    results: dict[str, dict[str, float | int]] = {}
    for subcat, matches in sorted(buckets.items()):
        results[subcat] = {
            "count": len(matches),
            "accuracy": sum(matches) / len(matches),
        }
    return results


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    subcategories: list[str],
) -> dict:
    precision, recall, f1 = precision_recall_f1(predictions, references)
    return {
        "overall": {
            "exact_match_accuracy": exact_match(predictions, references),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_examples": len(predictions),
        },
        "per_subcategory": per_subcategory_accuracy(
            predictions, references, subcategories
        ),
    }
