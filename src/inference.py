"""Model inference helpers."""

from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm


def load_model(model_name: str, device: str | None = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_predictions(
    model,
    tokenizer,
    prompts: Iterable[str],
    *,
    device: str,
    batch_size: int = 8,
    max_input_length: int = 2048,
    max_new_tokens: int = 512,
) -> list[str]:
    prompt_list = list(prompts)
    outputs: list[str] = []

    for start in tqdm(range(0, len(prompt_list), batch_size), desc="Generating"):
        batch = prompt_list[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs
