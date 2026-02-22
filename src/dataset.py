"""
dataset.py – Data loading and prompt formatting for Hinglish translation SFT.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default prompt template (plain-text fallback when chat template is disabled)
# ---------------------------------------------------------------------------
DEFAULT_PROMPT_TEMPLATE = (
    "Translate the following English sentence into natural, conversational Hinglish.\n\n"
    "English: {input}\n"
    "Hinglish: {output}"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a translation assistant. Translate the given English sentence into "
    "natural, conversational Hinglish \u2014 a blend of Hindi and English commonly "
    "used in everyday Indian speech. Keep it fluent and idiomatic; do not "
    "transliterate word-for-word."
)


def load_raw_data(data_path: str | Path, max_samples: Optional[int] = None) -> list[dict]:
    """Load raw JSON data from *data_path*.

    Expected format: a JSON array of ``{"input": ..., "output": ...}`` objects.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    logger.info("Loaded %d samples from %s", len(data), data_path)
    return data


def format_sample(
    item: dict,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> dict:
    """Plain-text fallback formatter.

    Returns a dict with three keys:
    - ``text``   – full prompt used for SFT training
    - ``input``  – raw source sentence (used to build inference-time prompt)
    - ``output`` – raw reference translation (used as ground truth in ``output.csv``)
    """
    input_text = item.get("input", "").strip()
    output_text = item.get("output", "").strip()
    text = prompt_template.format(input=input_text, output=output_text)
    return {"text": text, "input": input_text, "output": output_text}


def format_sample_chat(
    item: dict,
    tokenizer: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """Format a sample using the tokenizer's built-in chat template.

    This is the preferred path for instruction-tuned models (LLaMA-3, Qwen2.5,
    Mistral) because it inserts the correct model-specific special tokens
    (e.g. ``<|eot_id|>`` for LLaMA-3, ``<|im_end|>`` for Qwen, ``[INST]`` for
    Mistral) that the model was RLHF'd on.

    Returns the same three-key dict as :func:`format_sample`.
    """
    input_text = item.get("input", "").strip()
    output_text = item.get("output", "").strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate to Hinglish: {input_text}"},
        {"role": "assistant", "content": output_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # full text including assistant turn for SFT
    )
    return {"text": text, "input": input_text, "output": output_text}


def build_inference_prompt(
    input_text: str,
    tokenizer: Any,
    use_chat_template: bool,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    """Build a single inference-time prompt (no assistant turn).

    When *use_chat_template* is True the tokenizer renders the system + user
    messages and appends the generation-prompt token so the model knows to
    start generating the assistant reply.
    """
    if use_chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Translate to Hinglish: {input_text}"},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # appends <|start_header_id|>assistant<|end_header_id|> etc.
        )
    else:
        # Plain-text: strip the "{output}" placeholder and everything after it
        return prompt_template.split("{output}")[0].format(input=input_text)


def build_dataset(
    data_path: str | Path,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    eval_split: float = 0.1,
    max_samples: Optional[int] = None,
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    use_chat_template: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> DatasetDict:
    """Return a ``DatasetDict`` with ``"train"`` and ``"test"`` splits.

    When *tokenizer* is supplied and *use_chat_template* is ``True``, each
    sample is formatted via ``tokenizer.apply_chat_template`` so the model
    receives its native special tokens.  Otherwise the plain *prompt_template*
    is used as a fallback.

    Args:
        data_path:          Path to the JSON training file.
        prompt_template:    Plain-text fallback template with ``{input}`` / ``{output}``.
        eval_split:         Fraction of data held out for evaluation.
        max_samples:        Cap on the number of samples (``None`` = all).
        seed:               Random seed for the train/test split.
        tokenizer:          HuggingFace tokenizer (required for chat template).
        use_chat_template:  If ``True`` and *tokenizer* is provided, use chat format.
        system_prompt:      System message injected as the first chat turn.

    Returns:
        ``DatasetDict`` with ``train`` and ``test`` keys.
    """
    raw = load_raw_data(data_path, max_samples=max_samples)

    if use_chat_template and tokenizer is not None:
        logger.info("Formatting dataset with tokenizer chat template.")
        formatted = [format_sample_chat(item, tokenizer, system_prompt) for item in raw]
    else:
        if use_chat_template:
            logger.warning("use_chat_template=True but no tokenizer provided; falling back to plain template.")
        formatted = [format_sample(item, prompt_template) for item in raw]

    dataset = Dataset.from_list(formatted)

    if eval_split > 0.0:
        split = dataset.train_test_split(test_size=eval_split, seed=seed)
        logger.info(
            "Dataset split \u2192 train: %d  eval: %d",
            len(split["train"]),
            len(split["test"]),
        )
        return split
    else:
        logger.info("Using full dataset for training (%d samples)", len(dataset))
        return DatasetDict({"train": dataset})
