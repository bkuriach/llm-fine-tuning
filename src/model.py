"""
model.py – Model and tokenizer loading with QLoRA support.

Supports LLaMA, Qwen, and Mistral families out of the box.
The LoRA target modules are taken from the YAML config, so adding
a new model family only requires a new config file.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def _get_bnb_config(qlora_cfg: dict) -> BitsAndBytesConfig:
    """Build a ``BitsAndBytesConfig`` from the ``qlora`` section of the YAML."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(
        qlora_cfg.get("bnb_4bit_compute_dtype", "float16"), torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
    )


def load_tokenizer(model_name: str, hf_token: Optional[str] = None) -> AutoTokenizer:
    """Load tokenizer and ensure a pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Padding side: right for causal LM training
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded: %s  |  vocab_size=%d", model_name, len(tokenizer))
    return tokenizer


def load_model_and_tokenizer(
    model_cfg: dict,
    qlora_cfg: dict,
    lora_cfg: dict,
    hf_token: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model + tokenizer, apply QLoRA quantisation and LoRA adapters.

    Args:
        model_cfg:  ``model`` section from the resolved YAML config.
        qlora_cfg:  ``qlora`` section from the resolved YAML config.
        lora_cfg:   ``lora`` section from the resolved YAML config.
        hf_token:   HuggingFace token for gated models (optional).

    Returns:
        Tuple of ``(peft_model, tokenizer)``.
    """
    model_name: str = model_cfg["name"]
    torch_dtype_str: str = model_cfg.get("torch_dtype", "float16")
    attn_impl: Optional[str] = model_cfg.get("attn_implementation", None)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_str, torch.float16)

    tokenizer = load_tokenizer(model_name, hf_token=hf_token)

    # ── QLoRA quantisation ──────────────────────────────────────────
    bnb_config: Optional[BitsAndBytesConfig] = None
    if qlora_cfg.get("enabled", True):
        bnb_config = _get_bnb_config(qlora_cfg)
        logger.info("QLoRA enabled – loading model in 4-bit NF4")

    model_kwargs: dict = dict(
        pretrained_model_name_or_path=model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # ── LoRA adapters ───────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg.get("target_modules", None),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
