"""
train.py – Main SFT training entry-point.

Usage (local):
    python src/train.py --config configs/llama_config.yaml

Usage (AML – launched by aml/submit_job.py):
    python src/train.py --config configs/llama_config.yaml \\
                        --data_path $AZUREML_DATAREFERENCES_data
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# Allow imports from src/ when run directly from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trl import SFTConfig, SFTTrainer

from dataset import build_dataset, build_inference_prompt
from evaluate import evaluate as run_evaluate
from model import load_model_and_tokenizer
from utils import get_hf_token, get_output_dir, is_aml_run, load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    tokenizer,
    eval_dataset,
    output_dir: Path,
    prompt_template: str,
    use_chat_template: bool = False,
    system_prompt: str = "",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 0,
) -> Path:
    """Run decoding on *eval_dataset* and write results to ``output.csv``.

    The dataset must contain ``input`` and ``output`` columns (preserved by
    ``dataset.build_dataset``).

    Args:
        model:                Fine-tuned (PEFT) model.
        tokenizer:            Matching tokenizer.
        eval_dataset:         HuggingFace Dataset with ``input`` / ``output`` columns.
        output_dir:           Directory where ``output.csv`` will be written.
        prompt_template:      Plain-text template (fallback when chat template disabled).
        use_chat_template:    If ``True``, build prompts via tokenizer.apply_chat_template.
        system_prompt:        System message used when building chat-template prompts.
        batch_size:           Number of samples decoded in each forward pass.
        max_new_tokens:       Maximum new tokens per sample.
        num_beams:            Beam width (1 = greedy, >1 = beam search).
        no_repeat_ngram_size: Prevent repeated n-grams of this size (0 = disabled).

    Returns:
        Path to the written CSV file.
    """
    import torch

    # Left-padding keeps attention on the generated tokens during beam search.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    model.eval()

    inputs_list: list[str] = eval_dataset["input"]
    expected_list: list[str] = eval_dataset["output"]
    results: list[dict] = []

    mode = f"beam_search (beams={num_beams})" if num_beams > 1 else "greedy"
    logger.info(
        "Running inference on %d test samples (batch_size=%d, mode=%s) …",
        len(inputs_list), batch_size, mode,
    )

    for i in range(0, len(inputs_list), batch_size):
        batch_inputs = inputs_list[i : i + batch_size]
        batch_expected = expected_list[i : i + batch_size]

        prompts = [
            build_inference_prompt(
                inp,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
            )
            for inp in batch_inputs
        ]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["early_stopping"] = True
            gen_kwargs["do_sample"] = False
            if no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        else:
            gen_kwargs["do_sample"] = False  # greedy

        with torch.no_grad():
            output_ids = model.generate(**encoded, **gen_kwargs)

        # Slice off the prompt tokens so we only decode the newly generated part.
        prompt_len = encoded["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for inp, exp, pred in zip(batch_inputs, batch_expected, predictions):
            results.append({"input": inp, "expected": exp, "predicted": pred.strip()})

        if (i // batch_size + 1) % 10 == 0:
            logger.info("  … %d / %d", min(i + batch_size, len(inputs_list)), len(inputs_list))

    # Restore original padding side in case the tokenizer is reused.
    tokenizer.padding_side = orig_padding_side

    csv_path = output_dir / "output.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "expected", "predicted"])
        writer.writeheader()
        writer.writerows(results)

    logger.info("Inference complete – %d rows written to %s", len(results), csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT training for Hinglish translation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML model config (e.g. configs/llama_config.yaml)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override the training data path from the config (useful for AML data inputs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory from the config",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # ── Load & resolve config ───────────────────────────────────────
    logger.info("Loading config: %s", args.config)
    cfg = load_config(args.config)

    # CLI overrides
    if args.data_path:
        data_path = Path(args.data_path)
        if data_path.is_dir():
            # AML mounts URI_FOLDER inputs as directories.
            # Try to find the JSON file whose name matches the config, then
            # fall back to the first .json file found in the directory.
            config_filename = Path(cfg["data"]["train_file"]).name
            candidate = data_path / config_filename
            if candidate.exists():
                data_path = candidate
            else:
                json_files = list(data_path.glob("*.json"))
                if not json_files:
                    raise FileNotFoundError(f"No JSON files found in data directory: {data_path}")
                data_path = json_files[0]
                logger.warning("Config file '%s' not found in input dir; using '%s'", config_filename, data_path.name)
        cfg["data"]["train_file"] = str(data_path)
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir

    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ── MLflow / AML tracking ───────────────────────────────────────
    if is_aml_run():
        logger.info("Running inside Azure ML – metrics tracked via AML Studio")
        # Do NOT use mlflow.autolog() here – it conflicts with the transformers
        # MlflowCallback and AML enforces a 500-char limit on logged param values.
        # AML natively captures stdout/stderr and job metrics in its Studio UI.

    # ── Dataset ─────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    prompt_cfg = cfg["prompt"]
    prompt_template: str = prompt_cfg["template"]
    use_chat_template: bool = prompt_cfg.get("use_chat_template", False)
    system_prompt: str = prompt_cfg.get("system_prompt", "")

    # ── Model + tokenizer ───────────────────────────────────────────
    # Loaded BEFORE the dataset so we can pass the tokenizer to build_dataset
    # when use_chat_template is enabled (each model needs its own special tokens).
    model, tokenizer = load_model_and_tokenizer(
        model_cfg=cfg["model"],
        qlora_cfg=cfg.get("qlora", {}),
        lora_cfg=cfg["lora"],
        hf_token=get_hf_token(),
    )

    # Log parameter dtype summary for diagnostics
    from collections import Counter
    dtype_counts = Counter(str(p.dtype) for p in model.parameters())
    logger.info("Parameter dtypes after model load: %s", dict(dtype_counts))

    dataset = build_dataset(
        data_path=data_cfg["train_file"],
        prompt_template=prompt_template,
        eval_split=data_cfg.get("eval_split", 0.1),
        max_samples=data_cfg.get("max_samples"),
        seed=cfg["training"].get("seed", 42),
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", None)

    # ── SFTConfig (TRL) ─────────────────────────────────────────────
    train_cfg = cfg["training"]
    sft_config = SFTConfig(
        output_dir=str(output_dir),

        # Epochs / steps
        num_train_epochs=train_cfg.get("num_train_epochs", 3),

        # Batch size
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),

        # Optimiser
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=int(train_cfg.get("warmup_steps", 50)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),

        # Precision
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),

        # Gradient checkpointing
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),

        # Logging / checkpointing
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 200) if eval_dataset else None,
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True) if eval_dataset else False,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),

        # Reporting
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 42),

        # SFT-specific
        dataset_text_field="text",
        max_length=data_cfg.get("max_seq_length", 512),
        packing=False,                    # disable sample packing for translation tasks
        remove_unused_columns=False,
    )

    # ── Trainer ─────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training …")

    # Sanity check – catch CPU-only situations early
    import torch
    if torch.cuda.is_available():
        logger.info("CUDA available – device: %s", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError(
            "No CUDA device found. Training an 8B model on CPU is not feasible. "
            "Check that the compute cluster has a GPU and CUDA drivers are installed."
        )

    trainer.train()

    # ── Save ────────────────────────────────────────────────────────
    logger.info("Saving model to %s", output_dir)
    trainer.save_model(str(output_dir))

    # Save LoRA adapters separately for easy deployment
    lora_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    logger.info("LoRA adapter saved to %s", lora_dir)

    if is_aml_run():
        logger.info("Artifacts saved to %s – visible in AML Studio outputs", lora_dir)

    # ── Inference on test split ──────────────────────────────────────────────
    if eval_dataset is not None:
        logger.info("Running post-training inference on test split …")
        infer_cfg = cfg.get("inference", {})
        csv_path = run_inference(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            prompt_template=prompt_template,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            batch_size=infer_cfg.get("batch_size", 4),
            max_new_tokens=infer_cfg.get("max_new_tokens", 128),
            num_beams=infer_cfg.get("num_beams", 1),
            no_repeat_ngram_size=infer_cfg.get("no_repeat_ngram_size", 0),
        )

        # ── Automated metrics ────────────────────────────────────────────────
        logger.info("Computing evaluation metrics on %s …", csv_path)
        eval_cfg = cfg.get("evaluation", {})
        run_evaluate(
            csv_path=csv_path,
            output_dir=output_dir,
            bertscore_model=eval_cfg.get("bertscore_model", "xlm-roberta-large"),
            comet_model=eval_cfg.get("comet_model", "Unbabel/wmt22-comet-da"),
            skip_comet=eval_cfg.get("skip_comet", False),
        )
    else:
        logger.warning("No eval split available – skipping inference / output.csv generation.")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
