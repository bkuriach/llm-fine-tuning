"""
evaluate.py – Automated translation quality metrics for Hinglish output.

Computes three complementary metrics on a CSV produced by the inference step:
  • chrF    – character n-gram F-score (sacrebleu); robust for code-switched text
  • BERTScore – semantic similarity via multilingual embeddings (xlm-roberta-large)
  • COMET  – neural MT quality metric trained on human judgements (wmt22-comet-da)

Usage (standalone):
    python src/evaluate.py --csv outputs/llama/output.csv

Called automatically by train.py after inference when an output.csv exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (sources, references, hypotheses) from the output CSV."""
    sources, references, hypotheses = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sources.append(row["input"].strip())
            references.append(row["expected"].strip())
            hypotheses.append(row["predicted"].strip())
    logger.info("Loaded %d rows from %s", len(sources), csv_path)
    return sources, references, hypotheses


def compute_chrf(hypotheses: list[str], references: list[str]) -> dict:
    """chrF via sacrebleu.  Returns score dict with ``chrf`` key."""
    from sacrebleu.metrics import CHRF
    metric = CHRF()
    result = metric.corpus_score(hypotheses, [references])
    return {
        "chrf": round(result.score, 4),
    }


def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "xlm-roberta-large",
    batch_size: int = 32,
) -> dict:
    """BERTScore (F1) using *model_type*.

    xlm-roberta-large is the recommended backbone for multilingual / code-switched
    text; it handles both Latin-script Hinglish and Devanagari.
    """
    from bert_score import score as bscore
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Computing BERTScore with %s on %s …", model_type, device)
    _, _, F1 = bscore(
        hypotheses,
        references,
        model_type=model_type,
        lang="hi",           # closest proxy for Hinglish in BERTScore's lang map
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    return {
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def compute_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 16,
    gpus: int = 1,
) -> dict:
    """COMET score (reference-based) using *model_name*.

    ``wmt22-comet-da`` is the standard choice; it takes source + hypothesis +
    reference and returns a score in roughly [0, 1] (higher = better quality).

    Falls back gracefully if the comet package is unavailable.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        logger.warning("unbabel-comet not installed – skipping COMET metric.")
        return {"comet": None}

    import torch
    logger.info("Loading COMET model %s …", model_name)
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    gpus = min(gpus, 1 if torch.cuda.is_available() else 0)
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    # output.system_score is the corpus-level mean
    return {
        "comet": round(float(output.system_score), 4),
    }


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def evaluate(
    csv_path: Path,
    output_dir: Path | None = None,
    bertscore_model: str = "xlm-roberta-large",
    comet_model: str = "Unbabel/wmt22-comet-da",
    skip_comet: bool = False,
) -> dict:
    """Run all metrics against *csv_path* and return a results dict.

    Also writes ``metrics.json`` to *output_dir* (defaults to the same directory
    as the CSV).
    """
    sources, references, hypotheses = load_csv(csv_path)

    results: dict = {}

    # ── chrF ──────────────────────────────────────────────────────────
    logger.info("Computing chrF …")
    try:
        results.update(compute_chrf(hypotheses, references))
    except Exception as exc:
        logger.warning("chrF failed: %s", exc)
        results["chrf"] = None

    # ── BERTScore ─────────────────────────────────────────────────────
    logger.info("Computing BERTScore …")
    try:
        results.update(compute_bertscore(hypotheses, references, model_type=bertscore_model))
    except Exception as exc:
        logger.warning("BERTScore failed: %s", exc)
        results["bertscore_f1"] = None

    # ── COMET ─────────────────────────────────────────────────────────
    if not skip_comet:
        logger.info("Computing COMET …")
        try:
            results.update(compute_comet(sources, hypotheses, references, model_name=comet_model))
        except Exception as exc:
            logger.warning("COMET failed: %s", exc)
            results["comet"] = None
    else:
        logger.info("Skipping COMET (skip_comet=True).")
        results["comet"] = None

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        label = k.upper().replace("_", " ")
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {label:<20} {val}")
    print("=" * 50 + "\n")

    # Write metrics.json alongside output.csv
    out_dir = output_dir or csv_path.parent
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hinglish translation output.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to output.csv")
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="xlm-roberta-large",
        help="HuggingFace model for BERTScore (default: xlm-roberta-large)",
    )
    parser.add_argument(
        "--comet_model",
        type=str,
        default="Unbabel/wmt22-comet-da",
        help="COMET model name (default: Unbabel/wmt22-comet-da)",
    )
    parser.add_argument(
        "--skip_comet",
        action="store_true",
        help="Skip COMET (saves time; useful for quick local checks)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s – %(message)s")
    args = parse_args()
    evaluate(
        csv_path=Path(args.csv),
        bertscore_model=args.bertscore_model,
        comet_model=args.comet_model,
        skip_comet=args.skip_comet,
    )
