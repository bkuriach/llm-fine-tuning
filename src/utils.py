"""
utils.py – Configuration loading and miscellaneous helpers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env once when this module is first imported.
# Looks for .env in the project root (two levels up from src/).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading with base-config inheritance
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config, optionally merging with its ``base`` config.

    If the YAML contains a top-level ``base`` key pointing to another YAML file,
    that base is loaded first and deep-merged with the current config.

    Args:
        config_path: Path to the model-specific YAML config.

    Returns:
        Fully resolved configuration dictionary.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    base_key = cfg.pop("base", None)
    if base_key:
        # Resolve relative to project root (parent of configs/)
        base_path = config_path.parent.parent / base_key
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg: dict = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)

    logger.debug("Resolved config: %s", cfg)
    return cfg


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger for training runs."""
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


# ---------------------------------------------------------------------------
# AML / MLflow helpers
# ---------------------------------------------------------------------------

def is_aml_run() -> bool:
    """Return True when running inside an Azure ML job."""
    return "AZUREML_RUN_ID" in os.environ


def get_hf_token() -> str | None:
    """Return the HuggingFace token from the environment (loaded from .env)."""
    return os.environ.get("HF_TOKEN")


def get_output_dir(cfg: dict) -> Path:
    """Return the effective output directory.

    Inside AML the ``AZUREML_OUTPUT_DIR`` env var points to the mounted output;
    otherwise fall back to the value from the config.
    """
    aml_out = os.environ.get("AZUREML_OUTPUT_DIR")
    if aml_out:
        return Path(aml_out)
    return Path(cfg["training"]["output_dir"])
