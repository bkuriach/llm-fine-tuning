"""
submit_job.py – Submit a single-step QLoRA SFT training job to Azure ML.

Usage:
    # LLaMA (default)
    python aml/submit_job.py --config configs/llama_config.yaml

    # Qwen
    python aml/submit_job.py --config configs/qwen_config.yaml

    # Mistral
    python aml/submit_job.py --config configs/mistral_config.yaml

    # Dry-run (print job spec without submitting)
    python aml/submit_job.py --config configs/llama_config.yaml --dry_run

Prerequisites:
    pip install azure-ai-ml azure-identity

    If you need to authenticate with a HuggingFace gated model (e.g. LLaMA),
    set HF_TOKEN in a Key Vault secret or export it as an environment variable
    before submitting.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load .env from project root (silently ignored if the file doesn't exist)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# AML workspace details – read from .env; no hardcoded fallbacks
# ---------------------------------------------------------------------------
def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{var}' is not set. "
            f"Copy .env.example → .env and fill in your values."
        )
    return val

SUBSCRIPTION_ID = _require_env("AML_SUBSCRIPTION_ID")
RESOURCE_GROUP  = _require_env("AML_RESOURCE_GROUP")
WORKSPACE_NAME  = _require_env("AML_WORKSPACE_NAME")
COMPUTE_CLUSTER = _require_env("AML_COMPUTE_CLUSTER")

# Project root (one level up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_name(config_path: str) -> str:
    """Derive a short display name from the config file name."""
    stem = Path(config_path).stem              # e.g. llama_config
    stem = re.sub(r"_config$", "", stem)       # → llama
    return f"sft-{stem}-hinglish"


def _get_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def _get_or_create_environment(ml_client: MLClient) -> str:
    """Register (or reuse) the SFT environment and return its versioned id."""
    import yaml as _yaml

    env_spec_path = PROJECT_ROOT / "aml" / "environment_spec.yaml"
    conda_file_path = PROJECT_ROOT / "aml" / "environment.yml"

    with open(env_spec_path, "r") as f:
        spec = _yaml.safe_load(f)

    env = Environment(
        name=spec["name"],
        version=str(spec["version"]),
        description=spec.get("description", ""),
        image=spec["image"],
        conda_file=str(conda_file_path),
    )

    # Check if this version already exists to avoid redundant rebuilds
    try:
        existing = ml_client.environments.get(env.name, version=env.version)
        print(f"[environment] Reusing existing: {existing.name}:{existing.version}")
        return f"{existing.name}:{existing.version}"
    except Exception:
        registered = ml_client.environments.create_or_update(env)
        print(f"[environment] Registered: {registered.name}:{registered.version}")
        return f"{registered.name}:{registered.version}"


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------

def build_job(config_path: str, env_id: str, hf_token: str | None = None):
    """Build the AML command job."""
    display_name = _short_name(config_path)

    # Relative path of the config from project root (used inside the job)
    rel_config = Path(config_path)
    if rel_config.is_absolute():
        rel_config = rel_config.relative_to(PROJECT_ROOT)

    # Data input: upload the local data/ folder to AML default datastore
    data_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=str(PROJECT_ROOT / "data"),
    )

    environment_variables: dict = {
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        "HF_HOME": "/tmp/hf_cache",
    }
    if hf_token:
        environment_variables["HF_TOKEN"] = hf_token

    job = command(
        display_name=display_name,
        experiment_name="hinglish-sft",
        description=f"QLoRA SFT training – {display_name}",

        # ── Compute ──────────────────────────────────────────────
        compute=COMPUTE_CLUSTER,

        # ── Environment ──────────────────────────────────────────
        environment=env_id,
        environment_variables=environment_variables,

        # ── Code + command ───────────────────────────────────────
        # Upload the entire project root so that both src/ and configs/ are available
        code=str(PROJECT_ROOT),
        command=(
            "python src/train.py"
            f" --config {rel_config}"
            " --data_path ${{inputs.data}}"
            " --log_level INFO"
        ),

        # ── Inputs ───────────────────────────────────────────────
        inputs={"data": data_input},

        # ── Distribution (single-node, single-GPU for now) ───────
        # To enable multi-GPU, uncomment and set process_count_per_instance:
        # distribution=MpiDistribution(process_count_per_instance=4),
    )
    return job


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit QLoRA SFT job to Azure ML")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llama_config.yaml",
        help="Path to the model YAML config (relative to project root)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (e.g. LLaMA). "
             "If omitted, HF_TOKEN env var is used if set.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the job spec without submitting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Resolve config path relative to project root when a relative path is given
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    print(f"[submit] Config    : {config_path}")
    print(f"[submit] Workspace : {WORKSPACE_NAME}  |  Compute: {COMPUTE_CLUSTER}")

    ml_client = _get_client()
    env_id = _get_or_create_environment(ml_client)
    job = build_job(str(config_path), env_id, hf_token=hf_token)

    if args.dry_run:
        print("\n[dry_run] Job spec:")
        print(f"  display_name : {job.display_name}")
        print(f"  environment  : {env_id}")
        print(f"  compute      : {job.compute}")
        print(f"  command      : {job.command}")
        print("[dry_run] No job submitted.")
        return

    submitted = ml_client.jobs.create_or_update(job)
    print(f"\n[submit] Job submitted successfully!")
    print(f"  Name   : {submitted.name}")
    print(f"  Status : {submitted.status}")
    print(f"  Studio : {submitted.studio_url}")


if __name__ == "__main__":
    main()
