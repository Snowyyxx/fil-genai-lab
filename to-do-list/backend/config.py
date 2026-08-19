"""Runtime configuration for Meowstermind, read from the environment.

Everything has a sane default so `uvicorn main:app --reload` works with zero setup.
AWS credentials themselves are never read here - boto3 resolves them from the
standard chain (AWS_PROFILE, env vars, ~/.aws/credentials, instance role).
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # optional convenience: load backend/.env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # --- AWS / Bedrock -----------------------------------------------------
    aws_region: str = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

    # Leave BEDROCK_MODEL_ID unset to let the app discover an enabled open-weight
    # model in your account on first use (see zen_cat.CatBrain.resolve_model_id).
    model_id: str | None = os.getenv("BEDROCK_MODEL_ID") or None

    max_tokens: int = int(os.getenv("BEDROCK_MAX_TOKENS", "3000"))

    # Skip Bedrock entirely and always answer from the offline cat.
    force_mock: bool = _flag("MEOW_FORCE_MOCK")

    # --- App ---------------------------------------------------------------
    data_file: Path = Path(os.getenv("MEOW_DATA_FILE", Path(__file__).with_name("tasks.json")))
    allow_origins: list[str] = [
        o.strip() for o in os.getenv("MEOW_ALLOW_ORIGINS", "*").split(",") if o.strip()
    ]
    # Serve ../frontend from the API process (handy locally, required in Docker).
    frontend_dir: Path = Path(
        os.getenv("MEOW_FRONTEND_DIR", Path(__file__).resolve().parent.parent / "frontend")
    )


settings = Settings()
