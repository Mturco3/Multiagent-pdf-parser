"""Provide the command-line entry point for the lecture-note pipeline.

The module parses command-line arguments, loads optional YAML configuration,
runs ``Pipeline``, and clears the one-time cache flag after a successful run.
"""

import argparse
import sys
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.utilities.rate_limit import DailyQuotaExceededError

CONFIG_PATH = "config.yaml"
Config = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments while tolerating unquoted Windows paths."""
    parser = argparse.ArgumentParser(
        description="Convert PDF lecture slides into Markdown notes."
    )
    parser.add_argument(
        "pdf_path",
        nargs="*",
        help="Path to the input PDF. Quote paths that contain spaces."
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear this PDF's cache before running."
    )
    return parser.parse_args()


def load_config(filepath: str) -> Config:
    """Load the YAML configuration used when no PDF path is provided."""
    with open(filepath, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def disable_one_time_cache_clear(filepath: str, config: Config) -> None:
    """Persist completion of the configured one-time cache reset."""
    config["clear_cache_once"] = False
    with open(filepath, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(
            config,
            file_handle,
            sort_keys=False,
            allow_unicode=True
        )


def main() -> None:
    """Load configuration and run the transcript pipeline."""
    args = parse_args()
    config: Config | None = None
    clear_cache_once = False

    if args.pdf_path:
        pdf_path = " ".join(args.pdf_path).strip()
        clear_cache = args.clear_cache
    else:
        config = load_config(CONFIG_PATH)
        pdf_path = config["input_pdf"]
        clear_cache_once = bool(config.get("clear_cache_once", False))
        clear_cache = args.clear_cache or clear_cache_once

    # Import after loading .env because model settings are module constants.
    from src.pipeline import Pipeline

    pipeline = Pipeline(pdf_path, clear_cache=clear_cache)
    pipeline.run()

    if config is not None and clear_cache_once:
        disable_one_time_cache_clear(CONFIG_PATH, config)


if __name__ == "__main__":
    try:
        main()
    except DailyQuotaExceededError as error:
        print(f"[STOP] {error}")
        raise SystemExit(1)
