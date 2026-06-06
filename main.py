import argparse
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.utilities.rate_limit import DailyQuotaExceededError


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments while tolerating unquoted Windows paths."""
    parser = argparse.ArgumentParser(description="Convert PDF lecture slides into Markdown notes.")
    parser.add_argument("pdf_path", nargs="*", help="Path to the input PDF. Quote paths that contain spaces.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear this PDF's cache before running.")
    return parser.parse_args()


def main():
    """Load configuration and run the transcript pipeline."""
    args = parse_args()
    if args.pdf_path:
        pdf_path = " ".join(args.pdf_path).strip()
        clear_cache = args.clear_cache
    else:
        config_path = "config.yaml"
        with open(config_path, "r", encoding="utf-8") as file_handle:
            config = yaml.safe_load(file_handle)
        pdf_path = config["input_pdf"]
        clear_cache = args.clear_cache or bool(config.get("clear_cache_once", False))
        if clear_cache:
            config["clear_cache_once"] = False
            with open(config_path, "w", encoding="utf-8") as file_handle:
                yaml.safe_dump(config, file_handle, sort_keys=False, allow_unicode=True)

    from src.pipeline import Pipeline

    pipeline = Pipeline(pdf_path, clear_cache=clear_cache)
    pipeline.run()


if __name__ == "__main__":
    try:
        main()
    except DailyQuotaExceededError as error:
        print(f"[STOP] {error}")
        raise SystemExit(1)
