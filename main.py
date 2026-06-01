import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import Pipeline
from src.utilities.rate_limit import DailyQuotaExceededError


def main():
    """Load configuration and run the transcript pipeline."""
    if len(sys.argv) >= 2:
        pdf_path = " ".join(sys.argv[1:]).strip()
        clear_cache = False
    else:
        config_path = "config.yaml"
        with open(config_path, "r", encoding="utf-8") as file_handle:
            config = yaml.safe_load(file_handle)
        pdf_path = config["input_pdf"]
        clear_cache = bool(config.get("clear_cache_once", False))
        if clear_cache:
            config["clear_cache_once"] = False
            with open(config_path, "w", encoding="utf-8") as file_handle:
                yaml.safe_dump(config, file_handle, sort_keys=False, allow_unicode=True)

    pipeline = Pipeline(pdf_path, clear_cache=clear_cache)
    pipeline.run()


if __name__ == "__main__":
    try:
        main()
    except DailyQuotaExceededError as error:
        print(f"[STOP] {error}")
        raise SystemExit(1)
