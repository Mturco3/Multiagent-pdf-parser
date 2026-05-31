import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import Pipeline


def main():
    """Load configuration and run the transcript pipeline."""
    if len(sys.argv) == 2:
        pdf_path = sys.argv[1]
    else:
        with open("config.yaml", "r", encoding="utf-8") as file_handle:
            config = yaml.safe_load(file_handle)
        pdf_path = config["input_pdf"]

    pipeline = Pipeline(pdf_path)
    pipeline.run()


if __name__ == "__main__":
    main()
