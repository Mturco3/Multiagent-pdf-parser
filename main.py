import os
import sys
import yaml
from src.transcriber import Transcriber

if len(sys.argv) == 2:
    pdf_path = sys.argv[1]
else:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    pdf_path = config["input_pdf"]

Transcriber(pdf_path).run()
