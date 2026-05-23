import sys
import yaml
from dotenv import load_dotenv
from src.pipeline import Pipeline

load_dotenv()

if len(sys.argv) == 2:
    pdf_path = sys.argv[1]
else:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    pdf_path = config["input_pdf"]

Pipeline(pdf_path).run()
