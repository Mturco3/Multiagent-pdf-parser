import os
import sys
import yaml
from dotenv import load_dotenv
from src.orchestrator import Orchestrator

load_dotenv()

api_keys = {
    "GROQ": os.getenv("GROQ_API_KEY"),
    "GOOGLE": os.getenv("GOOGLE_API_KEY"),
}

if len(sys.argv) == 3:
    pdf_path = sys.argv[1]
    output_md_path = sys.argv[2]
else:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    pdf_path = config["input_pdf"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_md_path = os.path.join(output_dir, f"{pdf_name}.md")

Orchestrator(pdf_path, output_md_path, api_keys).run()
