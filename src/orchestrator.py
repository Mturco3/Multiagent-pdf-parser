from src.agents.text_extractor import TextExtractorAgent
from src.agents.syntactic_improver import SyntacticImproverAgent
from src.agents.content_checker import ContentCheckerAgent
from src.agents.image_extractor import ImageExtractorAgent
from src.agents.image_checker import ImageCheckerAgent
from src.agents.structure_agent import StructureAgent
from src.agents.md_writer import MarkdownWriterAgent


class Orchestrator:
    def __init__(self, pdf_path, output_md_path, api_keys):
        self.pdf_path = pdf_path
        self.output_md_path = output_md_path
        self.api_keys = api_keys
        self.text_extractor = TextExtractorAgent(pdf_path)
        self.syntactic_improver = SyntacticImproverAgent(api_keys)
        self.structure_agent = StructureAgent()
        self.content_checker = ContentCheckerAgent(pdf_path)
        self.image_extractor = ImageExtractorAgent(pdf_path, output_dir=__import__('os').path.dirname(output_md_path))
        self.image_checker = ImageCheckerAgent(api_keys)
        self.md_writer = MarkdownWriterAgent(output_md_path)

    def run(self):
        import os
        print("=" * 60)
        print("  University Notes Transcript Agent")
        print(f"  Input : {self.pdf_path}")
        print(f"  Output: {self.output_md_path}")
        print("=" * 60)

        # Step 1 — Text extraction
        pages = self.text_extractor.extract()

        # Step 2 — LLM-based syntactic improvement
        improved_pages = self.syntactic_improver.improve(pages)

        # Step 3 — Structure detection + deduplication
        structure = self.structure_agent.detect_structure(improved_pages)

        # Step 4 — Image extraction
        images_by_page = self.image_extractor.extract()

        # Step 5 — Image usefulness check
        images_by_page = self.image_checker.filter(images_by_page)

        # Step 6 — Content completeness check
        full_text = "\n".join(improved_pages.values())
        self.content_checker.check(full_text, structure=structure)

        # Step 7 — Markdown writing
        self.md_writer.write(improved_pages, images_by_page, structure)

        print("\n" + "=" * 60)
        print(f"  [OK] Done! Output: {self.output_md_path}")
        print("=" * 60)


if __name__ == "__main__":
    import os
    import sys
    import yaml
    from dotenv import load_dotenv
    load_dotenv()

    api_keys = {"GROQ": os.getenv("GROQ_API_KEY"), "GOOGLE": os.getenv("GOOGLE_API_KEY")}

    if len(sys.argv) == 3:
        pdf_path = sys.argv[1]
        output_md_path = sys.argv[2]
    else:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        with open(os.path.normpath(config_path), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        pdf_path = config["input_pdf"]
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_md_path = os.path.join(output_dir, f"{pdf_name}.md")

    orchestrator = Orchestrator(pdf_path, output_md_path, api_keys)
    orchestrator.run()
