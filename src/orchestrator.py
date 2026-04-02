from agents.text_extractor import TextExtractorAgent
from agents.syntactic_improver import SyntacticImproverAgent
from agents.content_checker import ContentCheckerAgent
from agents.image_extractor import ImageExtractorAgent
from agents.image_captioner import ImageCaptionerAgent
from agents.structure_agent import StructureAgent
from agents.md_writer import MarkdownWriterAgent

class Orchestrator:
    def __init__(self, pdf_path, output_md_path, api_keys):
        self.pdf_path = pdf_path
        self.output_md_path = output_md_path
        self.api_keys = api_keys
        self.text_extractor = TextExtractorAgent(pdf_path)
        self.syntactic_improver = SyntacticImproverAgent(api_keys)
        self.structure_agent = StructureAgent()
        self.content_checker = ContentCheckerAgent(pdf_path)
        self.image_extractor = ImageExtractorAgent(pdf_path)
        self.image_captioner = ImageCaptionerAgent(api_keys)
        self.md_writer = MarkdownWriterAgent(output_md_path)

    def run(self):
        print("Extracting text...")
        text = self.text_extractor.extract()
        print("Detecting structure...")
        structure = self.structure_agent.detect_structure(text)
        print("Improving text...")
        improved_text = self.syntactic_improver.improve(text)
        print("Extracting images and tables...")
        images, tables, diagrams = self.image_extractor.extract()
        print("Captioning images...")
        image_captions = [self.image_captioner.caption(img) for img in images]
        print("Checking content completeness and concepts...")
        self.content_checker.check(improved_text, structure=structure)
        print("Writing Markdown...")
        self.md_writer.write(improved_text, images, tables, diagrams, image_captions, structure)
        print(f"Markdown file generated at {self.output_md_path}")

if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    pdf_path = sys.argv[1]
    output_md_path = sys.argv[2]
    api_keys = {"GROQ": os.getenv("GROQ_API_KEY"), "GOOGLE": os.getenv("GOOGLE_API_KEY")}
    orchestrator = Orchestrator(pdf_path, output_md_path, api_keys)
    orchestrator.run()
