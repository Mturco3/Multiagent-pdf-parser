class MarkdownWriterAgent:
    def __init__(self, output_md_path):
        self.output_md_path = output_md_path

    def write(self, text, images, tables, diagrams, image_captions=None, structure=None):
        with open(self.output_md_path, "w", encoding="utf-8") as f:
            # Use structure if available
            if structure:
                for section in structure:
                    if section.get("heading"):
                        f.write(f"## {section['heading']}\n\n")
                    for item in section.get("content", []):
                        if item["type"] == "paragraph":
                            f.write(item["text"] + "\n\n")
                        elif item["type"] == "list_item":
                            f.write(f"- {item['text'][2:]}\n")
                    f.write("\n")
            else:
                f.write("# Notes\n\n")
                f.write(text + "\n\n")
            if images:
                f.write("## Images\n")
                for idx, img in enumerate(images):
                    caption = image_captions[idx] if image_captions and idx < len(image_captions) else ""
                    f.write(f"![Image]({img}) {caption}\n")
            if tables:
                f.write("## Tables\n")
                for table in tables:
                    f.write(f"{table}\n")
            if diagrams:
                f.write("## Diagrams\n")
                for diagram in diagrams:
                    f.write(f"{diagram}\n")
