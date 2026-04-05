import os


class MarkdownWriterAgent:
    def __init__(self, output_md_path):
        self.output_md_path = output_md_path

    def write(self, pages: dict, images_by_page: dict, structure=None):
        print(f"\n[MarkdownWriter] Writing Markdown to '{os.path.basename(self.output_md_path)}'...")
        sections_written = 0
        images_written = 0

        with open(self.output_md_path, "w", encoding="utf-8") as f:
            if structure:
                seen_headings = set()
                written_image_pages = set()

                for section in structure:
                    page_num = section.get("page")
                    heading = section.get("heading")

                    if heading:
                        normalized = heading.strip().lower()
                        if normalized not in seen_headings:
                            seen_headings.add(normalized)
                            f.write(f"## {heading}\n\n")
                            sections_written += 1

                    for item in section.get("content", []):
                        if item["type"] == "paragraph":
                            f.write(item["text"] + "\n\n")
                        elif item["type"] == "list_item":
                            f.write(f"- {item['text'][2:]}\n")

                    if page_num and page_num not in written_image_pages:
                        page_imgs = images_by_page.get(page_num, [])
                        for img_path in page_imgs:
                            f.write(f"\n![Image]({img_path})\n")
                            images_written += 1
                        if page_imgs:
                            written_image_pages.add(page_num)

                    f.write("\n")
            else:
                f.write("# Notes\n\n")
                for page_num in sorted(pages.keys()):
                    f.write(pages[page_num] + "\n\n")
                    for img_path in images_by_page.get(page_num, []):
                        f.write(f"\n![Image]({img_path})\n")
                        images_written += 1

        print(f"[MarkdownWriter] [OK] Written {sections_written} sections and {images_written} image references")
