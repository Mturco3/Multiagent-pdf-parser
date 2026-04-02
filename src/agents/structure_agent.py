import re

class StructureAgent:
    def __init__(self):
        pass

    def detect_structure(self, text):
        """
        Detects sections, headings, paragraphs, lists, and captions in the extracted text.
        Returns a structured representation (dict or list of sections).
        """
        structure = []
        lines = text.splitlines()
        current_section = {"heading": None, "content": []}
        for line in lines:
            if re.match(r"^\s*([A-Z][A-Za-z0-9\s]+):?$", line):
                if current_section["heading"] or current_section["content"]:
                    structure.append(current_section)
                current_section = {"heading": line.strip(), "content": []}
            elif re.match(r"^\s*[-*+] ", line):
                current_section["content"].append({"type": "list_item", "text": line.strip()})
            elif line.strip():
                current_section["content"].append({"type": "paragraph", "text": line.strip()})
        if current_section["heading"] or current_section["content"]:
            structure.append(current_section)
        return structure
