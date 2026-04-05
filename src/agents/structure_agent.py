import re
from difflib import SequenceMatcher

SIMILARITY_THRESHOLD = 0.82  # paragraphs more similar than this are considered duplicates


class StructureAgent:
    def __init__(self):
        pass

    def detect_structure(self, pages: dict):
        """
        Takes a dict {page_num: text} and returns a deduplicated list of sections,
        each carrying its page_num so images can be placed inline.

        Returns: [{"page": int, "heading": str|None, "content": [...]}]
        """
        print(f"\n[StructureAgent] Detecting structure across {len(pages)} pages...")

        structure = []

        for page_num in sorted(pages.keys()):
            text = pages[page_num]
            lines = text.splitlines()
            current_section = {"page": page_num, "heading": None, "content": []}

            for line in lines:
                if re.match(r"^\s*([A-Z][A-Za-z0-9\s]+):?$", line):
                    if current_section["heading"] or current_section["content"]:
                        structure.append(current_section)
                    current_section = {"page": page_num, "heading": line.strip(), "content": []}
                elif re.match(r"^\s*[-*+] ", line):
                    current_section["content"].append({"type": "list_item", "text": line.strip()})
                elif line.strip():
                    current_section["content"].append({"type": "paragraph", "text": line.strip()})

            if current_section["heading"] or current_section["content"]:
                structure.append(current_section)

        before = sum(len(s["content"]) for s in structure)
        structure = self._deduplicate(structure)
        after = sum(len(s["content"]) for s in structure)

        print(f"[StructureAgent] [OK] Built {len(structure)} sections — removed {before - after} duplicate/near-duplicate paragraphs")
        return structure

    def _deduplicate(self, structure: list) -> list:
        """
        Remove consecutive paragraphs that are near-identical to any of the last
        few paragraphs seen (sliding window of 3). Keeps the longest version.
        """
        recent_paragraphs = []  # sliding window of last N paragraph texts
        window = 3

        for section in structure:
            filtered_content = []
            for item in section["content"]:
                if item["type"] != "paragraph":
                    filtered_content.append(item)
                    continue

                text = item["text"]
                is_duplicate = False
                for prev in recent_paragraphs:
                    ratio = SequenceMatcher(None, prev, text).ratio()
                    if ratio >= SIMILARITY_THRESHOLD:
                        # Keep the longer version — replace in recent if this is longer
                        if len(text) > len(prev):
                            idx = recent_paragraphs.index(prev)
                            recent_paragraphs[idx] = text
                            # Also replace in filtered_content if already written
                            for i, fc in enumerate(filtered_content):
                                if fc.get("text") == prev:
                                    filtered_content[i] = item
                        is_duplicate = True
                        break

                if not is_duplicate:
                    filtered_content.append(item)
                    recent_paragraphs.append(text)
                    if len(recent_paragraphs) > window:
                        recent_paragraphs.pop(0)

            section["content"] = filtered_content

        # Remove empty sections (no heading and no content)
        return [s for s in structure if s.get("heading") or s.get("content")]
