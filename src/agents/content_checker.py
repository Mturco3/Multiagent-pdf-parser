import fitz
import nltk
from nltk.corpus import stopwords
import re

class ContentCheckerAgent:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english'))

    def check(self, markdown_text, structure=None):
        print(f"\n[ContentChecker] Checking content completeness...")
        doc = fitz.open(self.pdf_path)
        pdf_words = set()
        pdf_concepts = set()
        for page in doc:
            words = re.findall(r"\w+", page.get_text())
            pdf_words.update(w.lower() for w in words if w.lower() not in self.stopwords)
            # Simple concept extraction: use capitalized words as concepts
            pdf_concepts.update(w for w in words if w.istitle() and w.lower() not in self.stopwords)
        doc.close()
        md_words = set(re.findall(r"\w+", markdown_text.lower()))
        md_concepts = set()
        if structure:
            for section in structure:
                if section.get("heading"):
                    md_concepts.add(section["heading"].strip())
        # Word coverage
        missing = pdf_words - md_words
        if missing:
            print(f"[ContentChecker] [WARN] Missing words: {sorted(list(missing))[:20]} ...")
        else:
            print("[ContentChecker] [OK] All content words present in Markdown")
        # Concept coverage
        missing_concepts = pdf_concepts - md_concepts
        if missing_concepts:
            print(f"[ContentChecker] [WARN] Missing concepts: {sorted(list(missing_concepts))[:10]} ...")
        else:
            print("[ContentChecker] [OK] All key concepts present in Markdown")
        # Hallucination check (simple): flag words in markdown not in PDF
        hallucinated = md_words - pdf_words
        if hallucinated:
            print(f"[ContentChecker] [WARN] Potential hallucinated words: {sorted(list(hallucinated))[:10]} ...")
        print(f"[ContentChecker] [OK] Content check complete")
