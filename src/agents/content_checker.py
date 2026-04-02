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
            print(f"Warning: Missing words in Markdown: {sorted(list(missing))[:20]} ...")
        else:
            print("All content words are present in the Markdown.")
        # Concept coverage
        missing_concepts = pdf_concepts - md_concepts
        if missing_concepts:
            print(f"Warning: Missing concepts in Markdown: {sorted(list(missing_concepts))[:10]} ...")
        else:
            print("All key concepts are present in the Markdown.")
        # Hallucination check (simple): flag words in markdown not in PDF
        hallucinated = md_words - pdf_words
        if hallucinated:
            print(f"Potential hallucinated words in Markdown: {sorted(list(hallucinated))[:10]} ...")
