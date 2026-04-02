import fitz
import os
from PIL import Image

class ImageExtractorAgent:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract(self):
        doc = fitz.open(self.pdf_path)
        images = []
        for i, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                img_path = f"image_page{i+1}_{img_index+1}.{ext}"
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                images.append(img_path)
        doc.close()
        # Table/diagram extraction can be added with camelot or similar
        tables, diagrams = [], []
        return images, tables, diagrams
