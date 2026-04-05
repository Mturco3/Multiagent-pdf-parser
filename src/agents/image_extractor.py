import fitz
import os

# Images smaller than this in either dimension are considered logos/icons
MIN_IMAGE_SIZE_PX = 100

class ImageExtractorAgent:
    def __init__(self, pdf_path, output_dir=None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir

    def extract(self):
        """
        Returns a dict {page_num: [img_path, ...]} with useless images removed.
        Useless images are: images on page 1 (title/logo slide) and images
        smaller than MIN_IMAGE_SIZE_PX in either dimension.
        """
        doc = fitz.open(self.pdf_path)
        pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        base_dir = self.output_dir if self.output_dir else os.path.dirname(self.pdf_path)
        img_dir = os.path.join(base_dir, f"{pdf_name}_images")
        os.makedirs(img_dir, exist_ok=True)

        print(f"\n[ImageExtractor] Extracting images from '{os.path.basename(self.pdf_path)}'...")
        print(f"[ImageExtractor] Saving images to '{img_dir}'")
        images_by_page = {}

        for i, page in enumerate(doc):
            page_num = i + 1
            page_images = []

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip page 1 (title slide with university logos)
                if page_num == 1:
                    continue

                # Skip small images (logos, icons, decorative elements)
                if width < MIN_IMAGE_SIZE_PX or height < MIN_IMAGE_SIZE_PX:
                    continue

                img_path = os.path.join(img_dir, f"page{page_num}_{img_index + 1}.{ext}")
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                page_images.append(img_path)

            if page_images:
                images_by_page[page_num] = page_images

        doc.close()
        total = sum(len(v) for v in images_by_page.values())
        print(f"[ImageExtractor] [OK] Extracted {total} images across {len(images_by_page)} pages")
        return images_by_page
