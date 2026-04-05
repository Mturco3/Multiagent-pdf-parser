import base64
import os
import requests


class ImageCheckerAgent:
    def __init__(self, api_keys):
        self.api_key = api_keys.get("GROQ")
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def filter(self, images_by_page: dict) -> dict:
        """
        Takes a dict {page_num: [img_path, ...]} and returns a filtered version
        keeping only images that are useful for study notes.
        """
        total = sum(len(imgs) for imgs in images_by_page.values())
        print(f"\n[ImageChecker | {self.model}] Evaluating {total} extracted images...")

        filtered = {}
        kept = 0
        discarded = 0

        for page_num, img_paths in images_by_page.items():
            useful = []
            for img_path in img_paths:
                name = os.path.basename(img_path)
                print(f"[ImageChecker | {self.model}] Checking {name}...", end=" ", flush=True)
                try:
                    if self._is_useful(img_path):
                        print("[OK] kept")
                        useful.append(img_path)
                        kept += 1
                    else:
                        print("[SKIP] discarded (not useful)")
                        discarded += 1
                except Exception as e:
                    print(f"[WARN] error ({e}), keeping by default")
                    useful.append(img_path)
                    kept += 1
            if useful:
                filtered[page_num] = useful

        print(f"[ImageChecker | {self.model}] [OK] Done — kept {kept}, discarded {discarded} out of {total} images")
        return filtered

    def _is_useful(self, img_path: str) -> bool:
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        ext = os.path.splitext(img_path)[1].lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Is this image useful for university lecture notes? "
                                "Answer 'yes' if it is a diagram, chart, graph, equation, figure, or table that explains a concept. "
                                "Answer 'no' if it is a decorative background, watermark, logo, blank area, or pure text already captured in the notes. "
                                "Reply with only 'yes' or 'no'."
                            ),
                        },
                    ],
                }],
                "max_tokens": 5,
            },
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip().lower()
        return answer.startswith("y")
