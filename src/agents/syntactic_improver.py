import os
import re
import time
import requests


class SyntacticImproverAgent:
    def __init__(self, api_keys):
        self.api_key = api_keys.get("GROQ")
        self.model = "llama-3.3-70b-versatile"
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.rules = self._load_rules()

    def _load_rules(self):
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompt.md")
        with open(os.path.normpath(prompt_path), "r", encoding="utf-8") as f:
            return f.read()

    def improve(self, pages: dict) -> dict:
        """
        Takes a dict {page_num: text} and returns {page_num: improved_text}.
        Sends all pages in a single request with [PAGE N] markers so the model
        has full document context. Markers are preserved in the response.
        """
        combined = ""
        for page_num in sorted(pages.keys()):
            combined += f"[PAGE {page_num}]\n{pages[page_num]}\n"

        system_prompt = (
            "You are a transcription engine. "
            "You output ONLY the transcribed text, nothing else. "
            "You never explain, plan, reason, self-correct, or comment. "
            "You never use bullet points, dashes, or asterisks for emphasis. "
            "Your response begins immediately with the first [PAGE N] marker and contains nothing else besides transcribed content and [PAGE N] markers."
        )

        user_prompt = (
            f"{self.rules}\n\n"
            "Transcribe the lecture slides below. "
            "Skip page 1 (title slide). "
            "Start your response with [PAGE 2]. "
            "Keep every [PAGE N] marker on its own line exactly as written. "
            "Output nothing except the transcribed text and the [PAGE N] markers.\n\n"
            f"{combined}"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        print(f"\n[SyntacticImprover | {self.model}] Sending {len(pages)} pages to model for improvement...")

        for attempt in range(5):
            try:
                response = requests.post(self.api_url, json=data, headers=headers)
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(f"Failed to connect to Groq API: {e}") from e

            if response.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"[SyntacticImprover | {self.model}] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            break

        resp_json = response.json()
        if "choices" not in resp_json:
            raise RuntimeError(f"Unexpected API response: {resp_json}")

        raw = resp_json["choices"][0]["message"]["content"]
        raw = self._strip_reasoning(raw)
        result = self._parse_pages(raw, pages)
        print(f"[SyntacticImprover | {self.model}] [OK] Improved {len(result)} pages (page 1 skipped as title slide)")
        return result

    def _strip_reasoning(self, text: str) -> str:
        """Remove anything before the first [PAGE N] marker to discard model reasoning."""
        match = re.search(r"\[PAGE \d+\]", text)
        if match:
            return text[match.start():]
        return text

    def _remove_duplicate_sentences(self, text: str) -> str:
        """Remove duplicate consecutive sentences, keeping the first occurrence."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen = []
        for sentence in sentences:
            normalized = sentence.strip().lower()
            if normalized not in [s.strip().lower() for s in seen]:
                seen.append(sentence)
        return ' '.join(seen)

    def _parse_pages(self, raw: str, original_pages: dict) -> dict:
        """Split the model response back into per-page chunks using [PAGE N] markers."""
        result = {}
        parts = re.split(r"\[PAGE (\d+)\]", raw)
        it = iter(parts[1:])
        for page_num_str, content in zip(it, it):
            result[int(page_num_str)] = self._remove_duplicate_sentences(content.strip())
        if not result:
            return original_pages
        return result
