import json
import os
import re
import time

from google import genai
from google.genai import types
from pydantic import ValidationError

from .models import SlideReview, SlideType, SuggestedAction, ActionType
from .normalizer import normalize

MODEL = "gemma-4-31b-it"
_MIN_INTERVAL = 4.1  # seconds between calls to stay safely under 15 RPM

_SYSTEM_PROMPT = """You are a university notes editor. You receive the text of a single lecture slide and return a structured JSON analysis.

Return a JSON object with this exact schema:
{
  "slide_type": "<type>",
  "title": "<slide title or null>",
  "is_continuation": <true|false>,
  "key_concepts": ["<concept>", ...],
  "summary": "<one-sentence summary or null>",
  "actions": [
    {
      "action": "<action_type>",
      "original_fragment": "<exact excerpt from the slide>"
    }
  ]
}

slide_type values:
- "content": normal lecture content slide
- "image_description": slide that is mainly a figure/diagram with little text
- "introduction": title slide or section intro
- "course_info": logistics, syllabus, deadlines, references

action_type values:
- "insert_connectivity": two or more consecutive sentences/bullets that could be joined with a transitional phrase into fluent prose
- "remove_personal_pronouns": sentence using "we", "you", "our", "your" that should be made impersonal
- "remove_useless_bullets": bullet that is just a concept label with no explanatory content (e.g. "• Polymorphism")
- "define_acronym": acronym used without being defined on this slide
- "incomplete_sentence": fragment that lacks a verb or enough context to be understood standalone

Rules:
- title: extract verbatim from the slide if present, otherwise null
- is_continuation: true if this slide clearly continues from a previous one (e.g. starts with "cont.", "moreover", mid-sentence, etc.)
- key_concepts: list of main technical concepts introduced or used (empty list if none)
- summary: one sentence for content slides, null for introduction/course_info/image_description
- actions: only flag issues that are clearly present. Empty list if none.
- For insert_connectivity: only merge bullets that are logically sequential or causally linked.
- Return ONLY the JSON object, no markdown, no explanation."""


def _parse_response(raw: str, slide_number: int) -> SlideReview:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = raw.rstrip('`').strip()

    data = json.loads(raw)

    # Gemma sometimes returns a bare array — treat it as the actions list
    action_items = data if isinstance(data, list) else data.get("actions", [])

    actions = []
    for item in action_items:
        try:
            actions.append(SuggestedAction(**item))
        except (ValidationError, KeyError):
            continue

    return SlideReview(
        slide_number=slide_number,
        slide_type=SlideType(data.get("slide_type", "content")),
        title=data.get("title"),
        is_continuation=bool(data.get("is_continuation", False)),
        key_concepts=data.get("key_concepts", []),
        summary=data.get("summary"),
        actions=actions,
    )


class LLMChecker:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        self._client = genai.Client(api_key=api_key)

    def _check_one(self, slide_number: int, raw_text: str) -> SlideReview:
        normalized_text = normalize(raw_text)

        if len(normalized_text.strip()) < 20:
            print(f"  [slide {slide_number:03d}] skipped (empty)")
            return SlideReview(
                slide_number=slide_number,
                slide_type=SlideType.INTRODUCTION,
                title=None,
                is_continuation=False,
                key_concepts=[],
                summary=None,
                actions=[],
            )

        print(f"  [slide {slide_number:03d}] sending request...")
        for attempt in range(1, 4):
            try:
                response = self._client.models.generate_content(
                    model=MODEL,
                    contents=f"Slide text:\n\n{normalized_text}",
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                    ),
                )
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  [slide {slide_number:03d}] ERROR after 3 attempts — skipping: {e}")
                    return SlideReview(
                        slide_number=slide_number,
                        slide_type=SlideType.CONTENT,
                        title=None,
                        is_continuation=False,
                        key_concepts=[],
                        summary=None,
                        actions=[],
                    )
                print(f"  [slide {slide_number:03d}] attempt {attempt} failed, retrying in 10s...")
                time.sleep(10)

        review = _parse_response(response.text, slide_number)
        title_str = f'"{review.title}"' if review.title else "no title"
        actions_str = f"{len(review.actions)} action(s)" if review.actions else "no actions"
        print(f"  [slide {slide_number:03d}] done — {review.slide_type.value}, {title_str}, {actions_str}")
        return review

    def check_all(self, slides: list[tuple[int, str]]) -> list[SlideReview]:
        """Process all slides sequentially with throttle."""
        results = []
        total = len(slides)
        for num, text in slides:
            print(f"  [{num}/{total}]", end=" ", flush=True)
            results.append(self._check_one(num, text))
            time.sleep(_MIN_INTERVAL)
        return results
