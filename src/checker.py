import time

from pydantic_ai import Agent

from .models import SlideReview, SlideReviewResponse, SlideType
from .normalizer import normalize
from .utilities.prompts import CHECKER_SYSTEM_PROMPT

MODEL = "google:gemini-3.1-flash-lite"
WINDOW_SECONDS = 60

checker_agent = Agent(MODEL, output_type=SlideReviewResponse, instructions=CHECKER_SYSTEM_PROMPT, model_settings={"temperature": 0})


class LLMChecker:
    """Sends each slide's text to the LLM and collects structured reviews."""

    def check_one(self, slide_number: int, raw_text: str) -> SlideReview:
        """Review a single slide's text through the checker agent."""
        normalized_text = normalize(raw_text)

        # Skip near-empty slides without calling the LLM
        if len(normalized_text.strip()) < 10:
            print(f"[slide {slide_number:03d}] skipped (empty)")
            return SlideReview(slide_number=slide_number, slide_type=SlideType.INTRODUCTION, title=None, is_continuation=False, key_concepts=[], summary=None, actions=[])

        print(f"[slide {slide_number:03d}] sending request...")
        agent_result = checker_agent.run_sync(f"Slide text:\n\n{normalized_text}")
        review = SlideReview(slide_number=slide_number, **agent_result.output.model_dump())

        title_display = f'"{review.title}"' if review.title else "no title"
        actions_display = f"{len(review.actions)} action(s)" if review.actions else "no actions"
        print(f"[slide {slide_number:03d}] done — {review.slide_type.value}, {title_display}, {actions_display}")
        return review

    def check_all(self, slides: list[tuple[int, str]]) -> list[SlideReview]:
        """Review all slides, pausing after every 14 requests to stay under the API rate limit."""
        reviews = []
        total = len(slides)
        request_count = 0
        batch_start = time.time()

        for slide_number, text in slides:
            print(f"[{slide_number}/{total}]", end=" ", flush=True)

            # Pause when the batch limit is reached within the rate window
            if request_count >= 14 and time.time() - batch_start < WINDOW_SECONDS:
                remaining = WINDOW_SECONDS - (time.time() - batch_start)
                print(f"Rate limit reached — waiting {remaining:.1f}s...")
                time.sleep(remaining)
                request_count = 0
                batch_start = time.time()

            reviews.append(self.check_one(slide_number, text))
            request_count += 1

        return reviews
