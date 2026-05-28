import time

from pydantic_ai import Agent

from .checker import MODEL, WINDOW_SECONDS
from .models import SlideReview, SlideRewrite, SlideType
from .utilities.prompts import REWRITER_SYSTEM_PROMPT

rewriter_agent = Agent(MODEL, output_type=str, instructions=REWRITER_SYSTEM_PROMPT, model_settings={"temperature": 0})


class LLMRewriter:
    """Rewrites slide text applying review actions, producing per-slide JSON artifacts."""

    def rewrite_one(self, slide_text: str, review: SlideReview, previous_paragraph: str | None = None) -> SlideRewrite | None:
        """Rewrite a single slide applying its review actions, or skip non-content slides."""
        if review.slide_type in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION):
            print(f"[slide {review.slide_number:03d}] skipped ({review.slide_type.value})")
            return None

        # Format actions list for the prompt
        if review.actions:
            actions_text = "\n".join(f"- {action.action.value}: \"{action.original_fragment}\"" for action in review.actions)
        else:
            actions_text = "None"

        prompt = f"Slide title: {review.title or '(no title)'}\n\nOriginal slide text:\n\n{slide_text}\n\nActions to apply:\n{actions_text}"

        if previous_paragraph:
            prompt += f"\n\nPrevious paragraph (for transition context only — do NOT include this content in your output):\n{previous_paragraph}"

        print(f"[slide {review.slide_number:03d}] rewriting ({len(review.actions)} action(s))...")
        agent_result = rewriter_agent.run_sync(prompt)
        print(f"[slide {review.slide_number:03d}] done")

        return SlideRewrite(
            slide_number=review.slide_number,
            slide_type=review.slide_type.value,
            title=review.title,
            is_continuation=review.is_continuation,
            text=agent_result.output
        )

    def rewrite_all(self, slides: list[tuple[int, str]], reviews: list[SlideReview]) -> list[SlideRewrite]:
        """Rewrite all content slides, returning per-slide JSON artifacts."""
        review_by_number = {review.slide_number: review for review in reviews}
        results: list[SlideRewrite] = []
        request_count = 0
        batch_start = time.time()
        previous_paragraph: str | None = None

        for slide_number, text in slides:
            review = review_by_number[slide_number]
            total = len(slides)
            print(f"[{slide_number}/{total}]", end=" ", flush=True)

            if review.slide_type in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION):
                self.rewrite_one(text, review)
                continue

            # Pause when the batch limit is reached within the rate window
            if request_count >= 14 and time.time() - batch_start < WINDOW_SECONDS:
                remaining = WINDOW_SECONDS - (time.time() - batch_start)
                print(f"Rate limit reached — waiting {remaining:.1f}s...")
                time.sleep(remaining)
                request_count = 0
                batch_start = time.time()

            rewrite = self.rewrite_one(text, review, previous_paragraph)
            request_count += 1

            if rewrite is not None:
                results.append(rewrite)
                previous_paragraph = rewrite.text

        return results
