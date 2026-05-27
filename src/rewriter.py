import time

from pydantic_ai import Agent

from .checker import MODEL, WINDOW_SECONDS
from .models import SlideReview, SlideType
from .utilities.prompts import REWRITER_SYSTEM_PROMPT

rewriter_agent = Agent(MODEL, output_type=str, instructions=REWRITER_SYSTEM_PROMPT, model_settings={"temperature": 0})


class LLMRewriter:
    """Rewrites slide text applying review actions, then concatenates into a single document."""

    def rewrite_one(self, slide_text: str, review: SlideReview, previous_paragraph: str | None = None) -> str | None:
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
            prompt += f"\n\nPrevious paragraph (for context and flow):\n{previous_paragraph}"

        print(f"[slide {review.slide_number:03d}] rewriting ({len(review.actions)} action(s))...")
        agent_result = rewriter_agent.run_sync(prompt)
        print(f"[slide {review.slide_number:03d}] done")
        return agent_result.output

    def rewrite_all(self, slides: list[tuple[int, str]], reviews: list[SlideReview]) -> str:
        """Rewrite all content slides and concatenate them into a single markdown document."""
        review_by_number = {review.slide_number: review for review in reviews}
        rewritten_slides: list[tuple[SlideReview, str]] = []
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

            rewritten_text = self.rewrite_one(text, review, previous_paragraph)
            request_count += 1

            if rewritten_text is not None:
                rewritten_slides.append((review, rewritten_text))
                previous_paragraph = rewritten_text

        return self.concatenate(rewritten_slides)

    def concatenate(self, slides: list[tuple[SlideReview, str]]) -> str:
        """Join rewritten slides into one document, deduplicating consecutive titles."""
        parts: list[str] = []
        previous_title: str | None = None

        for review, text in slides:
            section_parts: list[str] = []

            if review.is_continuation:
                section_parts.append(text)
            else:
                if review.title and review.title != previous_title:
                    section_parts.append(f"## {review.title}\n")
                    previous_title = review.title
                section_parts.append(text)

            parts.append("\n".join(section_parts))

        return "\n\n".join(parts)
