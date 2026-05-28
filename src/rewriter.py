import time

from pydantic_ai import Agent

from .checker import MODEL, WINDOW_SECONDS
from .models import SlideReview, SlideRewrite, SlideType
from .utilities.prompts import REWRITER_SYSTEM_PROMPT

rewriter_agent = Agent(MODEL, output_type=str, instructions=REWRITER_SYSTEM_PROMPT, model_settings={"temperature": 0})


class LLMRewriter:
    """Rewrite slide text while preserving structure and content."""

    def build_introduction_output(self, review: SlideReview) -> SlideRewrite | None:
        """Return a heading-only output for introduction slides when possible."""
        if not review.title:
            return None

        return SlideRewrite(slide_number=review.slide_number, slide_type=review.slide_type.value, title=review.title, is_continuation=False, text="")

    def rewrite_one(self, slide_text: str, review: SlideReview, previous_paragraph: str | None = None) -> SlideRewrite | None:
        """Rewrite one slide or skip it when the slide should not be sent to the LLM."""
        if review.slide_type in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION):
            print(f"[slide {review.slide_number:03d}] skipped ({review.slide_type.value})")
            return None

        if review.slide_type == SlideType.INTRODUCTION:
            print(f"[slide {review.slide_number:03d}] heading only ({review.slide_type.value})")
            return self.build_introduction_output(review)

        if review.actions:
            actions_text = "\n".join(f"- {action.action.value}: \"{action.original_fragment}\"" for action in review.actions)
        else:
            actions_text = "None"

        prompt = f"Slide title: {review.title or '(no title)'}\n\nOriginal slide text:\n\n{slide_text}\n\nActions to apply:\n{actions_text}"
        if previous_paragraph:
            prompt += f"\n\nPrevious paragraph (for transition context only - do NOT include this content in your output):\n{previous_paragraph}"

        print(f"[slide {review.slide_number:03d}] rewriting ({len(review.actions)} action(s))...")
        agent_result = rewriter_agent.run_sync(prompt)
        print(f"[slide {review.slide_number:03d}] done")

        return SlideRewrite(slide_number=review.slide_number, slide_type=review.slide_type.value, title=review.title, is_continuation=review.is_continuation, text=agent_result.output)

    def rewrite_all(self, slides: list[tuple[int, str]], reviews: list[SlideReview]) -> list[SlideRewrite]:
        """Rewrite all slides that should appear in the final document."""
        review_by_number = {review.slide_number: review for review in reviews}
        results: list[SlideRewrite] = []
        request_count = 0
        batch_start = time.time()
        previous_paragraph: str | None = None
        current_section_title: str | None = None
        total = len(slides)

        for slide_number, text in slides:
            review = review_by_number[slide_number]
            print(f"[{slide_number}/{total}]", end=" ", flush=True)

            transition_paragraph = previous_paragraph
            starts_new_section = bool(review.title and review.title != current_section_title and not review.is_continuation)
            if starts_new_section:
                transition_paragraph = None

            if review.slide_type not in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION, SlideType.INTRODUCTION):
                if request_count >= 14 and time.time() - batch_start < WINDOW_SECONDS:
                    remaining = WINDOW_SECONDS - (time.time() - batch_start)
                    print(f"Rate limit reached - waiting {remaining:.1f}s...")
                    time.sleep(remaining)
                    request_count = 0
                    batch_start = time.time()

            rewrite = self.rewrite_one(text, review, transition_paragraph)

            if review.slide_type not in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION, SlideType.INTRODUCTION):
                request_count += 1

            if rewrite is None:
                continue

            results.append(rewrite)

            if rewrite.title and not rewrite.is_continuation:
                current_section_title = rewrite.title

            if rewrite.text:
                previous_paragraph = rewrite.text
            else:
                previous_paragraph = None

        return results
