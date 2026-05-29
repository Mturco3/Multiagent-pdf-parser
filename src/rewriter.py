import time

from pydantic_ai import Agent

from .checker import MODEL, WINDOW_SECONDS
from .models import SlideReview, SlideRewrite, SlideType
from .normalizer import normalize
from .utilities.prompts import REWRITER_SYSTEM_PROMPT

rewriter_agent = Agent(MODEL, output_type=str, instructions=REWRITER_SYSTEM_PROMPT, model_settings={"temperature": 0})


class LLMRewriter:
    """Rewrite slide text while preserving structure and content."""

    def build_introduction_output(self, review: SlideReview) -> SlideRewrite | None:
        """Return a heading-only output for introduction slides when possible."""
        if not review.title:
            return None

        return SlideRewrite(slide_number=review.slide_number, slide_type=review.slide_type.value, title=review.title, is_continuation=False, text="")

    def extract_body_text(self, slide_text: str, review: SlideReview) -> str:
        """Prepare deterministic slide body text without title or trailing page number noise."""
        lines = [line.strip() for line in normalize(slide_text).splitlines() if line.strip()]

        if review.title and lines and lines[0] == review.title:
            lines = lines[1:]

        if lines and lines[-1].isdigit():
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def build_passthrough_output(self, slide_text: str, review: SlideReview) -> SlideRewrite:
        """Return deterministic output when no approved rewrite should be performed."""
        return SlideRewrite(
            slide_number=review.slide_number,
            slide_type=review.slide_type.value,
            title=review.title,
            is_continuation=review.is_continuation,
            text=self.extract_body_text(slide_text, review),
        )

    def rewrite_one(self, slide_text: str, review: SlideReview, previous_paragraph: str | None = None) -> SlideRewrite | None:
        """Rewrite one slide or skip it when the slide should not be sent to the LLM."""
        if review.slide_type in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION):
            print(f"[slide {review.slide_number:03d}] skipped ({review.slide_type.value})")
            return None

        if review.slide_type == SlideType.INTRODUCTION:
            print(f"[slide {review.slide_number:03d}] heading only ({review.slide_type.value})")
            return self.build_introduction_output(review)

        if not review.reviewer_approved:
            print(f"[slide {review.slide_number:03d}] review not approved; preserving extracted body text")
            return self.build_passthrough_output(slide_text, review)

        if not review.actions:
            print(f"[slide {review.slide_number:03d}] no approved actions; preserving extracted body text")
            return self.build_passthrough_output(slide_text, review)

        body_text = self.extract_body_text(slide_text, review)
        actions_text = "\n".join(f"- {action.action.value}: \"{action.original_fragment}\"" for action in review.actions)

        prompt = f"Slide title: {review.title or '(no title)'}\n\nOriginal slide text:\n\n{body_text}\n\nActions to apply:\n{actions_text}"
        if previous_paragraph:
            prompt += f"\n\nPrevious paragraph (for transition context only - do NOT include this content in your output):\n{previous_paragraph}"

        print(f"[slide {review.slide_number:03d}] rewriting ({len(review.actions)} approved action(s))...")
        agent_result = rewriter_agent.run_sync(prompt)
        print(f"[slide {review.slide_number:03d}] done")

        return SlideRewrite(
            slide_number=review.slide_number,
            slide_type=review.slide_type.value,
            title=review.title,
            is_continuation=review.is_continuation,
            text=agent_result.output,
        )

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

            should_call_llm = (
                review.slide_type not in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION, SlideType.INTRODUCTION)
                and review.reviewer_approved
                and bool(review.actions)
            )
            if should_call_llm and request_count >= 14 and time.time() - batch_start < WINDOW_SECONDS:
                remaining = WINDOW_SECONDS - (time.time() - batch_start)
                print(f"Rate limit reached - waiting {remaining:.1f}s...")
                time.sleep(remaining)
                request_count = 0
                batch_start = time.time()

            rewrite = self.rewrite_one(text, review, transition_paragraph if should_call_llm else None)

            if should_call_llm:
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
