import unittest
from unittest.mock import patch

from bwa_backend import (
    Plan,
    Task,
    _clean_evidence,
    generate_and_place_images,
    sanitize_markdown_for_user_settings,
    resolve_router_mode,
)


class GenerationRulesTests(unittest.TestCase):
    def test_closed_book_routing_for_fundamental_topic(self):
        mode = resolve_router_mode(
            topic="What is Python async programming?",
            llm_decision={"needs_research": False, "mode": "closed_book"},
        )
        self.assertEqual(mode, "closed_book")

    def test_open_book_routing_for_recent_news(self):
        mode = resolve_router_mode(
            topic="OpenAI launches new agent platform this week",
            llm_decision={"needs_research": True, "mode": "hybrid"},
        )
        self.assertEqual(mode, "open_book")

    def test_disable_code_removes_fenced_code_blocks(self):
        markdown = "# Demo\n\n```python\nprint('hello')\n```\n\nA paragraph."
        cleaned = sanitize_markdown_for_user_settings(markdown, allow_code=False, allow_images=True)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("print('hello')", cleaned)
        self.assertIn("A paragraph.", cleaned)

    def test_disable_images_removes_image_markdown(self):
        markdown = "# Demo\n\n![Diagram](https://example.com/image.png)\n\nText"
        cleaned = sanitize_markdown_for_user_settings(markdown, allow_code=True, allow_images=False)
        self.assertNotIn("![", cleaned)
        self.assertNotIn("Diagram", cleaned)
        self.assertIn("Text", cleaned)

    def test_clean_evidence_removes_invalid_urls(self):
        cleaned = _clean_evidence([
            {"title": "Bad URL", "url": "javascript:alert(1)"},
            {"title": "Good URL", "url": "https://example.com/article"},
        ])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["url"], "https://example.com/article")

    def test_failed_image_generation_does_not_leak_error_block(self):
        plan = Plan(
            blog_title="Test blog",
            audience="General readers",
            tone="Clear",
            tasks=[
                Task(
                    id=1,
                    title="Intro",
                    goal="Explain",
                    bullets=["A", "B", "C"],
                    target_words=200,
                )
            ],
        )
        state = {
            "plan": plan,
            "merged_md": "# Test blog\n\nIntro paragraph\n\n[[IMAGE_1]]\n\nConclusion.",
            "md_with_placeholders": "# Test blog\n\nIntro paragraph\n\n[[IMAGE_1]]\n\nConclusion.",
            "image_specs": [{
                "placeholder": "[[IMAGE_1]]",
                "filename": "diagram.png",
                "alt": "Diagram",
                "caption": "Architecture overview",
                "prompt": "failing prompt",
            }],
            "include_images": True,
            "include_code": True,
        }

        with patch("bwa_backend._aigurulab_generate_image_url", side_effect=RuntimeError("API down")):
            result = generate_and_place_images(state)

        self.assertNotIn("IMAGE GENERATION FAILED", result["final"])
        self.assertNotIn("API down", result["final"])
        self.assertNotIn("failing prompt", result["final"])
        self.assertIn("Conclusion.", result["final"])


if __name__ == "__main__":
    unittest.main()
