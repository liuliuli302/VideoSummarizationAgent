import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm.client import LLMClient


class TestLLMClientJsonParsing(unittest.TestCase):
    def test_generate_json_accepts_fenced_json(self):
        client = LLMClient.__new__(LLMClient)
        client.generate_text = lambda system_prompt, user_prompt: """```json
{
  \"video_theme\": \"general\",
  \"global_summary\": \"demo\",
  \"expert_weights\": {\"story_agent\": 0.25},
  \"reason\": \"ok\"
}
```"""

        payload = client.generate_json(system_prompt="system", user_prompt="user")

        self.assertEqual(payload["video_theme"], "general")
        self.assertEqual(payload["global_summary"], "demo")

    def test_generate_json_retries_when_first_response_has_no_json(self):
        client = LLMClient.__new__(LLMClient)
        responses = iter(
            [
                "I cannot comply with that format.",
                """```json
{
  \"score\": 0.8,
  \"reason\": \"retry ok\"
}
```""",
            ]
        )
        client.generate_text = lambda system_prompt, user_prompt: next(responses)

        payload = client.generate_json(system_prompt="system", user_prompt="user")

        self.assertEqual(payload["score"], 0.8)
        self.assertEqual(payload["reason"], "retry ok")


if __name__ == "__main__":
    unittest.main()