import unittest

from tts_services.openai_tts_service import _is_retryable_tts_error


class OpenAITTSServiceTests(unittest.TestCase):
    def test_spaced_timeout_message_is_retryable(self):
        self.assertTrue(_is_retryable_tts_error(RuntimeError("Request timed out.")))

    def test_unrelated_error_is_not_retryable(self):
        self.assertFalse(_is_retryable_tts_error(RuntimeError("Invalid API key")))

