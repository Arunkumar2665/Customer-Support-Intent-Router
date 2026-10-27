import sys
import unittest
from pathlib import Path
from pydantic import ValidationError

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from support_router import (
    SupportTicketResult,
    classify_ticket,
)


class TestSupportRouter(unittest.TestCase):
    def test_billing_classification(self):
        result = classify_ticket(
            "We were charged twice for our subscription.",
            mode="mock"
        )
        self.assertEqual(result.category, "billing")
        self.assertEqual(result.priority, "high")

    def test_feature_request_classification(self):
        result = classify_ticket(
            "It would be useful to export dashboard charts.",
            mode="mock"
        )
        self.assertEqual(result.category, "feature_request")
        self.assertEqual(result.priority, "low")

    def test_urgent_account_access(self):
        result = classify_ticket(
            "I cannot log in. Our team is blocked from working.",
            mode="mock"
        )
        self.assertEqual(result.category, "account_access")
        self.assertEqual(result.priority, "urgent")

    def test_structured_output_fields(self):
        result = classify_ticket(
            "Where can I download the invoice?",
            mode="mock"
        )
        self.assertIsInstance(result, SupportTicketResult)
        self.assertTrue(result.summary)
        self.assertTrue(result.draft_response)
        self.assertTrue(result.reasoning_summary)

    def test_invalid_mode_raises_error(self):
        with self.assertRaises(ValueError):
            classify_ticket("Test message", mode="unknown")

    def test_invalid_category_fails_validation(self):
        with self.assertRaises(ValidationError):
            SupportTicketResult(
                category="not_a_category",
                priority="high",
                summary="Invalid category test",
                draft_response="Please contact support for assistance.",
                reasoning_summary="Validation test."
            )


if __name__ == "__main__":
    unittest.main()
