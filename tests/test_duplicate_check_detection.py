"""
Unit tests for duplicate check deposit fraud detection.

This module tests the detection of duplicate check deposits - when the same
physical check is deposited multiple times at different banks or accounts.
"""

import unittest
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Transaction, Account
from app.services.check_fraud_rules import (
    create_duplicate_check_rule,
    create_rapid_check_sequence_rule,
    create_check_amount_mismatch_rule,
    is_check_deposit,
    extract_check_info
)
from app.services.context_provider import ContextProvider


class TestDuplicateCheckDetection(unittest.TestCase):
    """Test cases for duplicate check deposit detection."""

    def setUp(self):
        """Set up test database and sample data."""
        # Create in-memory SQLite database for testing
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db = Session()

        # Create test accounts
        self.account1 = Account(
            account_id="ACC_001",
            creation_date=datetime.utcnow().isoformat(),
            risk_tier="medium",
            status="active"
        )
        self.account2 = Account(
            account_id="ACC_002",
            creation_date=datetime.utcnow().isoformat(),
            risk_tier="medium",
            status="active"
        )
        self.db.add(self.account1)
        self.db.add(self.account2)
        self.db.commit()

        # Initialize context provider
        self.context_provider = ContextProvider(self.db)

    def tearDown(self):
        """Clean up test database."""
        self.db.close()

    def test_duplicate_check_same_account(self):
        """Test detection of duplicate check deposit to the same account."""
        # Create first check deposit
        first_deposit = Transaction(
            transaction_id="TX_001",
            account_id="ACC_001",
            timestamp=(datetime.utcnow() - timedelta(days=5)).isoformat(),
            amount=1500.0,
            direction="credit",
            transaction_type="CHECK_DEPOSIT",
            counterparty_id="CHECK_001",
            description="Check deposit",
            tx_metadata=json.dumps({
                "check_number": "1234",
                "check_amount": 1500.0,
                "routing_number": "021000021",
                "account_number": "987654321",
                "drawer": "John Doe"
            })
        )
        self.db.add(first_deposit)
        self.db.commit()

        # Create duplicate check deposit (same check, same account)
        current_transaction = {
            "transaction_id": "TX_002",
            "account_id": "ACC_001",
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 1500.0,
            "direction": "credit",
            "transaction_type": "CHECK_DEPOSIT",
            "counterparty_id": "CHECK_001",
            "description": "Check deposit",
            "tx_metadata": json.dumps({
                "check_number": "1234",
                "check_amount": 1500.0,
                "routing_number": "021000021",
                "account_number": "987654321",
                "drawer": "John Doe"
            })
        }

        # Get check context
        context = self.context_provider.get_check_context(current_transaction)

        # Verify duplicate was detected
        self.assertIn("duplicate_checks", context)
        self.assertEqual(len(context["duplicate_checks"]), 1)
        self.assertEqual(context["duplicate_checks"][0]["transaction_id"], "TX_001")

        # Test the rule
        rule = create_duplicate_check_rule(self.db)
        result = rule.evaluate(current_transaction, context)
        self.assertTrue(result, "Rule should trigger on duplicate check")

    def test_duplicate_check_different_accounts(self):
        """Test detection of duplicate check deposited to different accounts."""
        # Create first check deposit to account 1
        first_deposit = Transaction(
            transaction_id="TX_003",
            account_id="ACC_001",
            timestamp=(datetime.utcnow() - timedelta(days=10)).isoformat(),
            amount=2500.0,
            direction="credit",
            transaction_type="MOBILE_DEPOSIT",
            counterparty_id="CHECK_002",
            description="Mobile check deposit",
            tx_metadata=json.dumps({
                "check_number": "5678",
                "check_amount": 2500.0,
                "routing_number": "011000015",
                "account_number": "123456789",
                "drawer": "Jane Smith"
            })
        )
        self.db.add(first_deposit)
        self.db.commit()

        # Create duplicate check deposit to account 2 (different account!)
        current_transaction = {
            "transaction_id": "TX_004",
            "account_id": "ACC_002",  # Different account
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 2500.0,
            "direction": "credit",
            "transaction_type": "CHECK_DEPOSIT",
            "counterparty_id": "CHECK_002",
            "description": "Check deposit",
            "tx_metadata": json.dumps({
                "check_number": "5678",  # Same check number
                "check_amount": 2500.0,  # Same amount
                "routing_number": "011000015",  # Same routing
                "account_number": "123456789",  # Same source account
                "drawer": "Jane Smith"
            })
        }

        # Get check context
        context = self.context_provider.get_check_context(current_transaction)

        # Verify duplicate was detected across different accounts
        self.assertIn("duplicate_checks", context)
        self.assertEqual(len(context["duplicate_checks"]), 1)
        self.assertEqual(context["duplicate_checks"][0]["account_id"], "ACC_001")

        # Test the rule
        rule = create_duplicate_check_rule(self.db)
        result = rule.evaluate(current_transaction, context)
        self.assertTrue(result, "Rule should trigger on duplicate check across accounts")

    def test_no_duplicate_different_check_number(self):
        """Test that different check numbers don't trigger duplicate detection."""
        # Create first check deposit
        first_deposit = Transaction(
            transaction_id="TX_005",
            account_id="ACC_001",
            timestamp=(datetime.utcnow() - timedelta(days=3)).isoformat(),
            amount=1000.0,
            direction="credit",
            transaction_type="CHECK_DEPOSIT",
            counterparty_id="CHECK_003",
            description="Check deposit",
            tx_metadata=json.dumps({
                "check_number": "1111",
                "check_amount": 1000.0,
                "routing_number": "021000021",
                "account_number": "111111111"
            })
        )
        self.db.add(first_deposit)
        self.db.commit()

        # Create second check deposit with different check number
        current_transaction = {
            "transaction_id": "TX_006",
            "account_id": "ACC_001",
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 1000.0,
            "direction": "credit",
            "transaction_type": "CHECK_DEPOSIT",
            "counterparty_id": "CHECK_004",
            "description": "Check deposit",
            "tx_metadata": json.dumps({
                "check_number": "2222",  # Different check number
                "check_amount": 1000.0,
                "routing_number": "021000021",
                "account_number": "111111111"
            })
        }

        # Get check context
        context = self.context_provider.get_check_context(current_transaction)

        # Verify no duplicate was detected
        self.assertNotIn("duplicate_checks", context)

        # Test the rule
        rule = create_duplicate_check_rule(self.db)
        result = rule.evaluate(current_transaction, context)
        self.assertFalse(result, "Rule should not trigger on different check numbers")

    def test_no_duplicate_different_amount(self):
        """Test that same check number with different amounts doesn't trigger (indicates alteration)."""
        # Create first check deposit
        first_deposit = Transaction(
            transaction_id="TX_007",
            account_id="ACC_001",
            timestamp=(datetime.utcnow() - timedelta(days=7)).isoformat(),
            amount=500.0,
            direction="credit",
            transaction_type="CHECK_DEPOSIT",
            counterparty_id="CHECK_005",
            description="Check deposit",
            tx_metadata=json.dumps({
                "check_number": "3333",
                "check_amount": 500.0,
                "routing_number": "021000021",
                "account_number": "222222222"
            })
        )
        self.db.add(first_deposit)
        self.db.commit()

        # Create second deposit with same check number but different amount
        current_transaction = {
            "transaction_id": "TX_008",
            "account_id": "ACC_001",
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 5000.0,  # Different amount!
            "direction": "credit",
            "transaction_type": "CHECK_DEPOSIT",
            "counterparty_id": "CHECK_005",
            "description": "Check deposit",
            "tx_metadata": json.dumps({
                "check_number": "3333",
                "check_amount": 5000.0,  # Different amount
                "routing_number": "021000021",
                "account_number": "222222222"
            })
        }

        # Get check context
        context = self.context_provider.get_check_context(current_transaction)

        # Should not be detected as duplicate (different amount indicates possible alteration)
        self.assertNotIn("duplicate_checks", context)

        # But should detect amount mismatch
        self.assertIn("check_amount_mismatch", context)

    def test_rapid_check_sequence_detection(self):
        """Test detection of rapid sequence of check deposits."""
        # Create multiple check deposits in the last hour
        now = datetime.utcnow()
        for i in range(5):
            check = Transaction(
                transaction_id=f"TX_CHECK_{i}",
                account_id="ACC_001",
                timestamp=(now - timedelta(minutes=10*i)).isoformat(),
                amount=1000.0 + i * 100,
                direction="credit",
                transaction_type="CHECK_DEPOSIT",
                counterparty_id=f"CHECK_{i}",
                description="Check deposit",
                tx_metadata=json.dumps({
                    "check_number": f"444{i}",
                    "check_amount": 1000.0 + i * 100
                })
            )
            self.db.add(check)
        self.db.commit()

        # Current transaction
        current_transaction = {
            "transaction_id": "TX_009",
            "account_id": "ACC_001",
            "timestamp": now.isoformat(),
            "amount": 1500.0,
            "direction": "credit",
            "transaction_type": "CHECK_DEPOSIT",
            "counterparty_id": "CHECK_NEW",
            "description": "Check deposit",
            "tx_metadata": json.dumps({
                "check_number": "4445",
                "check_amount": 1500.0
            })
        }

        # Get check context
        context = self.context_provider.get_check_context(current_transaction)

        # Verify rapid check metrics
        self.assertIn("check_count_1h", context)
        self.assertGreaterEqual(context["check_count_1h"], 5)
        self.assertGreaterEqual(context["check_amount_1h"], 5000.0)

        # Test the rule
        rule = create_rapid_check_sequence_rule(
            min_checks_per_hour=5,
            min_total_amount=5000.0
        )
        result = rule.evaluate(current_transaction, context)
        self.assertTrue(result, "Rule should trigger on rapid check sequence")

    def test_check_amount_mismatch_detection(self):
        """Test detection of check amount mismatches (possible alteration)."""
        # Create multiple deposits of check 9999 with amount $1000
        for i in range(3):
            check = Transaction(
                transaction_id=f"TX_AMT_{i}",
                account_id="ACC_001",
                timestamp=(datetime.utcnow() - timedelta(days=30*i)).isoformat(),
                amount=1000.0,
                direction="credit",
                transaction_type="CHECK_DEPOSIT",
                counterparty_id="CHECK_9999",
                description="Check deposit",
                tx_metadata=json.dumps({
                    "check_number": "9999",
                    "check_amount": 1000.0,
                    "routing_number": "021000021"
                })
            )
            self.db.add(check)
        self.db.commit()

        # Now deposit same check number with significantly different amount
        current_transaction = {
            "transaction_id": "TX_010",
            "account_id": "ACC_001",
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 10000.0,  # 10x the usual amount!
            "direction": "credit",
            "transaction_type": "CHECK_DEPOSIT",
            "counterparty_id": "CHECK_9999",
            "description": "Check deposit",
            "tx_metadata": json.dumps({
                "check_number": "9999",
                "check_amount": 10000.0,  # Significantly different
                "routing_number": "021000021"
            })
        }

        # Get check context
        context = self.context_provider.get_check_context(current_transaction)

        # Verify amount mismatch was detected
        self.assertIn("check_amount_mismatch", context)
        mismatch = context["check_amount_mismatch"]
        self.assertEqual(mismatch["previous_amount"], 1000.0)
        self.assertEqual(mismatch["current_amount"], 10000.0)
        self.assertGreater(mismatch["deviation_percent"], 5.0)

        # Test the rule
        rule = create_check_amount_mismatch_rule(max_deviation_percent=5.0)
        result = rule.evaluate(current_transaction, context)
        self.assertTrue(result, "Rule should trigger on amount mismatch")

    def test_is_check_deposit_helper(self):
        """Test the is_check_deposit helper function."""
        # Valid check deposit
        check_tx = {
            "direction": "credit",
            "transaction_type": "CHECK_DEPOSIT"
        }
        self.assertTrue(is_check_deposit(check_tx))

        # Mobile deposit
        mobile_tx = {
            "direction": "credit",
            "transaction_type": "MOBILE_DEPOSIT"
        }
        self.assertTrue(is_check_deposit(mobile_tx))

        # Not a check (ACH transfer)
        ach_tx = {
            "direction": "credit",
            "transaction_type": "ACH"
        }
        self.assertFalse(is_check_deposit(ach_tx))

        # Outgoing check (debit)
        outgoing_check = {
            "direction": "debit",
            "transaction_type": "CHECK"
        }
        self.assertFalse(is_check_deposit(outgoing_check))

    def test_extract_check_info_helper(self):
        """Test the extract_check_info helper function."""
        # Transaction with check metadata
        transaction = {
            "tx_metadata": json.dumps({
                "check_number": "12345",
                "check_amount": 1500.0,
                "routing_number": "021000021",
                "account_number": "987654321",
                "payee": "John Doe",
                "drawer": "Jane Smith",
                "check_date": "2025-10-15"
            })
        }

        check_info = extract_check_info(transaction)

        self.assertEqual(check_info["check_number"], "12345")
        self.assertEqual(check_info["amount"], 1500.0)
        self.assertEqual(check_info["routing_number"], "021000021")
        self.assertEqual(check_info["account_number"], "987654321")
        self.assertEqual(check_info["payee"], "John Doe")
        self.assertEqual(check_info["drawer"], "Jane Smith")
        self.assertEqual(check_info["check_date"], "2025-10-15")

    def test_duplicate_outside_lookback_window(self):
        """Test that duplicates outside the lookback window are not detected."""
        # Create check deposit 100 days ago (outside default 90-day window)
        old_deposit = Transaction(
            transaction_id="TX_OLD",
            account_id="ACC_001",
            timestamp=(datetime.utcnow() - timedelta(days=100)).isoformat(),
            amount=3000.0,
            direction="credit",
            transaction_type="CHECK_DEPOSIT",
            counterparty_id="CHECK_OLD",
            description="Check deposit",
            tx_metadata=json.dumps({
                "check_number": "7777",
                "check_amount": 3000.0
            })
        )
        self.db.add(old_deposit)
        self.db.commit()

        # Create "duplicate" check deposit today
        current_transaction = {
            "transaction_id": "TX_NEW",
            "account_id": "ACC_001",
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 3000.0,
            "direction": "credit",
            "transaction_type": "CHECK_DEPOSIT",
            "counterparty_id": "CHECK_OLD",
            "description": "Check deposit",
            "tx_metadata": json.dumps({
                "check_number": "7777",
                "check_amount": 3000.0
            })
        }

        # Get check context
        context = self.context_provider.get_check_context(current_transaction)

        # Verify no duplicate was detected (outside lookback window)
        self.assertNotIn("duplicate_checks", context)


if __name__ == "__main__":
    unittest.main()
