# tests/test_account_takeover.py
"""
Unit tests for account takeover fraud detection functionality.
"""
import unittest
from datetime import datetime, timedelta
import uuid
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import (
    Base, Account, AccountChangeHistory, Transaction
)
from app.services.rules_engine import RulesEngine
from app.services.context_provider import ContextProvider
from app.services.risk_scoring import RiskScorer
from app.services.decision_engine import DecisionEngine
from app.services.account_takeover_rules import (
    initialize_account_takeover_rules,
    create_phone_change_before_transfer_rule,
    create_immediate_transfer_after_phone_change_rule,
    create_unverified_phone_change_transfer_rule,
    create_large_transfer_after_phone_change_rule,
)


class TestAccountTakeoverDetection(unittest.TestCase):
    """Test cases for account takeover fraud detection."""

    def setUp(self):
        """Set up test database and components."""
        # Create in-memory SQLite database
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        SessionLocal = sessionmaker(bind=self.engine)
        self.db = SessionLocal()

        # Initialize components
        self.rules_engine = RulesEngine()
        self.context_provider = ContextProvider(self.db)

        # Add account takeover rules
        takeover_rules = initialize_account_takeover_rules()
        for rule in takeover_rules:
            self.rules_engine.add_rule(rule)

        self.risk_scorer = RiskScorer(self.rules_engine)
        self.decision_engine = DecisionEngine(self.risk_scorer)

    def tearDown(self):
        """Clean up test database."""
        self.db.close()

    def _create_test_account(self) -> Account:
        """Create a test account."""
        account = Account(
            account_id="TEST_ACC_" + str(uuid.uuid4())[:8],
            risk_tier="medium",
            creation_date=(datetime.utcnow() - timedelta(days=365)).isoformat()
        )
        self.db.add(account)
        self.db.commit()
        return account

    def _create_phone_change(
        self,
        account: Account,
        hours_ago: int = 24,
        verified: bool = False,
        change_type: str = "phone",
        flagged: bool = False
    ) -> AccountChangeHistory:
        """Create a test phone/device change."""
        change_time = datetime.utcnow() - timedelta(hours=hours_ago)
        change = AccountChangeHistory(
            change_id=str(uuid.uuid4()),
            employee_id="EMP_" + str(uuid.uuid4())[:8],
            account_id=account.account_id,
            timestamp=change_time.isoformat(),
            change_type=change_type,
            old_value="+1-555-0100",
            new_value="+1-555-0199",
            change_source="user_portal",
            verified=verified,
            flagged_as_suspicious=flagged
        )
        self.db.add(change)
        self.db.commit()
        return change

    def _create_outgoing_transfer(
        self,
        account: Account,
        amount: float = 1000.0,
        counterparty_id: str = "EXTERNAL_123",
        is_new_counterparty: bool = False
    ) -> dict:
        """Create a test outgoing transfer transaction."""
        tx = {
            "transaction_id": str(uuid.uuid4()),
            "account_id": account.account_id,
            "amount": amount,
            "direction": "debit",
            "transaction_type": "WIRE",
            "description": "Wire transfer",
            "timestamp": datetime.utcnow().isoformat(),
            "counterparty_id": counterparty_id
        }

        # If not a new counterparty, create a historical transaction
        if not is_new_counterparty:
            historical_tx = Transaction(
                transaction_id=str(uuid.uuid4()),
                account_id=account.account_id,
                counterparty_id=counterparty_id,
                amount=500.0,
                direction="debit",
                transaction_type="WIRE",
                timestamp=(datetime.utcnow() - timedelta(days=60)).isoformat()
            )
            self.db.add(historical_tx)
            self.db.commit()

        return tx

    def test_legitimate_transfer_no_phone_change(self):
        """Test that legitimate transfer without phone change has low risk."""
        account = self._create_test_account()
        transaction = self._create_outgoing_transfer(account, is_new_counterparty=False)

        # Get context and evaluate
        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should be low to medium risk, auto-approved
        self.assertLess(result["risk_assessment"]["risk_score"], 0.5)
        self.assertEqual(result["decision"], "auto_approve")

    def test_phone_change_before_transfer_detected(self):
        """Test detection of transfer shortly after phone change."""
        account = self._create_test_account()

        # Create phone change 12 hours ago
        self._create_phone_change(account, hours_ago=12, verified=True)

        # Create outgoing transfer now
        transaction = self._create_outgoing_transfer(account, amount=2000.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should trigger phone change before transfer rule
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("phone_change_before_transfer_24h", triggered)
        self.assertGreater(result["risk_assessment"]["risk_score"], 0.2)

    def test_unverified_phone_change_high_risk(self):
        """Test that unverified phone change before transfer is high risk."""
        account = self._create_test_account()

        # Create UNVERIFIED phone change
        self._create_phone_change(account, hours_ago=24, verified=False)

        transaction = self._create_outgoing_transfer(account, amount=3000.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should trigger unverified phone change rule
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("unverified_phone_change_transfer_48h", triggered)
        self.assertGreater(result["risk_assessment"]["risk_score"], 0.25)
        # Manual review depends on overall risk threshold

    def test_suspicious_phone_change_flagged(self):
        """Test detection of transfers after suspicious phone changes."""
        account = self._create_test_account()

        # Create phone change flagged as suspicious
        self._create_phone_change(account, hours_ago=10, verified=False, flagged=True)

        transaction = self._create_outgoing_transfer(account, amount=5000.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should trigger suspicious phone change rule
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("suspicious_phone_change_transfer_48h", triggered)
        self.assertGreater(result["risk_assessment"]["risk_score"], 0.5)

    def test_immediate_transfer_after_phone_change_critical(self):
        """Test immediate transfer (within 1 hour) after phone change."""
        account = self._create_test_account()

        # Create phone change 30 minutes ago
        self._create_phone_change(account, hours_ago=0.5, verified=True)

        transaction = self._create_outgoing_transfer(account, amount=10000.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should trigger immediate transfer rule (critical)
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("immediate_transfer_after_phone_change_1h", triggered)
        self.assertGreater(result["risk_assessment"]["risk_score"], 0.4)
        # Should also trigger large transfer rule
        self.assertIn("large_transfer_after_phone_change_5000", triggered)

    def test_large_transfer_after_phone_change(self):
        """Test large transfer after phone change increases risk."""
        account = self._create_test_account()

        # Create phone change
        self._create_phone_change(account, hours_ago=20, verified=True)

        # Create LARGE transfer
        transaction = self._create_outgoing_transfer(account, amount=15000.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should trigger large transfer rule (threshold is $5000)
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("large_transfer_after_phone_change_5000", triggered)

    def test_rapid_phone_changes_detected(self):
        """Test detection of multiple phone changes in short period."""
        account = self._create_test_account()

        # Create multiple phone changes in 24 hours
        self._create_phone_change(account, hours_ago=20, change_type="phone")
        self._create_phone_change(account, hours_ago=10, change_type="device")

        transaction = self._create_outgoing_transfer(account, amount=2000.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should trigger rapid changes rule
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("rapid_phone_changes_2_in_24h", triggered)

    def test_new_counterparty_after_phone_change(self):
        """Test transfer to new counterparty after phone change."""
        account = self._create_test_account()

        # Create phone change
        self._create_phone_change(account, hours_ago=15, verified=True)

        # Create transfer to NEW counterparty
        transaction = self._create_outgoing_transfer(
            account,
            amount=3000.0,
            counterparty_id="NEW_UNKNOWN_ACCOUNT",
            is_new_counterparty=True
        )

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should trigger new counterparty after phone change rule
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("new_counterparty_after_phone_change_48h", triggered)

    def test_first_transfer_after_phone_change(self):
        """Test detection of first transfer after phone change."""
        account = self._create_test_account()

        # Create some historical transactions (before phone change)
        historical = Transaction(
            transaction_id=str(uuid.uuid4()),
            account_id=account.account_id,
            counterparty_id="REGULAR_PAYEE",
            amount=1000.0,
            direction="debit",
            transaction_type="ACH",
            timestamp=(datetime.utcnow() - timedelta(days=30)).isoformat()
        )
        self.db.add(historical)
        self.db.commit()

        # Create phone change 6 hours ago
        self._create_phone_change(account, hours_ago=6, verified=True)

        # This should be the first transfer after the phone change
        transaction = self._create_outgoing_transfer(account, amount=2500.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should detect first transfer after phone change
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("first_transfer_after_phone_change", triggered)

    def test_phone_change_context_provider(self):
        """Test that context provider correctly gathers phone change data."""
        account = self._create_test_account()

        # Create phone changes at different times
        self._create_phone_change(account, hours_ago=0.5, verified=True)
        self._create_phone_change(account, hours_ago=36, verified=False, flagged=True)

        transaction = self._create_outgoing_transfer(account, amount=1000.0)

        context = self.context_provider.get_transaction_context(transaction)

        # Verify context contains expected keys
        self.assertIn("phone_changes_count_1h", context)
        self.assertEqual(context["phone_changes_count_1h"], 1)

        self.assertIn("phone_changes_count_48h", context)
        self.assertEqual(context["phone_changes_count_48h"], 2)

        self.assertIn("is_outgoing_transfer", context)
        self.assertTrue(context["is_outgoing_transfer"])

        self.assertIn("hours_since_phone_change", context)
        self.assertLess(context["hours_since_phone_change"], 1)

        self.assertIn("unverified_phone_changes_48h", context)
        self.assertEqual(context["unverified_phone_changes_48h"], 1)

        self.assertIn("suspicious_phone_changes_48h", context)
        self.assertEqual(context["suspicious_phone_changes_48h"], 1)

    def test_incoming_transaction_not_flagged(self):
        """Test that incoming transactions don't trigger takeover rules."""
        account = self._create_test_account()

        # Create phone change
        self._create_phone_change(account, hours_ago=12, verified=False)

        # Create INCOMING transaction
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "account_id": account.account_id,
            "amount": 5000.0,
            "direction": "credit",  # Incoming
            "transaction_type": "WIRE",
            "description": "Wire deposit",
            "timestamp": datetime.utcnow().isoformat(),
            "counterparty_id": "SENDER_123"
        }

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should not trigger takeover rules (they only apply to outgoing)
        triggered = result["risk_assessment"]["triggered_rules"]
        takeover_rules = [r for r in triggered if "phone_change" in r or "takeover" in r]
        self.assertEqual(len(takeover_rules), 0)

    def test_old_phone_change_not_flagged(self):
        """Test that old phone changes don't trigger recent change rules."""
        account = self._create_test_account()

        # Create phone change 100 hours ago (beyond detection window)
        self._create_phone_change(account, hours_ago=100, verified=True)

        transaction = self._create_outgoing_transfer(account, amount=2000.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should not trigger 48h window rules
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertNotIn("phone_change_before_transfer_24h", triggered)
        self.assertNotIn("phone_change_before_transfer_48h_extended", triggered)

    def test_multiple_risk_factors_compound(self):
        """Test that multiple risk factors compound to very high risk."""
        account = self._create_test_account()

        # Create unverified, flagged phone change very recently
        self._create_phone_change(
            account,
            hours_ago=0.5,
            verified=False,
            flagged=True
        )

        # Large transfer to new counterparty
        transaction = self._create_outgoing_transfer(
            account,
            amount=20000.0,
            counterparty_id="SUSPICIOUS_ACCOUNT",
            is_new_counterparty=True
        )

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should trigger multiple rules and have very high risk
        triggered = result["risk_assessment"]["triggered_rules"]

        # Expect multiple rules to trigger
        self.assertIn("immediate_transfer_after_phone_change_1h", triggered)
        self.assertIn("unverified_phone_change_transfer_48h", triggered)
        self.assertIn("suspicious_phone_change_transfer_48h", triggered)
        self.assertIn("large_transfer_after_phone_change_5000", triggered)
        self.assertIn("new_counterparty_after_phone_change_48h", triggered)

        # Should be high risk with multiple rules triggered
        self.assertGreater(result["risk_assessment"]["risk_score"], 0.6)
        # Verify at least 5 rules triggered
        self.assertGreaterEqual(len(triggered), 5)

    def test_device_change_also_detected(self):
        """Test that device changes (not just phone) are also detected."""
        account = self._create_test_account()

        # Create DEVICE change (not phone)
        self._create_phone_change(
            account,
            hours_ago=10,
            verified=False,
            change_type="device"
        )

        transaction = self._create_outgoing_transfer(account, amount=3000.0)

        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)

        # Should still trigger rules (device changes are also indicators)
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("phone_change_before_transfer_24h", triggered)


if __name__ == "__main__":
    unittest.main()
