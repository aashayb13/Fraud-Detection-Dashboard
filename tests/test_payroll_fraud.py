# tests/test_payroll_fraud.py
"""
Unit tests for payroll fraud detection functionality.
"""
import unittest
from datetime import datetime, timedelta
import uuid
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import (
    Base, Account, Employee, AccountChangeHistory,
    Transaction, RiskAssessment
)
from app.services.rules_engine import RulesEngine
from app.services.context_provider import ContextProvider
from app.services.risk_scoring import RiskScorer
from app.services.decision_engine import DecisionEngine
from app.services.payroll_fraud_rules import (
    initialize_payroll_fraud_rules,
    is_payroll_transaction,
)


class TestPayrollFraudDetection(unittest.TestCase):
    """Test cases for payroll fraud detection."""

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

        # Add payroll fraud rules
        payroll_rules = initialize_payroll_fraud_rules(self.db)
        for rule in payroll_rules:
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
            risk_tier="medium"
        )
        self.db.add(account)
        self.db.commit()
        return account

    def _create_test_employee(
        self,
        account: Account,
        employee_id: str = None
    ) -> Employee:
        """Create a test employee."""
        employee = Employee(
            employee_id=employee_id or "EMP_" + str(uuid.uuid4())[:8],
            account_id=account.account_id,
            name="Test Employee",
            email="test@company.com",
            payroll_account_number="1234567890",
            payroll_routing_number="021000021",
            payroll_frequency="biweekly",
            hire_date=(datetime.utcnow() - timedelta(days=365)).isoformat(),
            last_payroll_date=(datetime.utcnow() - timedelta(days=14)).isoformat()
        )
        self.db.add(employee)
        self.db.commit()
        return employee

    def _create_account_change(
        self,
        employee: Employee,
        days_ago: int = 5,
        verified: bool = False,
        source: str = "phone_request"
    ) -> AccountChangeHistory:
        """Create a test account change."""
        change_date = datetime.utcnow() - timedelta(days=days_ago)
        change = AccountChangeHistory(
            change_id=str(uuid.uuid4()),
            employee_id=employee.employee_id,
            account_id=employee.account_id,
            timestamp=change_date.isoformat(),
            change_type="account_number",
            old_value="1234567890",
            new_value="9999999999",
            change_source=source,
            verified=verified
        )
        self.db.add(change)
        self.db.commit()
        return change

    def _create_payroll_transaction(
        self,
        employee: Employee,
        amount: float = 5000.0
    ) -> dict:
        """Create a test payroll transaction."""
        return {
            "transaction_id": str(uuid.uuid4()),
            "account_id": employee.account_id,
            "amount": amount,
            "transaction_type": "direct_deposit",
            "description": "Payroll",
            "timestamp": datetime.utcnow().isoformat(),
            "counterparty_id": "PAYROLL_SYSTEM",
            "metadata": json.dumps({"employee_id": employee.employee_id})
        }

    def test_payroll_transaction_detection(self):
        """Test that payroll transactions are correctly identified."""
        # Test various payroll transaction types
        payroll_txs = [
            {"transaction_type": "direct_deposit", "description": "Salary"},
            {"transaction_type": "ach_credit", "description": "payroll"},
            {"transaction_type": "payroll", "description": "wages"},
            {"transaction_type": "other", "description": "Direct Deposit - Payroll"},
        ]

        for tx in payroll_txs:
            self.assertTrue(
                is_payroll_transaction(tx),
                f"Failed to detect payroll: {tx}"
            )

        # Test non-payroll transactions
        non_payroll = {"transaction_type": "transfer", "description": "rent"}
        self.assertFalse(is_payroll_transaction(non_payroll))

    def test_legitimate_payroll_low_risk(self):
        """Test that legitimate payroll has low risk score."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)
        transaction = self._create_payroll_transaction(employee)

        # Get context and evaluate
        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        # Should be low risk and auto-approved
        self.assertLess(result["risk_assessment"]["risk_score"], 0.3)
        self.assertEqual(result["decision"], "auto_approve")

    def test_recent_unverified_account_change_high_risk(self):
        """Test that recent unverified account change triggers high risk."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Create unverified account change 5 days ago
        self._create_account_change(
            employee,
            days_ago=5,
            verified=False,
            source="phone_request"
        )

        transaction = self._create_payroll_transaction(employee)

        # Get context and evaluate
        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        # Should be high risk and require manual review
        self.assertGreater(result["risk_assessment"]["risk_score"], 0.6)
        self.assertEqual(result["decision"], "manual_review")

        # Check that specific rules were triggered
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("payroll_recent_account_change", triggered)
        self.assertIn("payroll_unverified_account_change", triggered)

    def test_verified_account_change_lower_risk(self):
        """Test that verified account changes have lower risk."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Create VERIFIED account change
        self._create_account_change(
            employee,
            days_ago=5,
            verified=True,
            source="employee_portal"
        )

        transaction = self._create_payroll_transaction(employee)

        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        # Should still flag recent change but not unverified
        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("payroll_recent_account_change", triggered)
        self.assertNotIn("payroll_unverified_account_change", triggered)

    def test_suspicious_change_source_detection(self):
        """Test detection of suspicious change sources."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Create change via email (suspicious)
        self._create_account_change(
            employee,
            days_ago=5,
            verified=False,
            source="email_request"
        )

        transaction = self._create_payroll_transaction(employee)

        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("payroll_suspicious_change_source", triggered)

    def test_rapid_account_changes_detection(self):
        """Test detection of multiple rapid account changes."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Create multiple changes
        for days_ago in [60, 45, 30, 7]:
            change_date = datetime.utcnow() - timedelta(days=days_ago)
            change = AccountChangeHistory(
                change_id=str(uuid.uuid4()),
                employee_id=employee.employee_id,
                account_id=employee.account_id,
                timestamp=change_date.isoformat(),
                change_type="account_number",
                old_value="1234567890",
                new_value=f"999999{days_ago:04d}",
                change_source="email_request",
                verified=False
            )
            self.db.add(change)
        self.db.commit()

        transaction = self._create_payroll_transaction(employee)

        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("payroll_rapid_account_changes", triggered)

    def test_weekend_account_change_detection(self):
        """Test detection of weekend account changes."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Find last Saturday
        today = datetime.utcnow()
        days_since_saturday = (today.weekday() - 5) % 7
        last_saturday = today - timedelta(days=days_since_saturday)

        # Create weekend change
        change = AccountChangeHistory(
            change_id=str(uuid.uuid4()),
            employee_id=employee.employee_id,
            account_id=employee.account_id,
            timestamp=last_saturday.isoformat(),
            change_type="account_number",
            old_value="1234567890",
            new_value="9999999999",
            change_source="email_request",
            verified=False
        )
        self.db.add(change)
        self.db.commit()

        transaction = self._create_payroll_transaction(employee)

        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("payroll_weekend_account_change", triggered)

    def test_off_hours_account_change_detection(self):
        """Test detection of off-hours account changes."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Create change at 2 AM
        late_night = datetime.utcnow().replace(hour=2, minute=0, second=0)

        change = AccountChangeHistory(
            change_id=str(uuid.uuid4()),
            employee_id=employee.employee_id,
            account_id=employee.account_id,
            timestamp=late_night.isoformat(),
            change_type="account_number",
            old_value="1234567890",
            new_value="9999999999",
            change_source="email_request",
            verified=False
        )
        self.db.add(change)
        self.db.commit()

        transaction = self._create_payroll_transaction(employee)

        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("payroll_off_hours_account_change", triggered)

    def test_high_value_payroll_detection(self):
        """Test detection of high-value payroll transactions."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Create high-value payroll
        transaction = self._create_payroll_transaction(employee, amount=25000.0)

        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("payroll_high_value", triggered)

    def test_payroll_context_provider(self):
        """Test that payroll context provider returns correct data."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Create some changes
        self._create_account_change(employee, days_ago=10, verified=False)
        self._create_account_change(employee, days_ago=5, verified=True)

        transaction = self._create_payroll_transaction(employee)

        context = self.context_provider.get_payroll_context(transaction)

        # Verify context contains expected keys
        self.assertIn("employee_id", context)
        self.assertEqual(context["employee_id"], employee.employee_id)
        self.assertIn("employee_name", context)
        self.assertIn("total_account_changes", context)
        self.assertEqual(context["total_account_changes"], 2)
        self.assertIn("unverified_changes_count", context)
        self.assertEqual(context["unverified_changes_count"], 1)

    def test_first_payroll_after_change(self):
        """Test detection of first payroll after account change."""
        account = self._create_test_account()
        employee = self._create_test_employee(account)

        # Create change after last payroll date
        last_payroll = datetime.fromisoformat(employee.last_payroll_date)
        change_date = last_payroll + timedelta(days=2)

        change = AccountChangeHistory(
            change_id=str(uuid.uuid4()),
            employee_id=employee.employee_id,
            account_id=employee.account_id,
            timestamp=change_date.isoformat(),
            change_type="account_number",
            old_value="1234567890",
            new_value="9999999999",
            change_source="employee_portal",
            verified=True
        )
        self.db.add(change)
        self.db.commit()

        transaction = self._create_payroll_transaction(employee)

        context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)
        context.update(payroll_context)

        result = self.decision_engine.evaluate(transaction, context)

        triggered = result["risk_assessment"]["triggered_rules"]
        self.assertIn("payroll_first_after_account_change", triggered)


if __name__ == "__main__":
    unittest.main()
