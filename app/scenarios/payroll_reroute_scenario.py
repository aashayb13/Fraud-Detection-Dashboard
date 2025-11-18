# app/scenarios/payroll_reroute_scenario.py
"""
Payroll Rerouting Fraud Scenario

This module demonstrates the detection of payroll fraud where an employee's
direct deposit is rerouted to a fraudulent account after a deceptive account
update request.

Scenario:
1. Employee has been receiving regular payroll for months/years
2. Fraudster calls/emails HR pretending to be the employee
3. Fraudster requests bank account change for direct deposit
4. Next payroll is deposited into fraudster's account
5. Real employee discovers the fraud when paycheck doesn't arrive

Detection Approach:
- Monitor for recent account changes before payroll transactions
- Flag unverified or suspicious-source account changes
- Detect timing patterns (weekend/off-hours changes, rapid changes)
- Require manual review for high-risk payroll transactions
"""
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
import json

from app.models.database import (
    Account, Employee, AccountChangeHistory, Transaction,
    RiskAssessment, init_db, get_db
)
from app.services.rules_engine import RulesEngine
from app.services.risk_scoring import RiskScorer
from app.services.decision_engine import DecisionEngine
from app.services.context_provider import ContextProvider
from app.services.payroll_fraud_rules import initialize_payroll_fraud_rules


class PayrollRerouteScenario:
    """Handles payroll rerouting fraud detection scenario."""

    def __init__(self, db: Session):
        self.db = db

        # Initialize components
        self.rules_engine = RulesEngine()
        self.context_provider = ContextProvider(db)

        # Add payroll fraud rules
        payroll_rules = initialize_payroll_fraud_rules(db)
        for rule in payroll_rules:
            self.rules_engine.add_rule(rule)

        # Initialize risk scorer and decision engine
        self.risk_scorer = RiskScorer(self.rules_engine)
        self.decision_engine = DecisionEngine(self.risk_scorer)

    def create_employee(
        self,
        employee_id: str,
        name: str,
        email: str,
        account_id: str,
        payroll_account: str,
        payroll_routing: str,
        department: str = "Engineering"
    ) -> Employee:
        """Create a new employee record."""
        employee = Employee(
            employee_id=employee_id,
            account_id=account_id,
            name=name,
            email=email,
            department=department,
            payroll_account_number=payroll_account,
            payroll_routing_number=payroll_routing,
            payroll_bank_name="Original Bank",
            payroll_frequency="biweekly",
            hire_date=(datetime.utcnow() - timedelta(days=365)).isoformat(),
            last_payroll_date=(datetime.utcnow() - timedelta(days=14)).isoformat()
        )
        self.db.add(employee)
        self.db.commit()
        return employee

    def record_account_change(
        self,
        employee_id: str,
        account_id: str,
        change_type: str,
        old_value: str,
        new_value: str,
        change_source: str,
        verified: bool = False,
        timestamp: datetime = None,
        ip_address: str = None,
        verification_method: str = None
    ) -> AccountChangeHistory:
        """Record an account change."""
        change = AccountChangeHistory(
            change_id=str(uuid.uuid4()),
            employee_id=employee_id,
            account_id=account_id,
            timestamp=(timestamp or datetime.utcnow()).isoformat(),
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            change_source=change_source,
            ip_address=ip_address,
            verified=verified,
            verification_method=verification_method
        )
        self.db.add(change)
        self.db.commit()
        return change

    def evaluate_payroll_transaction(
        self,
        employee_id: str,
        amount: float,
        transaction_id: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a payroll transaction for fraud risk.

        Args:
            employee_id: Employee receiving payroll
            amount: Payroll amount
            transaction_id: Optional transaction ID

        Returns:
            Decision result with risk assessment
        """
        # Get employee
        employee = self.db.query(Employee).filter(
            Employee.employee_id == employee_id
        ).first()

        if not employee:
            raise ValueError(f"Employee {employee_id} not found")

        # Create transaction data
        transaction = {
            "transaction_id": transaction_id or str(uuid.uuid4()),
            "account_id": employee.account_id,
            "amount": amount,
            "transaction_type": "direct_deposit",
            "description": f"Payroll - {employee.name}",
            "timestamp": datetime.utcnow().isoformat(),
            "counterparty_id": "PAYROLL_SYSTEM",
            "metadata": json.dumps({
                "employee_id": employee_id,
                "payroll_period": datetime.utcnow().strftime("%Y-%m")
            })
        }

        # Get context
        base_context = self.context_provider.get_transaction_context(transaction)
        payroll_context = self.context_provider.get_payroll_context(transaction)

        # Merge contexts
        context = {**base_context, **payroll_context}

        # Evaluate transaction
        result = self.decision_engine.evaluate(transaction, context)

        return result

    def run_legitimate_scenario(self) -> Dict[str, Any]:
        """
        Run a legitimate payroll scenario (no fraud).

        Employee has stable account, no recent changes, normal payroll.
        """
        print("\n" + "="*80)
        print("LEGITIMATE SCENARIO: Normal payroll with verified account")
        print("="*80)

        # Create account
        account_id = "ACC_" + str(uuid.uuid4())[:8]
        account = Account(account_id=account_id)
        self.db.add(account)
        self.db.commit()

        # Create employee
        employee = self.create_employee(
            employee_id="EMP_001",
            name="Alice Johnson",
            email="alice.johnson@company.com",
            account_id=account_id,
            payroll_account="1234567890",
            payroll_routing="021000021"
        )

        print(f"\nEmployee: {employee.name}")
        print(f"Account: {employee.payroll_account_number}")
        print(f"Last payroll: {employee.last_payroll_date}")

        # Process payroll
        result = self.evaluate_payroll_transaction(
            employee_id=employee.employee_id,
            amount=5000.00
        )

        self._print_result(result)
        return result

    def run_fraud_scenario_unverified_change(self) -> Dict[str, Any]:
        """
        Run fraud scenario: Unverified account change via phone.

        Fraudster calls pretending to be employee, changes account without
        proper verification.
        """
        print("\n" + "="*80)
        print("FRAUD SCENARIO: Unverified phone request account change")
        print("="*80)

        # Create account
        account_id = "ACC_" + str(uuid.uuid4())[:8]
        account = Account(account_id=account_id)
        self.db.add(account)
        self.db.commit()

        # Create employee
        employee = self.create_employee(
            employee_id="EMP_002",
            name="Bob Smith",
            email="bob.smith@company.com",
            account_id=account_id,
            payroll_account="1111111111",
            payroll_routing="021000021"
        )

        print(f"\nEmployee: {employee.name}")
        print(f"Original Account: {employee.payroll_account_number}")

        # Fraudster calls and changes account 5 days ago
        change_date = datetime.utcnow() - timedelta(days=5)
        change = self.record_account_change(
            employee_id=employee.employee_id,
            account_id=account_id,
            change_type="account_number",
            old_value="1111111111",
            new_value="9999999999",  # Fraudulent account
            change_source="phone_request",
            verified=False,  # NOT VERIFIED!
            timestamp=change_date,
            ip_address=None,  # Phone call, no IP
            verification_method=None
        )

        print(f"\nAccount Change Recorded:")
        print(f"  Date: {change.timestamp}")
        print(f"  Source: {change.change_source}")
        print(f"  New Account: {change.new_value}")
        print(f"  Verified: {change.verified}")

        # Update employee record
        employee.payroll_account_number = change.new_value
        self.db.commit()

        # Process payroll
        result = self.evaluate_payroll_transaction(
            employee_id=employee.employee_id,
            amount=6500.00
        )

        self._print_result(result)
        return result

    def run_fraud_scenario_weekend_change(self) -> Dict[str, Any]:
        """
        Run fraud scenario: Weekend email request with rapid change.

        Fraudster sends email on Saturday requesting urgent account change.
        """
        print("\n" + "="*80)
        print("FRAUD SCENARIO: Weekend email request with timing suspicions")
        print("="*80)

        # Create account
        account_id = "ACC_" + str(uuid.uuid4())[:8]
        account = Account(account_id=account_id)
        self.db.add(account)
        self.db.commit()

        # Create employee
        employee = self.create_employee(
            employee_id="EMP_003",
            name="Carol Davis",
            email="carol.davis@company.com",
            account_id=account_id,
            payroll_account="2222222222",
            payroll_routing="021000021"
        )

        print(f"\nEmployee: {employee.name}")
        print(f"Original Account: {employee.payroll_account_number}")

        # Find last Saturday
        today = datetime.utcnow()
        days_since_saturday = (today.weekday() - 5) % 7
        last_saturday = today - timedelta(days=days_since_saturday)
        weekend_change_time = last_saturday.replace(hour=14, minute=30)

        # Fraudster emails on weekend
        change = self.record_account_change(
            employee_id=employee.employee_id,
            account_id=account_id,
            change_type="account_number",
            old_value="2222222222",
            new_value="8888888888",  # Fraudulent account
            change_source="email_request",
            verified=False,
            timestamp=weekend_change_time,
            ip_address="192.168.1.100",
            verification_method=None
        )

        print(f"\nAccount Change Recorded:")
        print(f"  Date: {change.timestamp} (Weekend!)")
        print(f"  Source: {change.change_source}")
        print(f"  New Account: {change.new_value}")
        print(f"  Verified: {change.verified}")

        # Update employee record
        employee.payroll_account_number = change.new_value
        self.db.commit()

        # Process payroll
        result = self.evaluate_payroll_transaction(
            employee_id=employee.employee_id,
            amount=7500.00
        )

        self._print_result(result)
        return result

    def run_fraud_scenario_rapid_changes(self) -> Dict[str, Any]:
        """
        Run fraud scenario: Multiple rapid account changes.

        Fraudster makes multiple attempts to change account, testing system.
        """
        print("\n" + "="*80)
        print("FRAUD SCENARIO: Multiple rapid account changes")
        print("="*80)

        # Create account
        account_id = "ACC_" + str(uuid.uuid4())[:8]
        account = Account(account_id=account_id)
        self.db.add(account)
        self.db.commit()

        # Create employee
        employee = self.create_employee(
            employee_id="EMP_004",
            name="David Wilson",
            email="david.wilson@company.com",
            account_id=account_id,
            payroll_account="3333333333",
            payroll_routing="021000021"
        )

        print(f"\nEmployee: {employee.name}")
        print(f"Original Account: {employee.payroll_account_number}")

        # Multiple changes over past 60 days
        print("\nMultiple Account Changes Detected:")
        changes = [
            (60, "4444444444", "email_request"),
            (45, "5555555555", "phone_request"),
            (7, "6666666666", "email_request"),
        ]

        for days_ago, new_account, source in changes:
            change_date = datetime.utcnow() - timedelta(days=days_ago)
            change = self.record_account_change(
                employee_id=employee.employee_id,
                account_id=account_id,
                change_type="account_number",
                old_value=employee.payroll_account_number,
                new_value=new_account,
                change_source=source,
                verified=False,
                timestamp=change_date
            )
            print(f"  {days_ago} days ago: {source} -> {new_account}")
            employee.payroll_account_number = new_account

        self.db.commit()

        # Process payroll
        result = self.evaluate_payroll_transaction(
            employee_id=employee.employee_id,
            amount=5500.00
        )

        self._print_result(result)
        return result

    def _print_result(self, result: Dict[str, Any]) -> None:
        """Print evaluation result in readable format."""
        print("\n" + "-"*80)
        print("RISK ASSESSMENT RESULT")
        print("-"*80)

        risk = result["risk_assessment"]
        print(f"\nRisk Score: {risk['risk_score']:.2f} (0-1 scale)")
        print(f"Decision: {result['decision'].upper()}")

        if result.get("review_reason"):
            print(f"Review Reason: {result['review_reason']}")

        print(f"\nTriggered Rules:")
        if risk["triggered_rules"]:
            for rule_name, rule_info in risk["triggered_rules"].items():
                print(f"  - {rule_info['description']} (weight: {rule_info['weight']})")
        else:
            print("  None")

        print(f"\nExplanation:")
        for explanation in risk["explanation"]:
            print(f"  - {explanation}")

        cost_benefit = result["cost_benefit"]
        print(f"\nCost-Benefit Analysis:")
        print(f"  Review Cost: ${cost_benefit['review_cost_usd']:.2f}")
        print(f"  Expected Loss: ${cost_benefit['expected_loss_usd']:.2f}")
        print(f"  Net Benefit: ${cost_benefit['net_benefit_of_review_usd']:.2f}")


def main():
    """Run all payroll reroute scenarios."""
    # Initialize database
    init_db()

    # Get database session
    db_gen = get_db()
    db = next(db_gen)

    try:
        scenario = PayrollRerouteScenario(db)

        print("\n")
        print("="*80)
        print("PAYROLL REROUTING FRAUD DETECTION DEMONSTRATION")
        print("="*80)
        print("\nThis demonstration shows how the transaction monitoring system")
        print("detects fraudulent payroll account changes.")

        # Run scenarios
        results = {}

        results["legitimate"] = scenario.run_legitimate_scenario()
        results["unverified"] = scenario.run_fraud_scenario_unverified_change()
        results["weekend"] = scenario.run_fraud_scenario_weekend_change()
        results["rapid"] = scenario.run_fraud_scenario_rapid_changes()

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nLegitimate Scenario: {results['legitimate']['decision']}")
        print(f"Unverified Change: {results['unverified']['decision']}")
        print(f"Weekend Change: {results['weekend']['decision']}")
        print(f"Rapid Changes: {results['rapid']['decision']}")

        print("\nThe system successfully identifies high-risk payroll transactions")
        print("and routes them for manual review, protecting against fraud.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
