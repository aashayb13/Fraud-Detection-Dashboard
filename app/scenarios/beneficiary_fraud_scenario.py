# app/scenarios/beneficiary_fraud_scenario.py
"""
Rapid Beneficiary Addition Fraud Scenario

This module demonstrates the detection of fraud where a compromised administrator
account is used to rapidly add many new beneficiaries followed by fraudulent payments.

Scenario:
1. Attacker compromises an administrator account
2. Attacker uses automated script to add multiple new beneficiaries
3. Beneficiaries are added rapidly from same IP/source
4. Beneficiaries are not properly verified
5. Large payments are immediately made to newly added beneficiaries
6. Real administrator discovers fraud when reviewing audit logs

Detection Approach:
- Monitor for rapid addition of many beneficiaries in short time windows
- Flag beneficiaries added from same IP address or user (scripted)
- Detect payments to recently added beneficiaries
- Flag high ratio of payments to new vs established beneficiaries
- Require manual review for payments to unverified beneficiaries
"""
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid

from app.models.database import (
    Account, Beneficiary, Transaction,
    RiskAssessment, init_db, get_db
)
from app.services.rules_engine import RulesEngine
from app.services.risk_scoring import RiskScorer
from app.services.decision_engine import DecisionEngine
from app.services.context_provider import ContextProvider
from app.services.beneficiary_fraud_rules import initialize_beneficiary_fraud_rules


class BeneficiaryFraudScenario:
    """Handles rapid beneficiary addition fraud detection scenario."""

    def __init__(self, db: Session):
        self.db = db

        # Initialize components
        self.rules_engine = RulesEngine()
        self.context_provider = ContextProvider(db, enable_chain_analysis=False)

        # Add beneficiary fraud rules
        beneficiary_rules = initialize_beneficiary_fraud_rules(db)
        for rule in beneficiary_rules:
            self.rules_engine.add_rule(rule)

        # Initialize risk scorer and decision engine
        self.risk_scorer = RiskScorer(self.rules_engine)
        self.decision_engine = DecisionEngine(self.risk_scorer)

    def add_beneficiary(
        self,
        account_id: str,
        beneficiary_name: str,
        added_by: str,
        addition_source: str = "admin_portal",
        ip_address: str = None,
        verified: bool = False,
        timestamp: datetime = None
    ) -> Beneficiary:
        """Add a new beneficiary to an account."""
        beneficiary = Beneficiary(
            beneficiary_id=str(uuid.uuid4()),
            account_id=account_id,
            counterparty_id="PAY_" + str(uuid.uuid4())[:12],
            beneficiary_name=beneficiary_name,
            beneficiary_account_number=str(uuid.uuid4())[:10],
            beneficiary_routing_number="021000021",
            beneficiary_bank_name="External Bank",
            beneficiary_type="individual",
            added_timestamp=(timestamp or datetime.utcnow()).isoformat(),
            added_by=added_by,
            addition_source=addition_source,
            ip_address=ip_address,
            verified=verified
        )
        self.db.add(beneficiary)
        self.db.commit()
        return beneficiary

    def evaluate_payment_transaction(
        self,
        account_id: str,
        beneficiary: Beneficiary,
        amount: float,
        transaction_id: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a payment transaction for fraud risk.

        Args:
            account_id: Account making the payment
            beneficiary: Beneficiary receiving payment
            amount: Payment amount
            transaction_id: Optional transaction ID

        Returns:
            Decision result with risk assessment
        """
        # Create transaction data
        transaction = {
            "transaction_id": transaction_id or str(uuid.uuid4()),
            "account_id": account_id,
            "amount": amount,
            "direction": "debit",
            "transaction_type": "WIRE",
            "description": f"Payment to {beneficiary.beneficiary_name}",
            "timestamp": datetime.utcnow().isoformat(),
            "counterparty_id": beneficiary.counterparty_id
        }

        # Get context
        context = self.context_provider.get_transaction_context(transaction)

        # Evaluate transaction
        result = self.decision_engine.evaluate(transaction, context)

        return result

    def run_legitimate_scenario(self) -> Dict[str, Any]:
        """
        Run a legitimate beneficiary scenario (no fraud).

        Normal business adds a few beneficiaries over time with proper verification.
        """
        print("\n" + "="*80)
        print("LEGITIMATE SCENARIO: Normal beneficiary addition and payment")
        print("="*80)

        # Create account
        account_id = "ACC_" + str(uuid.uuid4())[:8]
        account = Account(account_id=account_id)
        self.db.add(account)
        self.db.commit()

        print(f"\nAccount: {account_id}")
        print(f"Scenario: Adding verified beneficiary for legitimate payment")

        # Add beneficiary 60 days ago (well-established)
        beneficiary_time = datetime.utcnow() - timedelta(days=60)
        beneficiary = self.add_beneficiary(
            account_id=account_id,
            beneficiary_name="Acme Supplies Inc",
            added_by="ADMIN_ALICE",
            addition_source="admin_portal",
            ip_address="10.0.1.50",
            verified=True,
            timestamp=beneficiary_time
        )

        print(f"\nBeneficiary: {beneficiary.beneficiary_name}")
        print(f"Added: 60 days ago (well-established)")
        print(f"Verified: Yes")
        print(f"Source: {beneficiary.addition_source}")

        # Process payment
        result = self.evaluate_payment_transaction(
            account_id=account_id,
            beneficiary=beneficiary,
            amount=5000.00
        )

        self._print_result(result)
        return result

    def run_fraud_scenario_rapid_additions(self) -> Dict[str, Any]:
        """
        Run fraud scenario: Rapid addition of many beneficiaries.

        Compromised admin account adds 8 beneficiaries in 6 hours from same IP.
        """
        print("\n" + "="*80)
        print("FRAUD SCENARIO: Rapid beneficiary additions from compromised account")
        print("="*80)

        # Create account
        account_id = "ACC_" + str(uuid.uuid4())[:8]
        account = Account(account_id=account_id)
        self.db.add(account)
        self.db.commit()

        print(f"\nAccount: {account_id}")
        print(f"Scenario: Compromised admin rapidly adds 8 beneficiaries")

        # Attacker IP and compromised admin
        attacker_ip = "203.0.113.42"
        compromised_admin = "ADMIN_COMPROMISED"

        # Add 8 beneficiaries in 6 hours from same IP (scripted attack)
        beneficiaries = []
        print(f"\nAdding beneficiaries from IP {attacker_ip}:")
        for i in range(8):
            hours_ago = 6 - (i * 0.5)
            timestamp = datetime.utcnow() - timedelta(hours=hours_ago)

            beneficiary = self.add_beneficiary(
                account_id=account_id,
                beneficiary_name=f"Suspicious Payee {i+1}",
                added_by=compromised_admin,
                addition_source="api",  # Suggests scripted
                ip_address=attacker_ip,
                verified=False,  # Not verified!
                timestamp=timestamp
            )
            beneficiaries.append(beneficiary)
            print(f"  - {beneficiary.beneficiary_name} ({hours_ago:.1f} hours ago)")

        print(f"\nTotal beneficiaries added in 6 hours: {len(beneficiaries)}")
        print(f"All from same IP: {attacker_ip}")
        print(f"All by same user: {compromised_admin}")
        print(f"All unverified!")

        # Now attempt payment to one of the newly added beneficiaries
        print(f"\nAttempting large payment to newly added beneficiary...")
        result = self.evaluate_payment_transaction(
            account_id=account_id,
            beneficiary=beneficiaries[5],
            amount=15000.00
        )

        self._print_result(result)
        return result

    def run_fraud_scenario_bulk_payments_to_new(self) -> Dict[str, Any]:
        """
        Run fraud scenario: Rapid additions followed by multiple payments.

        Shows high ratio of payments to newly added beneficiaries.
        """
        print("\n" + "="*80)
        print("FRAUD SCENARIO: Bulk payments to newly added beneficiaries")
        print("="*80)

        # Create account
        account_id = "ACC_" + str(uuid.uuid4())[:8]
        account = Account(account_id=account_id)
        self.db.add(account)
        self.db.commit()

        print(f"\nAccount: {account_id}")

        # Add some old legitimate beneficiaries
        old_ben = self.add_beneficiary(
            account_id=account_id,
            beneficiary_name="Legitimate Vendor",
            added_by="ADMIN_BOB",
            verified=True,
            timestamp=datetime.utcnow() - timedelta(days=90)
        )

        # Add 6 new suspicious beneficiaries in last 18 hours
        attacker_ip = "198.51.100.123"
        new_beneficiaries = []
        print(f"\nAdding 6 suspicious beneficiaries in last 18 hours:")
        for i in range(6):
            hours_ago = 18 - (i * 2)
            timestamp = datetime.utcnow() - timedelta(hours=hours_ago)

            beneficiary = self.add_beneficiary(
                account_id=account_id,
                beneficiary_name=f"Shell Company {i+1}",
                added_by="ADMIN_COMPROMISED",
                addition_source="bulk_upload",
                ip_address=attacker_ip,
                verified=False,
                timestamp=timestamp
            )
            new_beneficiaries.append(beneficiary)
            print(f"  - {beneficiary.beneficiary_name} ({hours_ago} hours ago)")

        # Create payments to 4 new beneficiaries
        print(f"\nCreating payments to newly added beneficiaries:")
        for i in range(4):
            tx = Transaction(
                transaction_id=f"TX_FRAUD_{i}",
                account_id=account_id,
                amount=8000.0,
                direction="debit",
                transaction_type="WIRE",
                counterparty_id=new_beneficiaries[i].counterparty_id,
                timestamp=(datetime.utcnow() - timedelta(hours=6-i)).isoformat()
            )
            self.db.add(tx)
            print(f"  - ${tx.amount:,.2f} to {new_beneficiaries[i].beneficiary_name}")

        # One payment to old legitimate beneficiary
        tx = Transaction(
            transaction_id="TX_LEGIT_1",
            account_id=account_id,
            amount=3000.0,
            direction="debit",
            transaction_type="ACH",
            counterparty_id=old_ben.counterparty_id,
            timestamp=(datetime.utcnow() - timedelta(hours=12)).isoformat()
        )
        self.db.add(tx)
        self.db.commit()

        print(f"\nPayment ratio: 4 to new beneficiaries vs 1 to established (80%)")

        # Evaluate current payment to another new beneficiary
        print(f"\nAttempting another payment to new beneficiary...")
        result = self.evaluate_payment_transaction(
            account_id=account_id,
            beneficiary=new_beneficiaries[5],
            amount=12000.00
        )

        self._print_result(result)
        return result

    def _print_result(self, result: Dict[str, Any]):
        """Print evaluation result in a formatted way."""
        print("\n" + "-"*80)
        print("RISK ASSESSMENT")
        print("-"*80)

        risk_score = result["risk_assessment"]["risk_score"]
        print(f"\nRisk Score: {risk_score:.2f}")

        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        elif risk_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"

        print(f"Risk Level: {risk_level}")
        print(f"Decision: {result['decision'].upper()}")

        if result["risk_assessment"]["triggered_rules"]:
            print(f"\nTriggered Rules ({len(result['risk_assessment']['triggered_rules'])}):")
            for rule_name, rule_info in result["risk_assessment"]["triggered_rules"].items():
                print(f"  - {rule_info['description']} (weight: {rule_info['weight']})")
        else:
            print("\nNo suspicious patterns detected")

        if result.get("cost_benefit"):
            cb = result["cost_benefit"]
            print(f"\nCost-Benefit Analysis:")
            print(f"  Expected Loss: ${cb.get('expected_loss', 0):,.2f}")
            print(f"  Review Cost: ${cb.get('review_cost', 0):,.2f}")

        print("\n" + "="*80)


def main():
    """Run all beneficiary fraud scenarios."""
    print("\n" + "#"*80)
    print("# BENEFICIARY FRAUD DETECTION DEMONSTRATION")
    print("#"*80)
    print("\nThis demonstrates detection of rapid beneficiary addition fraud")
    print("where compromised admin accounts are used to onboard fake payees.")

    # Initialize database
    init_db()
    db = next(get_db())

    try:
        scenario = BeneficiaryFraudScenario(db)

        # Run scenarios
        print("\n\n" + "="*80)
        print("SCENARIO 1: LEGITIMATE BUSINESS OPERATION")
        print("="*80)
        scenario.run_legitimate_scenario()

        print("\n\n" + "="*80)
        print("SCENARIO 2: RAPID BENEFICIARY ADDITIONS (COMPROMISED ACCOUNT)")
        print("="*80)
        scenario.run_fraud_scenario_rapid_additions()

        print("\n\n" + "="*80)
        print("SCENARIO 3: BULK PAYMENTS TO NEW BENEFICIARIES")
        print("="*80)
        scenario.run_fraud_scenario_bulk_payments_to_new()

        print("\n\n" + "#"*80)
        print("# DEMONSTRATION COMPLETE")
        print("#"*80)
        print("\nKey Findings:")
        print("- Legitimate beneficiary additions and payments are auto-approved")
        print("- Rapid additions from same IP/user trigger high-risk alerts")
        print("- Payments to newly added unverified beneficiaries require review")
        print("- Multiple layered rules provide robust fraud detection")
        print("\n")

    finally:
        db.close()


if __name__ == "__main__":
    main()
