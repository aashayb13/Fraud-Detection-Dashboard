# app/scenarios/vendor_impersonation_scenario.py
"""
Vendor Impersonation / BEC Fraud Scenario

This module demonstrates the detection of vendor impersonation fraud where a
supplier's bank details are fraudulently changed and a payment is sent to the
new (fraudulent) account, typically the same day.

Scenario:
1. Company has an established supplier relationship with regular payments
2. Attacker impersonates the vendor via email/phone
3. Attacker requests "urgent" bank account update for payment
4. Payment is processed to the fraudulent account shortly after the change
5. Real vendor never receives payment and reports the issue later

This is one of the most common Business Email Compromise (BEC) attack patterns,
costing businesses billions annually.

Detection Approach:
- Monitor timing between account changes and payments (same-day is critical)
- Flag unverified or email/phone-sourced account changes
- Detect first payments to newly changed accounts
- Monitor weekend/off-hours change patterns
- Require manual review for high-risk payments
"""
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
import json

from app.models.database import (
    Account, Beneficiary, BeneficiaryChangeHistory, Transaction,
    RiskAssessment, init_db, get_db
)
from app.services.rules_engine import RulesEngine
from app.services.risk_scoring import RiskScorer
from app.services.decision_engine import DecisionEngine
from app.services.context_provider import ContextProvider
from app.services.beneficiary_fraud_rules import initialize_beneficiary_fraud_rules


class VendorImpersonationScenario:
    """Handles vendor impersonation/BEC fraud detection scenario."""

    def __init__(self, db: Session):
        self.db = db

        # Initialize components
        self.rules_engine = RulesEngine()
        self.context_provider = ContextProvider(db)

        # Add beneficiary fraud rules
        beneficiary_rules = initialize_beneficiary_fraud_rules(db)
        for rule in beneficiary_rules:
            self.rules_engine.add_rule(rule)

        # Initialize risk scorer and decision engine
        self.risk_scorer = RiskScorer(self.rules_engine)
        self.decision_engine = DecisionEngine(self.risk_scorer)

    def create_beneficiary(
        self,
        beneficiary_id: str,
        name: str,
        account_id: str,
        bank_account: str,
        bank_routing: str,
        beneficiary_type: str = "supplier",
        email: str = None,
        verified: bool = True,
        days_since_registration: int = 365
    ) -> Beneficiary:
        """Create a new beneficiary/vendor record."""
        beneficiary = Beneficiary(
            beneficiary_id=beneficiary_id,
            account_id=account_id,
            name=name,
            beneficiary_type=beneficiary_type,
            email=email,
            bank_account_number=bank_account,
            bank_routing_number=bank_routing,
            bank_name="Original Bank",
            registration_date=(datetime.utcnow() - timedelta(days=days_since_registration)).isoformat(),
            last_payment_date=(datetime.utcnow() - timedelta(days=30)).isoformat(),
            total_payments_received=12,
            total_amount_received=120000.0,
            verified=verified,
            status="active"
        )
        self.db.add(beneficiary)
        self.db.commit()
        return beneficiary

    def record_beneficiary_change(
        self,
        beneficiary_id: str,
        account_id: str,
        change_type: str,
        old_value: str,
        new_value: str,
        change_source: str,
        verified: bool = False,
        timestamp: datetime = None,
        requestor_name: str = None,
        requestor_email: str = None,
        verification_method: str = None,
        ip_address: str = None
    ) -> BeneficiaryChangeHistory:
        """Record a beneficiary account change."""
        change = BeneficiaryChangeHistory(
            change_id=str(uuid.uuid4()),
            beneficiary_id=beneficiary_id,
            account_id=account_id,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            change_source=change_source,
            requestor_name=requestor_name,
            requestor_email=requestor_email,
            timestamp=(timestamp or datetime.utcnow()).isoformat(),
            verified=verified,
            verification_method=verification_method,
            ip_address=ip_address
        )
        self.db.add(change)
        self.db.commit()
        return change

    def create_payment_transaction(
        self,
        transaction_id: str,
        account_id: str,
        beneficiary_id: str,
        amount: float,
        description: str = "Invoice Payment"
    ) -> Dict[str, Any]:
        """Create a payment transaction."""
        return {
            "transaction_id": transaction_id,
            "account_id": account_id,
            "counterparty_id": beneficiary_id,
            "amount": amount,
            "direction": "debit",
            "transaction_type": "vendor_payment",
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": json.dumps({"beneficiary_id": beneficiary_id})
        }

    def evaluate_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a transaction for fraud risk."""
        context = self.context_provider.get_transaction_context(transaction)
        result = self.decision_engine.evaluate(transaction, context)
        return result

    def run_scenario(self, scenario_name: str, scenario_func):
        """Run a scenario and print results."""
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*80}")

        result = scenario_func()

        print(f"\nRisk Assessment:")
        print(f"  Risk Score: {result['risk_assessment']['risk_score']:.2f}")
        print(f"  Decision: {result['decision'].upper()}")
        print(f"\nTriggered Rules:")
        for rule_name in result['risk_assessment']['triggered_rules']:
            print(f"  - {rule_name}")

        context = result['risk_assessment'].get('context', {})
        if context:
            print(f"\nContext Details:")
            for key, value in context.items():
                if not key.endswith('_changes') and not key.endswith('_details'):
                    print(f"  {key}: {value}")

        return result


def main():
    """Run vendor impersonation fraud detection scenarios."""
    print("Vendor Impersonation / BEC Fraud Detection Demo")
    print("=" * 80)

    # Initialize database
    init_db()
    db = next(get_db())

    # Create scenario handler
    scenario = VendorImpersonationScenario(db)

    # Create test account
    account = Account(account_id="ACC001", risk_tier="medium")
    db.add(account)
    db.commit()

    # -------------------------------------------------------------------------
    # Scenario 1: CRITICAL - Same-day payment after unverified email change
    # -------------------------------------------------------------------------
    def scenario_1():
        print("\nSetting up scenario...")
        print("- Established supplier with 1+ year history")
        print("- Email received requesting 'urgent' bank account update")
        print("- Account changed WITHOUT verification (red flag)")
        print("- Payment processed 2 hours after the change (CRITICAL)")

        # Create established vendor
        vendor = scenario.create_beneficiary(
            beneficiary_id="VENDOR001",
            name="ABC Office Supplies Inc.",
            account_id="ACC001",
            bank_account="****1234",
            bank_routing="021000021",
            email="accounts@abc-supplies.com",
            verified=True,
            days_since_registration=400
        )

        # Record account change via email (2 hours ago)
        change_time = datetime.utcnow() - timedelta(hours=2)
        scenario.record_beneficiary_change(
            beneficiary_id=vendor.beneficiary_id,
            account_id="ACC001",
            change_type="account_number",
            old_value="****1234",
            new_value="****9999",
            change_source="email_request",
            verified=False,
            timestamp=change_time,
            requestor_name="John Smith",
            requestor_email="j.smith@abc-supplies.com"  # Note: could be spoofed
        )

        # Process payment
        transaction = scenario.create_payment_transaction(
            transaction_id="TX001",
            account_id="ACC001",
            beneficiary_id=vendor.beneficiary_id,
            amount=15000.00,
            description="Invoice #12345 Payment"
        )

        return scenario.evaluate_transaction(transaction)

    scenario.run_scenario(
        "Same-Day Payment After Unverified Email Change",
        scenario_1
    )

    # -------------------------------------------------------------------------
    # Scenario 2: HIGH RISK - Payment within 3 days of phone change
    # -------------------------------------------------------------------------
    def scenario_2():
        print("\nSetting up scenario...")
        print("- Long-standing vendor relationship")
        print("- Phone call requesting account update 3 days ago")
        print("- Change verified via callback (good)")
        print("- High-value payment being processed")

        vendor = scenario.create_beneficiary(
            beneficiary_id="VENDOR002",
            name="XYZ Manufacturing Ltd.",
            account_id="ACC001",
            bank_account="****5678",
            bank_routing="021000021",
            email="ap@xyz-mfg.com",
            verified=True,
            days_since_registration=800
        )

        # Record verified change (3 days ago)
        change_time = datetime.utcnow() - timedelta(days=3)
        scenario.record_beneficiary_change(
            beneficiary_id=vendor.beneficiary_id,
            account_id="ACC001",
            change_type="routing_number",
            old_value="021000021",
            new_value="026009593",
            change_source="phone_request",
            verified=True,
            verification_method="callback",
            timestamp=change_time,
            requestor_name="Sarah Johnson"
        )

        # Process payment
        transaction = scenario.create_payment_transaction(
            transaction_id="TX002",
            account_id="ACC001",
            beneficiary_id=vendor.beneficiary_id,
            amount=25000.00,
            description="Invoice #INV-2024-089"
        )

        return scenario.evaluate_transaction(transaction)

    scenario.run_scenario(
        "Payment Within 3 Days of Phone Change (Verified)",
        scenario_2
    )

    # -------------------------------------------------------------------------
    # Scenario 3: MEDIUM RISK - New vendor first payment
    # -------------------------------------------------------------------------
    def scenario_3():
        print("\nSetting up scenario...")
        print("- Newly registered vendor (30 days ago)")
        print("- First payment being processed")
        print("- Moderate amount")
        print("- No account changes")

        vendor = scenario.create_beneficiary(
            beneficiary_id="VENDOR003",
            name="NewCo Services LLC",
            account_id="ACC001",
            bank_account="****3333",
            bank_routing="021000021",
            email="billing@newco-services.com",
            verified=True,
            days_since_registration=30
        )

        # First payment, no changes
        transaction = scenario.create_payment_transaction(
            transaction_id="TX003",
            account_id="ACC001",
            beneficiary_id=vendor.beneficiary_id,
            amount=5000.00,
            description="Invoice #001 - Initial Payment"
        )

        return scenario.evaluate_transaction(transaction)

    scenario.run_scenario(
        "First Payment to New Vendor (No Changes)",
        scenario_3
    )

    # -------------------------------------------------------------------------
    # Scenario 4: LOW RISK - Regular payment to established vendor
    # -------------------------------------------------------------------------
    def scenario_4():
        print("\nSetting up scenario...")
        print("- Well-established vendor (2+ years)")
        print("- Regular payment schedule")
        print("- No recent account changes")
        print("- Moderate amount")

        vendor = scenario.create_beneficiary(
            beneficiary_id="VENDOR004",
            name="Trusted Supplier Co.",
            account_id="ACC001",
            bank_account="****7777",
            bank_routing="021000021",
            email="payments@trusted-supplier.com",
            verified=True,
            days_since_registration=900
        )

        # Regular payment
        transaction = scenario.create_payment_transaction(
            transaction_id="TX004",
            account_id="ACC001",
            beneficiary_id=vendor.beneficiary_id,
            amount=8500.00,
            description="Monthly Invoice Payment"
        )

        return scenario.evaluate_transaction(transaction)

    scenario.run_scenario(
        "Regular Payment to Established Vendor",
        scenario_4
    )

    # -------------------------------------------------------------------------
    # Scenario 5: EXTREME RISK - Weekend change + same-day high-value payment
    # -------------------------------------------------------------------------
    def scenario_5():
        print("\nSetting up scenario...")
        print("- Established vendor")
        print("- Account changed on SUNDAY (weekend - red flag)")
        print("- Changed via EMAIL without verification")
        print("- High-value payment same day")
        print("- First payment after account change")

        vendor = scenario.create_beneficiary(
            beneficiary_id="VENDOR005",
            name="Global Tech Partners",
            account_id="ACC001",
            bank_account="****8888",
            bank_routing="021000021",
            email="finance@globaltech.com",
            verified=True,
            days_since_registration=600
        )

        # Weekend change (Sunday, 4 hours ago)
        now = datetime.utcnow()
        # Calculate last Sunday
        days_since_sunday = (now.weekday() + 1) % 7
        last_sunday = now - timedelta(days=days_since_sunday)
        change_time = last_sunday.replace(hour=14, minute=30) - timedelta(hours=4)

        scenario.record_beneficiary_change(
            beneficiary_id=vendor.beneficiary_id,
            account_id="ACC001",
            change_type="account_number",
            old_value="****8888",
            new_value="****4444",
            change_source="email_request",
            verified=False,
            timestamp=change_time,
            requestor_email="cfo@globaltech.com"
        )

        # High-value payment
        transaction = scenario.create_payment_transaction(
            transaction_id="TX005",
            account_id="ACC001",
            beneficiary_id=vendor.beneficiary_id,
            amount=50000.00,
            description="URGENT: Invoice Payment - Project Completion"
        )

        return scenario.evaluate_transaction(transaction)

    scenario.run_scenario(
        "Weekend Change + Same-Day High-Value Payment (EXTREME RISK)",
        scenario_5
    )

    print("\n" + "="*80)
    print("Demo Complete")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Same-day payments after account changes are CRITICAL risk indicators")
    print("2. Email/phone change requests require rigorous verification")
    print("3. Weekend/off-hours changes are suspicious")
    print("4. Multiple risk factors compound the fraud probability")
    print("5. Always verify account changes through secondary channels")
    print("\nBest Practices:")
    print("- Implement callback verification for all bank detail changes")
    print("- Require dual approval for changes + payments within 7 days")
    print("- Monitor for email domain spoofing (e.g., abc-supplies vs abc-supp1ies)")
    print("- Educate AP staff on BEC attack patterns")
    print("- Use authenticated portals for vendor communications")


if __name__ == "__main__":
    main()
