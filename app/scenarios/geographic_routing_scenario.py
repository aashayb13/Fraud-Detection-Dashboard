# app/scenarios/geographic_routing_scenario.py
"""
Geographic Payment Routing Fraud Scenario

This module demonstrates the detection of fraud where payments are routed to
unexpected or high-risk countries.

Scenario:
1. A vendor has consistently been paid domestically (e.g., US bank accounts)
2. Fraudster compromises vendor's account or email
3. Fraudster sends invoice with updated banking information (foreign account)
4. Payment is routed to overseas account instead of domestic account
5. Real vendor never receives payment; fraud is discovered later

Detection Approach:
- Track vendor payment history by country
- Flag payments to high-risk/sanctioned countries
- Detect domestic-to-foreign routing changes
- Identify first international payments
- Monitor for multiple country changes in short timeframe
"""
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
import json

from app.models.database import (
    Account, Transaction, RiskAssessment, init_db, get_db
)
from app.services.rules_engine import RulesEngine
from app.services.risk_scoring import RiskScorer
from app.services.decision_engine import DecisionEngine
from app.services.context_provider import ContextProvider
from app.services.geographic_fraud_rules import initialize_geographic_fraud_rules


class GeographicRoutingScenario:
    """Handles geographic payment routing fraud detection scenario."""

    def __init__(self, db: Session):
        self.db = db

        # Initialize components
        self.rules_engine = RulesEngine()
        self.context_provider = ContextProvider(db, enable_chain_analysis=False)

        # Add geographic fraud rules
        geographic_rules = initialize_geographic_fraud_rules(db)
        for rule in geographic_rules:
            self.rules_engine.add_rule(rule)

        # Initialize risk scorer and decision engine
        self.risk_scorer = RiskScorer(self.rules_engine)
        self.decision_engine = DecisionEngine(self.risk_scorer)

    def create_account(self, account_id: str) -> Account:
        """Create a new account."""
        account = Account(
            account_id=account_id,
            creation_date=(datetime.utcnow() - timedelta(days=730)).isoformat(),
            risk_tier="medium",
            status="active"
        )
        self.db.add(account)
        self.db.commit()
        return account

    def create_vendor_payment(
        self,
        account_id: str,
        counterparty_id: str,
        amount: float,
        country: str,
        transaction_type: str = "WIRE",
        description: str = "Vendor Payment",
        timestamp: datetime = None
    ) -> Transaction:
        """Create a vendor payment transaction."""
        tx_id = f"TX_{uuid.uuid4().hex[:12].upper()}"

        metadata = {
            "country": country,
            "country_code": country,
            "bank_country": country,
            "payment_purpose": "vendor_payment"
        }

        transaction = Transaction(
            transaction_id=tx_id,
            timestamp=(timestamp or datetime.utcnow()).isoformat(),
            account_id=account_id,
            counterparty_id=counterparty_id,
            amount=amount,
            direction="debit",  # Outgoing payment
            transaction_type=transaction_type,
            description=description,
            tx_metadata=json.dumps(metadata)
        )
        self.db.add(transaction)
        self.db.commit()
        return transaction

    def evaluate_payment(self, transaction: Transaction) -> Dict[str, Any]:
        """Evaluate a payment for fraud."""
        tx_dict = {
            "transaction_id": transaction.transaction_id,
            "account_id": transaction.account_id,
            "counterparty_id": transaction.counterparty_id,
            "amount": transaction.amount,
            "direction": transaction.direction,
            "transaction_type": transaction.transaction_type,
            "description": transaction.description,
            "tx_metadata": transaction.tx_metadata,
            "timestamp": transaction.timestamp
        }

        # Get context
        context = self.context_provider.get_transaction_context(tx_dict)

        # Evaluate
        result = self.decision_engine.evaluate(tx_dict, context)

        # Store risk assessment
        self._store_risk_assessment(transaction.transaction_id, result)

        return result

    def _store_risk_assessment(self, transaction_id: str, result: Dict[str, Any]):
        """Store risk assessment in database."""
        assessment = RiskAssessment(
            assessment_id=f"RISK_{uuid.uuid4().hex[:12].upper()}",
            transaction_id=transaction_id,
            risk_score=result["risk_assessment"]["risk_score"],
            triggered_rules=json.dumps(list(result["risk_assessment"]["triggered_rules"].keys())),
            decision=result["decision"],
            review_status="pending" if result["decision"] == "manual_review" else "approved",
            review_timestamp=datetime.utcnow().isoformat()
        )
        self.db.add(assessment)
        self.db.commit()


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_transaction_result(tx: Transaction, result: Dict[str, Any]):
    """Print transaction evaluation result."""
    print(f"\nTransaction: {tx.transaction_id}")
    print(f"Amount: ${tx.amount:,.2f}")
    print(f"Counterparty: {tx.counterparty_id}")

    metadata = json.loads(tx.tx_metadata) if tx.tx_metadata else {}
    country = metadata.get("country", "Unknown")
    print(f"Country: {country}")

    risk_score = result["risk_assessment"]["risk_score"]
    print(f"Risk Score: {risk_score:.2f}")
    print(f"Decision: {result['decision'].upper()}")

    if result["risk_assessment"]["triggered_rules"]:
        print("\nTriggered Rules:")
        for rule_name, rule_info in result["risk_assessment"]["triggered_rules"].items():
            print(f"  - {rule_info['description']}")
            print(f"    Weight: {rule_info['weight']}")


def main():
    """Run the geographic routing fraud scenario demonstration."""
    print_section("GEOGRAPHIC PAYMENT ROUTING FRAUD DETECTION DEMO")

    # Initialize database
    init_db()
    db = next(get_db())

    try:
        scenario = GeographicRoutingScenario(db)

        # Create test account
        account_id = "ACC_COMPANY_001"
        scenario.create_account(account_id)
        print(f"\nCreated account: {account_id}")

        # Scenario 1: Normal domestic vendor payments
        print_section("Scenario 1: Established Domestic Vendor (Baseline)")

        vendor_id = "VENDOR_SUPPLIES_123"
        print(f"\nVendor: {vendor_id}")
        print("Establishing payment history (6 months of domestic payments)...")

        # Create 6 months of domestic payment history
        for i in range(6):
            days_ago = 30 * (6 - i)
            timestamp = datetime.utcnow() - timedelta(days=days_ago)
            tx = scenario.create_vendor_payment(
                account_id=account_id,
                counterparty_id=vendor_id,
                amount=2500.0 + (i * 100),
                country="US",
                description=f"Monthly supplies payment #{i+1}",
                timestamp=timestamp
            )
            print(f"  Payment {i+1}: ${tx.amount:,.2f} to US account ({days_ago} days ago)")

        # Evaluate a normal domestic payment
        print("\nEvaluating current month's payment (domestic, as usual):")
        normal_payment = scenario.create_vendor_payment(
            account_id=account_id,
            counterparty_id=vendor_id,
            amount=2800.0,
            country="US",
            description="Monthly supplies payment - Current"
        )
        result = scenario.evaluate_payment(normal_payment)
        print_transaction_result(normal_payment, result)

        # Scenario 2: Vendor suddenly paid through foreign account
        print_section("Scenario 2: FRAUD - Domestic Vendor Suddenly Paid Overseas")

        print(f"\nVendor {vendor_id} has ONLY been paid to US accounts")
        print("Fraudster compromised vendor email and sent updated banking details...")
        print("New bank account is in an overseas location!\n")

        suspicious_payment = scenario.create_vendor_payment(
            account_id=account_id,
            counterparty_id=vendor_id,
            amount=2900.0,
            country="RO",  # Romania - unexpected change
            description="Monthly supplies payment",
            transaction_type="WIRE"
        )

        result = scenario.evaluate_payment(suspicious_payment)
        print_transaction_result(suspicious_payment, result)

        # Scenario 3: Payment to high-risk country
        print_section("Scenario 3: FRAUD - Payment to High-Risk Country")

        vendor_id_2 = "VENDOR_CONSULTING_456"
        print(f"\nNew vendor: {vendor_id_2}")
        print("First payment is being routed to a sanctioned country!\n")

        high_risk_payment = scenario.create_vendor_payment(
            account_id=account_id,
            counterparty_id=vendor_id_2,
            amount=15000.0,
            country="IR",  # Iran - sanctioned country
            description="Consulting services",
            transaction_type="WIRE"
        )

        result = scenario.evaluate_payment(high_risk_payment)
        print_transaction_result(high_risk_payment, result)

        # Scenario 4: Multiple country changes in short time
        print_section("Scenario 4: FRAUD - Multiple Country Changes (Account Takeover)")

        vendor_id_3 = "VENDOR_SOFTWARE_789"
        print(f"\nVendor: {vendor_id_3}")
        print("Establishing payment history...")

        # Create history with consistent country
        for i in range(4):
            days_ago = 60 - (i * 15)
            timestamp = datetime.utcnow() - timedelta(days=days_ago)
            scenario.create_vendor_payment(
                account_id=account_id,
                counterparty_id=vendor_id_3,
                amount=5000.0,
                country="GB",  # Consistent UK payments
                description=f"Software license payment",
                timestamp=timestamp
            )

        print("  4 payments to GB (United Kingdom) over 60 days")
        print("\nSudden pattern change - payments to multiple countries in 20 days:")

        # Now create suspicious pattern - multiple countries rapidly
        countries = ["DE", "NL", "SG"]  # Germany, Netherlands, Singapore
        for i, country in enumerate(countries):
            days_ago = 20 - (i * 7)
            timestamp = datetime.utcnow() - timedelta(days=days_ago)
            tx = scenario.create_vendor_payment(
                account_id=account_id,
                counterparty_id=vendor_id_3,
                amount=5000.0,
                country=country,
                description=f"Software license payment",
                timestamp=timestamp
            )
            print(f"  Payment to {country} ({days_ago} days ago)")

        # Current suspicious payment
        print("\nEvaluating current payment (4th different country in 20 days):")
        multi_country_payment = scenario.create_vendor_payment(
            account_id=account_id,
            counterparty_id=vendor_id_3,
            amount=5000.0,
            country="HK",  # Hong Kong
            description="Software license payment"
        )

        result = scenario.evaluate_payment(multi_country_payment)
        print_transaction_result(multi_country_payment, result)

        # Summary
        print_section("DEMONSTRATION COMPLETE")
        print("\nKey Fraud Patterns Detected:")
        print("1. Domestic vendor suddenly paid through foreign account")
        print("2. Payment routed to high-risk/sanctioned country")
        print("3. Multiple country changes indicating account compromise")
        print("\nThese patterns help identify:")
        print("  - Business Email Compromise (BEC)")
        print("  - Vendor account takeover")
        print("  - Invoice fraud")
        print("  - Money laundering through geographic layering")

    finally:
        db.close()


if __name__ == "__main__":
    main()
