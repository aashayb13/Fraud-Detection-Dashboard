"""Enhanced test data generator with realistic fraud scenarios"""
from datetime import datetime, timedelta
import random
import json
from app.models.database import init_db, get_db, Transaction, RiskAssessment, Account

# Realistic rule definitions matching the actual system
RULE_CATALOG = {
    "payment_to_high_risk_country": {"weight": 3.5, "description": "Payment routed to high-risk or sanctioned country"},
    "unexpected_country_routing": {"weight": 2.5, "description": "Payment routed to unexpected country based on vendor history"},
    "domestic_to_foreign_switch": {"weight": 3.0, "description": "Domestic-only vendor suddenly paid through foreign account"},
    "immediate_transfer_after_phone_change_1h": {"weight": 5.0, "description": "Outgoing transfer within 1 hour of phone change - critical account takeover alert"},
    "phone_change_before_transfer_24h": {"weight": 3.5, "description": "Outgoing transfer within 24 hours of phone/device change - possible account takeover"},
    "large_transfer_after_phone_change_5000": {"weight": 4.0, "description": "Large transfer (>=$5,000.00) within 48h of phone change - high-risk takeover"},
    "amount_exceeds_10000": {"weight": 2.0, "description": "Transaction amount exceeds $10,000.00"},
    "velocity_5_in_24h": {"weight": 1.5, "description": "More than 5 transactions in 24 hours"},
    "amount_deviation_3x": {"weight": 2.0, "description": "Transaction amount deviates from average by 3x"},
    "new_counterparty": {"weight": 1.0, "description": "Transaction with a new counterparty"},
    "payroll_recent_account_change": {"weight": 3.0, "description": "Payroll transaction to bank account changed within 30 days"},
    "payroll_unverified_account_change": {"weight": 4.0, "description": "Payroll transaction to account with unverified banking information changes"},
    "odd_hours_transaction": {"weight": 2.0, "description": "Transaction initiated during odd hours (22:00 - 06:00)"},
    "large_odd_hours_transaction_5000": {"weight": 3.5, "description": "Large transaction (>= $5,000.00) initiated during odd hours"},
    "odd_hours_pattern_deviation": {"weight": 4.0, "description": "Transaction at odd hours deviates significantly from customer's normal activity pattern"},
    "suspicious_chain_70": {"weight": 2.0, "description": "Suspicious transaction chain detected"},
    "layering_pattern_1": {"weight": 2.0, "description": "Layering pattern detected - multiple small credits consolidated"},
    "money_mule_72h": {"weight": 2.0, "description": "Money mule pattern detected: 5+ small incoming payments, 70%+ flow-through"},
}

def generate_scenario_transaction(scenario_type, account_id, base_time, tx_id):
    """Generate transaction based on fraud scenario"""

    if scenario_type == "clean":
        # Clean transaction - no red flags
        return {
            "transaction_id": f"TX_{tx_id:06d}",
            "account_id": account_id,
            "amount": random.uniform(100, 1000),
            "direction": "credit",
            "transaction_type": "ACH",
            "timestamp": base_time.isoformat(),
            "counterparty_id": f"COUNTER_{random.randint(1, 20):04d}",
            "description": "Regular payment",
            "triggered_rules": {}
        }

    elif scenario_type == "high_value":
        # High value transaction
        amount = random.uniform(15000, 50000)
        triggered_rules = {}
        if amount > 10000:
            triggered_rules["amount_exceeds_10000"] = RULE_CATALOG["amount_exceeds_10000"]

        return {
            "transaction_id": f"TX_{tx_id:06d}",
            "account_id": account_id,
            "amount": amount,
            "direction": "debit",
            "transaction_type": "WIRE",
            "timestamp": base_time.isoformat(),
            "counterparty_id": f"COUNTER_{random.randint(1, 50):04d}",
            "description": "Large wire transfer",
            "triggered_rules": triggered_rules
        }

    elif scenario_type == "odd_hours":
        # Odd hours transaction (10 PM - 6 AM)
        odd_hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])
        odd_time = base_time.replace(hour=odd_hour, minute=random.randint(0, 59))
        amount = random.uniform(5000, 25000)

        triggered_rules = {
            "odd_hours_transaction": RULE_CATALOG["odd_hours_transaction"],
            "large_odd_hours_transaction_5000": RULE_CATALOG["large_odd_hours_transaction_5000"]
        }

        if amount > 10000:
            triggered_rules["amount_exceeds_10000"] = RULE_CATALOG["amount_exceeds_10000"]

        return {
            "transaction_id": f"TX_{tx_id:06d}",
            "account_id": account_id,
            "amount": amount,
            "direction": "debit",
            "transaction_type": "WIRE",
            "timestamp": odd_time.isoformat(),
            "counterparty_id": f"COUNTER_{random.randint(1, 50):04d}",
            "description": "Late night wire transfer",
            "triggered_rules": triggered_rules
        }

    elif scenario_type == "account_takeover":
        # Account takeover scenario
        amount = random.uniform(8000, 30000)
        triggered_rules = {
            "immediate_transfer_after_phone_change_1h": RULE_CATALOG["immediate_transfer_after_phone_change_1h"],
            "large_transfer_after_phone_change_5000": RULE_CATALOG["large_transfer_after_phone_change_5000"],
            "new_counterparty": RULE_CATALOG["new_counterparty"]
        }

        if amount > 10000:
            triggered_rules["amount_exceeds_10000"] = RULE_CATALOG["amount_exceeds_10000"]

        return {
            "transaction_id": f"TX_{tx_id:06d}",
            "account_id": account_id,
            "amount": amount,
            "direction": "debit",
            "transaction_type": "WIRE",
            "timestamp": base_time.isoformat(),
            "counterparty_id": f"COUNTER_{random.randint(100, 150):04d}",
            "description": "Transfer to new account after phone change",
            "triggered_rules": triggered_rules
        }

    elif scenario_type == "geographic_fraud":
        # Geographic routing fraud
        amount = random.uniform(5000, 20000)
        triggered_rules = {
            "payment_to_high_risk_country": RULE_CATALOG["payment_to_high_risk_country"],
            "unexpected_country_routing": RULE_CATALOG["unexpected_country_routing"]
        }

        if amount > 10000:
            triggered_rules["amount_exceeds_10000"] = RULE_CATALOG["amount_exceeds_10000"]

        return {
            "transaction_id": f"TX_{tx_id:06d}",
            "account_id": account_id,
            "amount": amount,
            "direction": "debit",
            "transaction_type": "WIRE",
            "timestamp": base_time.isoformat(),
            "counterparty_id": f"COUNTER_{random.randint(1, 50):04d}",
            "description": "International wire to high-risk country",
            "triggered_rules": triggered_rules
        }

    elif scenario_type == "velocity":
        # High velocity
        amount = random.uniform(500, 2000)
        triggered_rules = {
            "velocity_5_in_24h": RULE_CATALOG["velocity_5_in_24h"]
        }

        return {
            "transaction_id": f"TX_{tx_id:06d}",
            "account_id": account_id,
            "amount": amount,
            "direction": "debit",
            "transaction_type": "ACH",
            "timestamp": base_time.isoformat(),
            "counterparty_id": f"COUNTER_{random.randint(1, 50):04d}",
            "description": "Rapid succession transaction",
            "triggered_rules": triggered_rules
        }

    else:
        return generate_scenario_transaction("clean", account_id, base_time, tx_id)


def calculate_risk_score(triggered_rules):
    """Calculate risk score based on triggered rule weights"""
    if not triggered_rules:
        return 0.05

    total_weight = sum(rule["weight"] for rule in triggered_rules.values())
    # Normalize against total possible weight (approximate)
    total_possible = 25.0  # Approximate max weight from all rules
    risk_score = min(total_weight / total_possible, 1.0)

    return risk_score


def generate_data():
    # Initialize database
    init_db()
    db = next(get_db())

    # Clear old data
    db.query(RiskAssessment).delete()
    db.query(Transaction).delete()
    db.query(Account).delete()
    db.commit()

    print("Generating realistic test data with fraud scenarios...")

    # Create accounts
    accounts = []
    for i in range(10):
        account = Account(
            account_id=f"ACC_{i:04d}",
            creation_date=datetime.utcnow().isoformat(),
            risk_tier="medium",
            status="active"
        )
        db.add(account)
        accounts.append(account)

    db.commit()
    print(f"Created {len(accounts)} accounts")

    # Create transactions with realistic scenarios
    scenario_distribution = [
        ("clean", 40),  # 40% clean transactions
        ("high_value", 15),  # 15% high value
        ("odd_hours", 15),  # 15% odd hours
        ("account_takeover", 10),  # 10% account takeover
        ("geographic_fraud", 10),  # 10% geographic fraud
        ("velocity", 10),  # 10% velocity
    ]

    scenarios = []
    for scenario, count in scenario_distribution:
        scenarios.extend([scenario] * count)

    random.shuffle(scenarios)

    base_time = datetime.utcnow()

    for i, scenario in enumerate(scenarios):
        account = random.choice(accounts)
        tx_time = base_time - timedelta(hours=random.randint(0, 48))

        tx_data = generate_scenario_transaction(scenario, account.account_id, tx_time, i)

        tx = Transaction(
            transaction_id=tx_data["transaction_id"],
            account_id=tx_data["account_id"],
            amount=tx_data["amount"],
            direction=tx_data["direction"],
            transaction_type=tx_data["transaction_type"],
            timestamp=tx_data["timestamp"],
            counterparty_id=tx_data["counterparty_id"],
            description=tx_data["description"]
        )
        db.add(tx)

        # Create risk assessment with realistic scoring
        triggered_rules = tx_data["triggered_rules"]
        risk_score = calculate_risk_score(triggered_rules)

        # Add some randomness
        risk_score = min(risk_score + random.uniform(-0.05, 0.05), 1.0)
        risk_score = max(risk_score, 0.0)

        # Determine decision
        if risk_score < 0.3:
            decision = "auto_approve"
        elif risk_score < 0.6:
            decision = "manual_review"
        else:
            decision = "manual_review"  # High risk always goes to review

        assessment = RiskAssessment(
            assessment_id=f"RISK_{i:06d}",
            transaction_id=tx.transaction_id,
            risk_score=risk_score,
            decision=decision,
            triggered_rules=json.dumps(triggered_rules),
            review_status="pending" if decision == "manual_review" else "approved",
            review_timestamp=tx.timestamp
        )
        db.add(assessment)

    db.commit()

    # Print statistics
    print(f"\nCreated {len(scenarios)} transactions with risk assessments:")
    for scenario, count in scenario_distribution:
        print(f"  - {scenario}: {count} transactions")

    print("\nDone! Refresh your dashboard to see the data.")

if __name__ == "__main__":
    generate_data()
