"""
Comprehensive Financial Fraud Detection & Compliance Dataset Generator

Generates synthetic data for:
- Historic transactions with risk scoring
- Customer profiles with KYC/AML/CDD status
- KYC verification events
- CDD/EDD investigation events
- Alert and analyst actions
- Rule performance and audit trails
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
import json

# Configuration
NUM_CUSTOMERS = 1000
NUM_TRANSACTIONS = 25000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 11, 7)

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Reference data
FIRST_NAMES = ["John", "Sarah", "Michael", "Emily", "David", "Jessica", "Robert", "Lisa",
               "James", "Maria", "William", "Jennifer", "Richard", "Linda", "Joseph", "Patricia",
               "Thomas", "Nancy", "Charles", "Karen", "Daniel", "Betty", "Matthew", "Margaret",
               "Anthony", "Sandra", "Donald", "Ashley", "Mark", "Dorothy"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
              "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Hall"]

CITIES = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
          "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
          "Austin, TX", "Jacksonville, FL", "Fort Worth, TX", "Columbus, OH", "San Francisco, CA",
          "Charlotte, NC", "Indianapolis, IN", "Seattle, WA", "Denver, CO", "Boston, MA"]

MERCHANT_CATEGORIES = ["Retail", "Restaurant", "Gas Station", "Grocery", "Online Shopping",
                       "Travel", "Entertainment", "Healthcare", "Utilities", "Insurance",
                       "Electronics", "Jewelry", "Wire Transfer", "Cryptocurrency", "Gambling",
                       "ATM Withdrawal", "Foreign Transaction", "Money Transfer"]

DEVICE_TYPES = ["Mobile - iOS", "Mobile - Android", "Desktop - Windows", "Desktop - Mac",
                "Tablet - iOS", "Tablet - Android", "POS Terminal", "ATM", "Web Browser"]

ANALYSTS = ["Sarah Chen", "Michael Rodriguez", "Emily Thompson", "David Kim", "Jessica Martinez",
            "Robert Johnson", "Lisa Anderson", "James Wilson", "Maria Garcia", "William Taylor"]

RULE_NAMES = [
    "amount_exceeds_10000",
    "velocity_5_in_24h",
    "odd_hours_transaction",
    "new_counterparty",
    "amount_deviation_3x",
    "payment_to_high_risk_country",
    "unexpected_country_routing",
    "phone_change_before_transfer_24h",
    "large_odd_hours_transaction_5000",
    "suspicious_chain_70",
    "layering_pattern_1",
    "money_mule_72h",
    "payroll_recent_account_change",
    "domestic_to_foreign_switch",
    "rapid_velocity_pattern",
    "high_risk_merchant_category",
    "unusual_device_login",
    "kyc_document_expired",
    "pep_match_detected",
    "sanctions_list_match"
]

def generate_customer_id():
    """Generate unique customer ID"""
    return f"CUST_{uuid.uuid4().hex[:8].upper()}"

def generate_transaction_id():
    """Generate unique transaction ID"""
    return f"TX_{uuid.uuid4().hex[:12].upper()}"

def generate_alert_id():
    """Generate unique alert ID"""
    return f"ALERT_{uuid.uuid4().hex[:10].upper()}"

def generate_kyc_event_id():
    """Generate unique KYC event ID"""
    return f"KYC_{uuid.uuid4().hex[:10].upper()}"

def generate_cdd_event_id():
    """Generate unique CDD event ID"""
    return f"CDD_{uuid.uuid4().hex[:10].upper()}"

def generate_edd_id():
    """Generate unique EDD investigation ID"""
    return f"EDD_{uuid.uuid4().hex[:10].upper()}"

def generate_rule_id():
    """Generate unique rule execution ID"""
    return f"RULE_{uuid.uuid4().hex[:10].upper()}"

def generate_customers(num_customers):
    """Generate customer profiles with KYC/AML/CDD status"""
    print(f"Generating {num_customers} customer profiles...")

    customers = []

    for i in range(num_customers):
        customer_id = generate_customer_id()
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)

        # Onboarding date between 2-3 years ago
        onboarding_date = START_DATE + timedelta(days=random.randint(0, 730))

        # Age between 18-75
        age = random.randint(18, 75)
        dob = datetime(random.randint(1949, 2007), random.randint(1, 12), random.randint(1, 28))

        # Customer segment with different risk profiles
        segment_choice = random.random()
        if segment_choice < 0.7:
            segment = "Retail"
            initial_risk = random.choices(["low", "medium", "high"], weights=[0.7, 0.25, 0.05])[0]
        elif segment_choice < 0.9:
            segment = "Small Business"
            initial_risk = random.choices(["low", "medium", "high"], weights=[0.5, 0.4, 0.1])[0]
        else:
            segment = "Corporate"
            initial_risk = random.choices(["low", "medium", "high"], weights=[0.4, 0.45, 0.15])[0]

        # PEP status (2% of customers)
        is_pep = random.random() < 0.02

        # KYC status
        kyc_status = random.choices(
            ["verified", "pending", "expired", "high-risk"],
            weights=[0.85, 0.08, 0.05, 0.02]
        )[0]

        # AML status
        aml_status = random.choices(
            ["clean", "prior_flag", "under_investigation", "cleared"],
            weights=[0.8, 0.12, 0.03, 0.05]
        )[0]

        # Current risk level (can evolve from initial)
        risk_evolution = random.random()
        if risk_evolution < 0.7:
            current_risk = initial_risk
        elif risk_evolution < 0.85:
            # Risk increased
            if initial_risk == "low":
                current_risk = random.choice(["medium", "high"])
            elif initial_risk == "medium":
                current_risk = "high"
            else:
                current_risk = "high"
        else:
            # Risk decreased
            if initial_risk == "high":
                current_risk = random.choice(["medium", "low"])
            elif initial_risk == "medium":
                current_risk = "low"
            else:
                current_risk = "low"

        # CDD review frequency based on risk
        if current_risk == "high" or is_pep:
            cdd_frequency = random.choice(["monthly", "quarterly"])
        elif current_risk == "medium":
            cdd_frequency = "quarterly"
        else:
            cdd_frequency = random.choice(["semi-annually", "annually"])

        # Last CDD review
        days_since_review = random.randint(0, 180)
        cdd_last_review = END_DATE - timedelta(days=days_since_review)

        # EDD required for high-risk customers
        edd_required = current_risk == "high" or is_pep or aml_status == "under_investigation"

        edd_reason = None
        if edd_required:
            edd_reasons = ["High transaction volume", "PEP status", "Sanctions screening match",
                          "Unusual transaction patterns", "High-risk jurisdiction",
                          "Large cash deposits", "Suspicious activity report filed"]
            edd_reason = random.choice(edd_reasons)

        # Source of funds
        sof_verified = random.random() < 0.9  # 90% verified

        customer = {
            "customer_id": customer_id,
            "full_name": f"{first_name} {last_name}",
            "date_of_birth": dob.strftime("%Y-%m-%d"),
            "address": f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Maple', 'Cedar'])} St, {random.choice(CITIES)}",
            "onboarding_date": onboarding_date.strftime("%Y-%m-%d"),
            "segment": segment,
            "KYC_status": kyc_status,
            "AML_status": aml_status,
            "risk_level_initial": initial_risk,
            "current_risk_level": current_risk,
            "PEP_status": "Y" if is_pep else "N",
            "cdd_last_review_date": cdd_last_review.strftime("%Y-%m-%d"),
            "cdd_review_frequency": cdd_frequency,
            "edd_required": "Y" if edd_required else "N",
            "edd_investigation_reason": edd_reason if edd_reason else "",
            "source_of_funds_verified": sof_verified,
            "account_balance": round(random.uniform(1000, 500000), 2) if segment != "Corporate" else round(random.uniform(50000, 5000000), 2)
        }

        customers.append(customer)

    return pd.DataFrame(customers)

def generate_transactions(customers_df, num_transactions):
    """Generate historic transactions"""
    print(f"Generating {num_transactions} transactions...")

    transactions = []
    customer_ids = customers_df['customer_id'].tolist()

    # Create risk profile mapping
    customer_risk_map = dict(zip(customers_df['customer_id'], customers_df['current_risk_level']))

    for i in range(num_transactions):
        transaction_id = generate_transaction_id()
        customer_id = random.choice(customer_ids)
        customer_risk = customer_risk_map[customer_id]

        # Transaction timestamp
        timestamp = START_DATE + timedelta(
            days=random.randint(0, (END_DATE - START_DATE).days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )

        # Amount based on customer risk (higher risk = more extreme amounts)
        if customer_risk == "high":
            if random.random() < 0.3:
                amount = round(random.uniform(10000, 100000), 2)  # Large transactions
            else:
                amount = round(random.uniform(100, 5000), 2)
        elif customer_risk == "medium":
            if random.random() < 0.1:
                amount = round(random.uniform(5000, 20000), 2)
            else:
                amount = round(random.uniform(50, 3000), 2)
        else:
            amount = round(random.uniform(10, 1000), 2)

        # Merchant category (high-risk customers more likely to use risky categories)
        if customer_risk == "high" and random.random() < 0.3:
            merchant_category = random.choice(["Wire Transfer", "Cryptocurrency", "Gambling",
                                               "Foreign Transaction", "Money Transfer"])
        else:
            merchant_category = random.choice(MERCHANT_CATEGORIES)

        # Location
        location = random.choice(CITIES)
        if customer_risk == "high" and random.random() < 0.2:
            # International high-risk location
            location = random.choice(["Moscow, Russia", "Lagos, Nigeria", "Caracas, Venezuela",
                                     "Tehran, Iran", "Pyongyang, North Korea", "Beijing, China"])

        # Device type
        device_type = random.choice(DEVICE_TYPES)

        # Risk score calculation
        base_risk = 0.1

        # Amount factor
        if amount > 10000:
            base_risk += 0.3
        elif amount > 5000:
            base_risk += 0.2

        # Customer risk factor
        if customer_risk == "high":
            base_risk += 0.3
        elif customer_risk == "medium":
            base_risk += 0.1

        # Merchant category factor
        if merchant_category in ["Wire Transfer", "Cryptocurrency", "Gambling", "Money Transfer"]:
            base_risk += 0.2

        # Location factor
        if "Russia" in location or "Nigeria" in location or "Iran" in location:
            base_risk += 0.25

        # Odd hours factor (10PM - 6AM)
        if timestamp.hour >= 22 or timestamp.hour < 6:
            base_risk += 0.15

        # Add some randomness
        risk_score = min(base_risk + random.uniform(-0.1, 0.1), 1.0)
        risk_score = max(risk_score, 0.0)
        risk_score = round(risk_score, 3)

        # Flag for review if risk score > 0.6
        flagged = risk_score > 0.6

        transaction = {
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "merchant_category": merchant_category,
            "location": location,
            "device_type": device_type,
            "risk_score": risk_score,
            "flagged_for_review": flagged
        }

        transactions.append(transaction)

    return pd.DataFrame(transactions)

def generate_kyc_events(customers_df):
    """Generate KYC verification events"""
    print("Generating KYC verification events...")

    kyc_events = []

    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        onboarding_date = datetime.strptime(customer['onboarding_date'], "%Y-%m-%d")

        # Initial KYC at onboarding
        initial_docs = ["Passport", "Driver License", "Utility Bill", "Bank Statement"]
        for doc_type in random.sample(initial_docs, random.randint(2, 4)):
            kyc_event = {
                "kyc_event_id": generate_kyc_event_id(),
                "customer_id": customer_id,
                "kyc_check_date": onboarding_date.strftime("%Y-%m-%d"),
                "document_type": doc_type,
                "document_verified": random.random() < 0.95,
                "reviewer": random.choice(ANALYSTS),
                "result": random.choices(["approved", "rejected", "pending"],
                                        weights=[0.9, 0.05, 0.05])[0],
                "expiry_date": (onboarding_date + timedelta(days=random.randint(365, 1825))).strftime("%Y-%m-%d"),
                "notes": ""
            }
            kyc_events.append(kyc_event)

        # Periodic re-verification events
        current_date = onboarding_date
        while current_date < END_DATE:
            # Check every 6-18 months
            current_date += timedelta(days=random.randint(180, 540))
            if current_date < END_DATE:
                doc_type = random.choice(["Passport", "Driver License", "Proof of Address", "Bank Statement"])
                kyc_event = {
                    "kyc_event_id": generate_kyc_event_id(),
                    "customer_id": customer_id,
                    "kyc_check_date": current_date.strftime("%Y-%m-%d"),
                    "document_type": doc_type,
                    "document_verified": random.random() < 0.92,
                    "reviewer": random.choice(ANALYSTS),
                    "result": random.choices(["approved", "rejected", "pending", "expired"],
                                            weights=[0.85, 0.05, 0.05, 0.05])[0],
                    "expiry_date": (current_date + timedelta(days=random.randint(365, 1825))).strftime("%Y-%m-%d"),
                    "notes": ""
                }
                kyc_events.append(kyc_event)

    return pd.DataFrame(kyc_events)

def generate_cdd_edd_events(customers_df):
    """Generate CDD/EDD events showing risk evolution"""
    print("Generating CDD/EDD events...")

    cdd_events = []
    edd_actions = []

    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        onboarding_date = datetime.strptime(customer['onboarding_date'], "%Y-%m-%d")
        current_risk = customer['current_risk_level']
        initial_risk = customer['risk_level_initial']

        # Initial CDD at onboarding
        cdd_event = {
            "cdd_event_id": generate_cdd_event_id(),
            "customer_id": customer_id,
            "event_date": onboarding_date.strftime("%Y-%m-%d"),
            "event_type": "initial_risk_assessment",
            "new_risk_score": 0.3 if initial_risk == "low" else 0.6 if initial_risk == "medium" else 0.85,
            "previous_risk_level": "",
            "new_risk_level": initial_risk,
            "summary": f"Initial onboarding risk assessment - Classified as {initial_risk} risk",
            "edd_id": "",
            "triggered_by": "Onboarding System",
            "reviewer": random.choice(ANALYSTS)
        }
        cdd_events.append(cdd_event)

        # Generate CDD review events based on frequency
        review_frequency_days = {
            "monthly": 30,
            "quarterly": 90,
            "semi-annually": 180,
            "annually": 365
        }

        freq = customer['cdd_review_frequency']
        days_between_reviews = review_frequency_days.get(freq, 90)

        current_date = onboarding_date + timedelta(days=days_between_reviews)
        prev_risk = initial_risk

        while current_date < END_DATE:
            # Determine if risk level changes
            risk_change_probability = 0.15 if current_risk != prev_risk else 0.05

            if random.random() < risk_change_probability:
                event_type = "risk_level_change"
                possible_changes = {
                    "low": ["low", "medium"],
                    "medium": ["low", "medium", "high"],
                    "high": ["medium", "high"]
                }
                new_risk = random.choice(possible_changes[prev_risk])
                summary = f"Risk level {'increased' if new_risk != prev_risk else 'maintained'} during periodic CDD review"
            else:
                event_type = "periodic_review"
                new_risk = prev_risk
                summary = f"Routine CDD review - Risk level maintained at {prev_risk}"

            new_risk_score = 0.3 if new_risk == "low" else 0.6 if new_risk == "medium" else 0.85
            new_risk_score += random.uniform(-0.1, 0.1)
            new_risk_score = round(max(0, min(1, new_risk_score)), 3)

            cdd_event = {
                "cdd_event_id": generate_cdd_event_id(),
                "customer_id": customer_id,
                "event_date": current_date.strftime("%Y-%m-%d"),
                "event_type": event_type,
                "new_risk_score": new_risk_score,
                "previous_risk_level": prev_risk,
                "new_risk_level": new_risk,
                "summary": summary,
                "edd_id": "",
                "triggered_by": "Automated CDD Schedule" if event_type == "periodic_review" else "Risk Model",
                "reviewer": random.choice(ANALYSTS)
            }
            cdd_events.append(cdd_event)

            # Trigger EDD if risk is high or specific conditions met
            if new_risk == "high" or (new_risk == "medium" and random.random() < 0.1):
                edd_id = generate_edd_id()
                investigation_start = current_date
                investigation_end = current_date + timedelta(days=random.randint(7, 45))

                edd_reasons = [
                    "Elevated risk score",
                    "Unusual transaction patterns detected",
                    "Large value transactions",
                    "Geographic risk factors",
                    "PEP association identified",
                    "Sanctions screening alert",
                    "Source of funds verification required"
                ]

                edd_steps = [
                    "Enhanced document collection",
                    "Source of wealth verification",
                    "Beneficial ownership analysis",
                    "Transaction pattern analysis",
                    "External database screening",
                    "Management review",
                    "Regulatory filing assessment"
                ]

                edd_outcomes = random.choices(
                    ["cleared", "cleared_with_conditions", "restrictions_imposed", "account_closed"],
                    weights=[0.6, 0.25, 0.1, 0.05]
                )[0]

                edd_action = {
                    "edd_id": edd_id,
                    "customer_id": customer_id,
                    "investigation_start": investigation_start.strftime("%Y-%m-%d"),
                    "investigation_end": investigation_end.strftime("%Y-%m-%d") if investigation_end < END_DATE else "",
                    "edd_reason": random.choice(edd_reasons),
                    "steps_taken": ", ".join(random.sample(edd_steps, random.randint(3, 6))),
                    "outcome": edd_outcomes,
                    "notes": f"EDD investigation completed. Customer {edd_outcomes.replace('_', ' ')}.",
                    "senior_reviewer": random.choice(ANALYSTS),
                    "regulatory_filing": "SAR Filed" if random.random() < 0.1 else "No filing required"
                }
                edd_actions.append(edd_action)

                # Link EDD to CDD event
                cdd_event["edd_id"] = edd_id
                cdd_event["summary"] += f" - EDD investigation initiated"

            prev_risk = new_risk
            current_date += timedelta(days=days_between_reviews + random.randint(-15, 15))

    return pd.DataFrame(cdd_events), pd.DataFrame(edd_actions)

def generate_alerts_and_analyst_actions(transactions_df, customers_df):
    """Generate alerts and analyst review actions"""
    print("Generating alerts and analyst actions...")

    alerts = []

    # Filter flagged transactions
    flagged_txns = transactions_df[transactions_df['flagged_for_review'] == True]

    customer_risk_map = dict(zip(customers_df['customer_id'], customers_df['current_risk_level']))

    for _, txn in flagged_txns.iterrows():
        alert_id = generate_alert_id()
        customer_risk = customer_risk_map.get(txn['customer_id'], 'low')

        # Alert type based on transaction characteristics
        alert_types = []
        if txn['risk_score'] > 0.8:
            alert_types.append("critical_high_risk")
        if txn['amount'] > 10000:
            alert_types.append("large_transaction")
        if "Transfer" in txn['merchant_category'] or "Cryptocurrency" in txn['merchant_category']:
            alert_types.append("high_risk_merchant")
        if customer_risk == "high":
            alert_types.append("high_risk_customer")
        if any(country in txn['location'] for country in ["Russia", "Nigeria", "Iran", "Venezuela"]):
            alert_types.append("geographic_risk")

        if not alert_types:
            alert_types = ["general_risk_alert"]

        alert_type = ", ".join(alert_types[:2])  # Take top 2 alert types

        # Alert timestamp = transaction timestamp
        alert_timestamp = datetime.strptime(txn['timestamp'], "%Y-%m-%d %H:%M:%S")

        # Analyst decision timing
        decision_delay_hours = random.randint(1, 72) if txn['risk_score'] < 0.8 else random.randint(0, 12)
        decision_timestamp = alert_timestamp + timedelta(hours=decision_delay_hours)

        # Analyst decision
        analyst = random.choice(ANALYSTS)

        # Decision based on risk score
        if txn['risk_score'] > 0.85:
            decision = random.choices(
                ["escalate", "deny", "SAR_filed", "approve_with_monitoring"],
                weights=[0.3, 0.3, 0.2, 0.2]
            )[0]
        elif txn['risk_score'] > 0.7:
            decision = random.choices(
                ["approve_with_monitoring", "escalate", "deny"],
                weights=[0.5, 0.3, 0.2]
            )[0]
        else:
            decision = random.choices(
                ["approve", "approve_with_monitoring", "escalate"],
                weights=[0.6, 0.3, 0.1]
            )[0]

        # Generate notes
        notes_templates = [
            f"Reviewed transaction for {txn['merchant_category']} - {decision.replace('_', ' ')}",
            f"Customer verification completed. Amount ${txn['amount']:.2f} - {decision.replace('_', ' ')}",
            f"Risk score {txn['risk_score']:.3f} reviewed. {decision.replace('_', ' ')}",
            f"Location check performed: {txn['location']}. {decision.replace('_', ' ')}"
        ]
        notes = random.choice(notes_templates)

        if decision == "SAR_filed":
            notes += " - Suspicious Activity Report filed with FinCEN"

        alert = {
            "alert_id": alert_id,
            "transaction_id": txn['transaction_id'],
            "customer_id": txn['customer_id'],
            "alert_type": alert_type,
            "alert_timestamp": alert_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "analyst_id": analyst,
            "analyst_decision": decision,
            "decision_timestamp": decision_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "time_to_decision_hours": decision_delay_hours,
            "notes": notes,
            "false_positive": random.random() < 0.3 if decision == "approve" else False
        }

        alerts.append(alert)

    return pd.DataFrame(alerts)

def generate_rule_performance_audit(transactions_df, alerts_df):
    """Generate rule performance and audit trail"""
    print("Generating rule performance and audit trail...")

    rule_executions = []

    # For each transaction, simulate rule evaluations
    for _, txn in transactions_df.iterrows():
        num_rules_evaluated = random.randint(5, 15)
        selected_rules = random.sample(RULE_NAMES, num_rules_evaluated)

        for rule_name in selected_rules:
            # Determine if rule triggered based on transaction characteristics
            triggered = False
            rule_score = 0.0

            if rule_name == "amount_exceeds_10000" and txn['amount'] > 10000:
                triggered = True
                rule_score = 0.2
            elif rule_name == "odd_hours_transaction":
                hour = int(txn['timestamp'].split()[1].split(':')[0])
                if hour >= 22 or hour < 6:
                    triggered = True
                    rule_score = 0.15
            elif rule_name == "payment_to_high_risk_country":
                if any(country in txn['location'] for country in ["Russia", "Nigeria", "Iran"]):
                    triggered = True
                    rule_score = 0.35
            elif "Transfer" in rule_name.lower() or "velocity" in rule_name.lower():
                triggered = random.random() < 0.1
                rule_score = random.uniform(0.1, 0.3) if triggered else 0
            else:
                triggered = random.random() < 0.05
                rule_score = random.uniform(0.05, 0.25) if triggered else 0

            # Decision outcome
            if triggered:
                if txn['flagged_for_review']:
                    decision_outcome = "true_positive" if random.random() < 0.7 else "false_positive"
                else:
                    decision_outcome = "false_negative"
            else:
                decision_outcome = "true_negative"

            rule_execution = {
                "rule_execution_id": generate_rule_id(),
                "rule_name": rule_name,
                "transaction_id": txn['transaction_id'],
                "customer_id": txn['customer_id'],
                "rule_triggered": triggered,
                "rule_score": round(rule_score, 3),
                "decision_outcome": decision_outcome,
                "timestamp": txn['timestamp'],
                "execution_time_ms": random.randint(5, 150),
                "rule_version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}"
            }

            rule_executions.append(rule_execution)

    # Generate audit trail for rule changes and manual overrides
    audit_trail = []

    # Rule updates
    for rule_name in RULE_NAMES:
        num_updates = random.randint(1, 5)
        update_date = START_DATE

        for _ in range(num_updates):
            update_date += timedelta(days=random.randint(60, 180))
            if update_date < END_DATE:
                audit_entry = {
                    "audit_id": f"AUDIT_{uuid.uuid4().hex[:10].upper()}",
                    "timestamp": update_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "audit_action": "rule_updated",
                    "entity_type": "rule",
                    "entity_id": rule_name,
                    "performed_by": random.choice(ANALYSTS),
                    "description": f"Rule {rule_name} threshold updated",
                    "previous_value": f"{random.uniform(0.5, 0.8):.2f}",
                    "new_value": f"{random.uniform(0.5, 0.8):.2f}",
                    "reason": "Performance optimization based on false positive rate"
                }
                audit_trail.append(audit_entry)

    # Manual overrides from alerts
    for _, alert in alerts_df.iterrows():
        if alert['analyst_decision'] in ['escalate', 'deny', 'SAR_filed']:
            audit_entry = {
                "audit_id": f"AUDIT_{uuid.uuid4().hex[:10].upper()}",
                "timestamp": alert['decision_timestamp'],
                "audit_action": "manual_override",
                "entity_type": "alert",
                "entity_id": alert['alert_id'],
                "performed_by": alert['analyst_id'],
                "description": f"Manual decision: {alert['analyst_decision']}",
                "previous_value": "auto_flag",
                "new_value": alert['analyst_decision'],
                "reason": alert['notes']
            }
            audit_trail.append(audit_entry)

    return pd.DataFrame(rule_executions), pd.DataFrame(audit_trail)

def main():
    """Main execution function"""
    print("=" * 80)
    print("FINANCIAL FRAUD DETECTION & COMPLIANCE DATASET GENERATOR")
    print("=" * 80)
    print()

    # Generate all datasets
    customers_df = generate_customers(NUM_CUSTOMERS)
    transactions_df = generate_transactions(customers_df, NUM_TRANSACTIONS)
    kyc_events_df = generate_kyc_events(customers_df)
    cdd_events_df, edd_actions_df = generate_cdd_edd_events(customers_df)
    alerts_df = generate_alerts_and_analyst_actions(transactions_df, customers_df)
    rule_executions_df, audit_trail_df = generate_rule_performance_audit(transactions_df, alerts_df)

    # Save to CSV
    print("\nSaving datasets to CSV files...")

    output_dir = "compliance_dataset"
    import os
    os.makedirs(output_dir, exist_ok=True)

    customers_df.to_csv(f"{output_dir}/customer_profiles.csv", index=False)
    transactions_df.to_csv(f"{output_dir}/transactions.csv", index=False)
    kyc_events_df.to_csv(f"{output_dir}/kyc_events.csv", index=False)
    cdd_events_df.to_csv(f"{output_dir}/cdd_events.csv", index=False)
    edd_actions_df.to_csv(f"{output_dir}/edd_actions.csv", index=False)
    alerts_df.to_csv(f"{output_dir}/alerts_analyst_actions.csv", index=False)
    rule_executions_df.to_csv(f"{output_dir}/rule_executions.csv", index=False)
    audit_trail_df.to_csv(f"{output_dir}/audit_trail.csv", index=False)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Dataset Summary:")
    print(f"   Customers: {len(customers_df):,}")
    print(f"   Transactions: {len(transactions_df):,}")
    print(f"   KYC Events: {len(kyc_events_df):,}")
    print(f"   CDD Events: {len(cdd_events_df):,}")
    print(f"   EDD Investigations: {len(edd_actions_df):,}")
    print(f"   Alerts: {len(alerts_df):,}")
    print(f"   Rule Executions: {len(rule_executions_df):,}")
    print(f"   Audit Trail Entries: {len(audit_trail_df):,}")

    print(f"\n📁 Files saved to: ./{output_dir}/")
    print("\n🎯 Customer Risk Distribution:")
    print(customers_df['current_risk_level'].value_counts())
    print("\n🚨 Alert Decision Distribution:")
    print(alerts_df['analyst_decision'].value_counts())
    print("\n✅ Transaction Review Rate:")
    flagged_rate = (transactions_df['flagged_for_review'].sum() / len(transactions_df)) * 100
    print(f"   {flagged_rate:.2f}% of transactions flagged for review")

    print("\n" + "=" * 80)
    print("✅ All datasets generated successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
