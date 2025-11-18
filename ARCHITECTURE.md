# Transaction Monitoring System - Architecture

## Overview

This is a **unified fraud detection platform** that integrates multiple fraud detection scenarios into a single monitoring system. All scenarios share the same infrastructure and contribute to a single risk assessment.

## Key Principle: Unified Integration

**All fraud scenarios are integrated at the root level** through a single `TransactionMonitor` class. When a transaction is evaluated:

1. **ALL rules from ALL scenarios** are evaluated simultaneously
2. A **single risk score** is calculated across all triggered rules
3. A **single decision** is made (approve/review/block)
4. **One dashboard** displays results from all scenarios

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         run.py                              │
│              UNIFIED TRANSACTION MONITOR                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         TransactionMonitor                          │   │
│  │                                                     │   │
│  │  Loads ALL rules from ALL scenarios into           │   │
│  │  a single RulesEngine                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   FRAUD SCENARIO MODULES              │
        │   (Each contributes rules)            │
        └───────────────────────────────────────┘
                            │
        ┌──────────┬────────┴────────┬──────────┐
        ▼          ▼                 ▼          ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Payroll     │ │   Credit     │ │    Wire      │ │   Account    │
│  Fraud       │ │   Card       │ │   Transfer   │ │   Takeover   │
│  Rules       │ │   Fraud      │ │   Fraud      │ │   Rules      │
│              │ │   Rules      │ │   Rules      │ │              │
│ • Unverified │ │ • Velocity   │ │ • Unusual    │ │ • Login      │
│   changes    │ │ • Location   │ │   dest.      │ │   patterns   │
│ • Weekend    │ │ • Amount     │ │ • High value │ │ • IP changes │
│   changes    │ │   patterns   │ │ • Rapid seq. │ │ • Device ID  │
│ • Rapid      │ │ • Merchant   │ │ • Foreign    │ │ • Session    │
│   changes    │ │   risk       │ │   country    │ │   anomalies  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │                │                │                │
        └────────────────┼────────────────┼────────────────┘
                         ▼
              ┌──────────────────────┐
              │   SHARED CORE        │
              │   INFRASTRUCTURE     │
              └──────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Rules       │ │  Risk        │ │  Decision    │
│  Engine      │ │  Scorer      │ │  Engine      │
│              │ │              │ │              │
│ Evaluates    │ │ Calculates   │ │ Determines   │
│ all rules    │ │ weighted     │ │ action based │
│ from all     │ │ risk score   │ │ on risk &    │
│ scenarios    │ │ (0-1 scale)  │ │ cost-benefit │
└──────────────┘ └──────────────┘ └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
              ┌──────────────────────┐
              │   DATABASE           │
              │                      │
              │ • Transactions       │
              │ • Risk Assessments   │
              │ • Accounts           │
              │ • Employees          │
              │ • Change History     │
              │ • Audit Logs         │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   DASHBOARD          │
              │                      │
              │ Shows aggregated     │
              │ results from ALL     │
              │ fraud scenarios      │
              └──────────────────────┘
```

## How Multiple Scenarios Integrate

### 1. Rule Registration (Startup)

When the system starts, `TransactionMonitor` loads rules from all scenarios:

```python
# In run.py - TransactionMonitor._load_all_rules()

# Load payroll fraud rules
payroll_rules = initialize_payroll_fraud_rules(db)
for rule in payroll_rules:
    self.rules_engine.add_rule(rule)

# Load credit card fraud rules
credit_rules = initialize_credit_fraud_rules(db)
for rule in credit_rules:
    self.rules_engine.add_rule(rule)

# Load wire fraud rules
wire_rules = initialize_wire_fraud_rules(db)
for rule in wire_rules:
    self.rules_engine.add_rule(rule)

# Result: ONE rules engine with ALL rules from ALL scenarios
```

### 2. Transaction Evaluation (Runtime)

When a transaction arrives, **ALL rules are evaluated**:

```python
# Single transaction evaluated against ALL scenarios
transaction = {
    "transaction_id": "TX123",
    "account_id": "ACC456",
    "amount": 5000.0,
    "transaction_type": "direct_deposit"
}

# This checks EVERY rule from EVERY scenario
result = monitor.evaluate_transaction(transaction)

# Result includes:
# - risk_score: 0.72 (calculated from ALL triggered rules)
# - triggered_rules: {
#     "payroll_recent_account_change": {...},
#     "payroll_unverified_change": {...},
#     "credit_unusual_amount": {...}  # Could trigger multiple scenarios!
#   }
# - decision: "manual_review"
```

### 3. Risk Score Calculation

The risk score is unified across all scenarios:

```
risk_score = sum(weight of ALL triggered rules from ALL scenarios)
             / sum(weight of ALL rules in system)
```

Example:
- Payroll rules triggered: weights = 4.0 + 3.5 = 7.5
- Credit rules triggered: weights = 2.0 = 2.0
- Total triggered weight: 9.5
- Total possible weight: 50.0 (all rules from all scenarios)
- **Risk Score = 9.5 / 50.0 = 0.19** (low-medium risk)

### 4. Dashboard Aggregation

The dashboard shows unified metrics:

```
Total Transactions: 1,247
Auto-Approved: 1,100 (88%)
Manual Review: 147 (12%)

Activity by Scenario:
  Payroll Fraud: 23 transactions, avg risk 0.45
  Credit Fraud: 89 transactions, avg risk 0.32
  Wire Fraud: 35 transactions, avg risk 0.51
```

## File Organization

```
transaction-monitoring/
├── run.py                          # MAIN ENTRY POINT - Integrates everything
├── ARCHITECTURE.md                 # This file
├── README.md                       # User documentation
│
├── app/
│   ├── models/
│   │   └── database.py            # Shared database models (all scenarios)
│   │
│   ├── services/                  # SHARED CORE SERVICES
│   │   ├── rules_engine.py        # Unified rules engine
│   │   ├── risk_scoring.py        # Unified risk scoring
│   │   ├── decision_engine.py     # Unified decision making
│   │   ├── context_provider.py    # Context gathering (all scenarios)
│   │   │
│   │   ├── payroll_fraud_rules.py    # Scenario-specific rules
│   │   ├── credit_fraud_rules.py     # Scenario-specific rules
│   │   ├── wire_fraud_rules.py       # Scenario-specific rules
│   │   └── account_takeover_rules.py # Scenario-specific rules
│   │
│   ├── scenarios/                 # Demo runners for each scenario
│   │   ├── payroll_reroute_scenario.py
│   │   ├── credit_fraud_scenario.py
│   │   └── wire_fraud_scenario.py
│   │
│   └── utils/
│       └── main.py                # Helper utilities
│
├── dashboard/
│   └── main.py                    # UNIFIED DASHBOARD - Shows all scenarios
│
├── config/
│   └── settings.py                # System-wide configuration
│
└── tests/
    ├── test_payroll_fraud.py      # Scenario-specific tests
    ├── test_credit_fraud.py
    └── test_integration.py        # Tests of unified system
```

## Data Flow Example

### Example: Employee Payroll Transaction

```
1. Transaction arrives:
   {
     "type": "direct_deposit",
     "amount": 7500,
     "employee_id": "EMP123"
   }

2. TransactionMonitor.evaluate_transaction() is called

3. Context is gathered:
   - Payroll context: Recent account changes? Last payroll date?
   - General context: Transaction history, velocity, patterns
   - Credit context: N/A (not a credit transaction)

4. ALL rules evaluated:
   ✓ payroll_recent_account_change: TRIGGERED (changed 5 days ago)
   ✓ payroll_unverified_change: TRIGGERED (no verification)
   ✓ payroll_high_value: TRIGGERED (>$5000)
   ✗ credit_unusual_merchant: NOT TRIGGERED (not a credit tx)
   ✗ wire_foreign_country: NOT TRIGGERED (not a wire tx)
   ... (all other rules evaluated)

5. Risk Score calculated:
   Triggered weights: 3.0 + 4.0 + 2.0 = 9.0
   Total weights: 50.0
   Risk Score: 9.0 / 50.0 = 0.18

6. Decision made:
   Score: 0.18 (below 0.3 threshold)
   BUT: High value + unverified change
   → Cost-benefit analysis: Review cost < Expected loss
   → Decision: "manual_review"

7. Result stored in database

8. Dashboard updated:
   - Overall stats incremented
   - Payroll scenario stats incremented
   - Review queue updated
```

## Adding New Fraud Scenarios

To add a new fraud scenario (e.g., "Account Takeover"):

### Step 1: Create Rule Module

```python
# app/services/account_takeover_rules.py

def create_ip_change_rule(db: Session, weight: float = 3.0) -> Rule:
    def condition(transaction, context):
        return context.get("ip_changed_recently", False)

    return Rule(
        name="account_takeover_ip_change",
        description="IP address changed recently",
        condition_func=condition,
        weight=weight
    )

def initialize_account_takeover_rules(db: Session) -> List[Rule]:
    return [
        create_ip_change_rule(db),
        create_device_change_rule(db),
        create_unusual_location_rule(db),
        # ... more rules
    ]
```

### Step 2: Register in run.py

```python
# In TransactionMonitor._load_all_rules()

from app.services.account_takeover_rules import initialize_account_takeover_rules

# Add to rule loading
takeover_rules = initialize_account_takeover_rules(self.db)
for rule in takeover_rules:
    self.rules_engine.add_rule(rule)
print(f"  ✓ Loaded {len(takeover_rules)} account takeover rules")
```

### Step 3: Add Context Provider (if needed)

```python
# In app/services/context_provider.py

def get_account_takeover_context(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Get account takeover specific context."""
    context = {}
    # Check IP changes, device fingerprints, etc.
    return context
```

### Step 4: Create Demo Scenario

```python
# app/scenarios/account_takeover_scenario.py

def main():
    # Demo various account takeover patterns
    pass
```

### Step 5: Add Tests

```python
# tests/test_account_takeover.py

class TestAccountTakeoverDetection(unittest.TestCase):
    # Test the new rules
    pass
```

**That's it!** The new scenario is now integrated with:
- ✅ Same rules engine
- ✅ Same risk scoring
- ✅ Same decision engine
- ✅ Same database
- ✅ Same dashboard

## Benefits of Unified Architecture

### 1. **Cross-Scenario Detection**
A single transaction can trigger rules from multiple scenarios:
```
Transaction: Wire transfer of $50,000
Triggers:
  - wire_high_value (Wire Fraud)
  - payroll_unusual_amount (Payroll Fraud - employee sending money out)
  - velocity_check (General fraud)
```

### 2. **Consistent Risk Scoring**
All scenarios use the same 0-1 risk scale and thresholds

### 3. **Single Source of Truth**
One database, one audit trail, one review queue

### 4. **Unified Monitoring**
Security team sees ALL fraud types in one dashboard

### 5. **Shared Infrastructure**
Don't duplicate rules engine, scoring, decision logic

### 6. **Easy Extensibility**
Adding new scenarios is just adding new rules to existing engine

## Production Deployment

In production, the system would:

1. **Listen to transaction queue** (Kafka, RabbitMQ, etc.)
2. **Evaluate each transaction** against all rules from all scenarios
3. **Store results** in database
4. **Send alerts** for high-risk transactions
5. **Update dashboard** in real-time
6. **Route to review queue** when manual review needed

```python
# Production mode (pseudocode)
monitor = TransactionMonitor(db)

def process_transaction_from_queue(transaction):
    result = monitor.evaluate_transaction(transaction)

    if result['decision'] == 'manual_review':
        send_alert_to_security_team(result)
        add_to_review_queue(result)

    update_dashboard_metrics(result)
    log_to_audit_trail(result)
```

## Summary

**The key insight:** This is NOT separate fraud detection systems working independently. This is ONE unified system where:

- Multiple fraud scenarios **contribute rules**
- ONE engine **evaluates all rules**
- ONE score **represents overall risk**
- ONE dashboard **shows everything**
- ONE decision **determines action**

Each scenario you build (payroll fraud, credit fraud, wire fraud, etc.) simply adds its rules to the shared pool, and they all work together to protect against fraud.
