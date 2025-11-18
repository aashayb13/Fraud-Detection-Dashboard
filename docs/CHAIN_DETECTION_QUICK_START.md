# Chain Detection - Quick Start Guide

## 🎯 Quick Setup (3 Steps)

### 1. Enable Chain Analysis

```python
from app.services.context_provider import ContextProvider

# Enable chain detection when creating context provider
context_provider = ContextProvider(db, enable_chain_analysis=True)
```

### 2. Add Chain Detection Rules

```python
from app.services.rules_engine import RulesEngine
from app.services.fraud_rules import get_balanced_chain_rules

rules_engine = RulesEngine()

# Add all balanced chain detection rules
for rule in get_balanced_chain_rules():
    rules_engine.add_rule(rule)
```

### 3. Evaluate Transactions

```python
# Get context with chain analysis
context = context_provider.get_transaction_context(transaction)

# Evaluate rules
triggered_rules = rules_engine.evaluate_all(transaction, context)

# Check for chain-related triggers
if triggered_rules:
    print(f"⚠️ Triggered {len(triggered_rules)} rules")
```

## 📊 What Gets Detected

| Pattern | Description | Example |
|---------|-------------|---------|
| **Credit-Refund-Transfer** | Credit → Refund → Transfer to different party | Hiding illicit funds origin |
| **Layering** | Multiple small credits → Large transfer | Breaking up source, consolidating |
| **Rapid Reversals** | Credit → Quick refund (< 6 hours) | Testing accounts, rapid movement |

## ⚙️ Rule Sets

Choose based on your risk tolerance:

```python
from app.services.fraud_rules import (
    get_high_security_chain_rules,    # More sensitive, more false positives
    get_balanced_chain_rules,         # Recommended for most use cases
    get_permissive_chain_rules        # Less sensitive, fewer false positives
)

# Example: High security environment
for rule in get_high_security_chain_rules():
    rules_engine.add_rule(rule)
```

## 🔍 Analyzing Results

```python
# Get chain analysis details
chain_analysis = context.get("chain_analysis", {})

print(f"Suspicious chains: {chain_analysis.get('has_suspicious_chains')}")
print(f"Chain count: {chain_analysis.get('chain_count')}")
print(f"Max suspicion: {chain_analysis.get('max_chain_suspicion'):.2f}")

# Breakdown by type
print(f"Credit-refund chains: {chain_analysis.get('credit_refund_count')}")
print(f"Layering patterns: {chain_analysis.get('layering_pattern_count')}")
print(f"Rapid reversals: {chain_analysis.get('rapid_reversal_count')}")
```

## 🎛️ Custom Configuration

### Create Individual Rules

```python
from app.services.fraud_rules import (
    create_suspicious_chain_rule,
    create_credit_refund_transfer_rule,
    create_layering_pattern_rule,
    create_rapid_reversal_rule
)

# Custom thresholds
rule = create_suspicious_chain_rule(
    suspicion_threshold=0.8,  # Only high-confidence triggers
    weight=3.0                # Higher impact on risk score
)

rules_engine.add_rule(rule)
```

### Adjust ChainAnalyzer Settings

```python
from app.services.chain_analyzer import ChainAnalyzer

analyzer = ChainAnalyzer(db)

# Customize detection parameters
analyzer.CHAIN_LOOKBACK_HOURS = 48        # Check last 48 hours (default: 72)
analyzer.SMALL_TRANSACTION_THRESHOLD = 50  # $50 is "small" (default: 100)
analyzer.RAPID_TIMEFRAME_HOURS = 3        # 3 hours is "rapid" (default: 6)
```

## 🧪 Testing Your Setup

```bash
# Validate logic (no dependencies required)
python tests/validate_chain_logic.py

# Full integration tests (requires SQLAlchemy)
pytest tests/test_chain_detection.py -v
```

## 📈 Integration with Existing System

Chain detection works seamlessly with existing components:

```python
from app.services.risk_scoring import RiskScorer
from app.services.decision_engine import DecisionEngine

# Setup (with chain rules added to rules_engine)
risk_scorer = RiskScorer(rules_engine)
decision_engine = DecisionEngine(risk_scorer)

# Evaluate transaction
result = decision_engine.evaluate(transaction, context)

# Chain triggers affect risk score and decision
print(f"Risk Score: {result['risk_assessment']['risk_score']:.2f}")
print(f"Decision: {result['decision']}")
```

## 🚨 Common Patterns

### Pattern 1: Simple Chain Detection

```python
# Most basic usage
context = context_provider.get_transaction_context(transaction)

if context.get("chain_analysis", {}).get("has_suspicious_chains"):
    print("⚠️ Suspicious chain detected!")
```

### Pattern 2: Detailed Analysis

```python
chain_analysis = context.get("chain_analysis", {})

for chain in chain_analysis.get("chains", []):
    if chain['suspicion_score'] > 0.7:
        print(f"High suspicion chain: {chain['pattern_type']}")
        print(f"  Transactions: {chain['transaction_ids']}")
        print(f"  Suspicion: {chain['suspicion_score']:.2f}")
```

### Pattern 3: Conditional Enabling

```python
# Only enable for high-value or suspicious accounts
enable_chain = (
    transaction['amount'] > 1000 or
    account.risk_tier == 'high'
)

context_provider = ContextProvider(
    db,
    enable_chain_analysis=enable_chain
)
```

## ⚡ Performance Tips

1. **High-volume accounts**: Reduce lookback period
   ```python
   analyzer.CHAIN_LOOKBACK_HOURS = 24
   ```

2. **Low-risk transactions**: Disable chain analysis
   ```python
   if transaction['amount'] < 10:
       enable_chain_analysis = False
   ```

3. **Database optimization**: Ensure indexes exist
   ```sql
   CREATE INDEX idx_tx_account_timestamp
   ON transactions(account_id, timestamp);

   CREATE INDEX idx_tx_counterparty
   ON transactions(counterparty_id);
   ```

## 📚 More Information

- **Full Documentation**: `docs/FRAUD_USE_CASE_CHAIN_DETECTION.md`
- **API Reference**: See docstrings in source files
- **Test Examples**: `tests/test_chain_detection.py`

## 🆘 Troubleshooting

**Issue**: False positives on legitimate businesses
```python
# Solution: Use permissive rules or increase thresholds
rules = get_permissive_chain_rules()
```

**Issue**: Missing fraud cases
```python
# Solution: Use high-security rules or decrease thresholds
rules = get_high_security_chain_rules()
```

**Issue**: Performance problems
```python
# Solution: Reduce lookback or disable for low-value transactions
analyzer.CHAIN_LOOKBACK_HOURS = 24
```

## ✅ Checklist

- [ ] Enable chain analysis in ContextProvider
- [ ] Add chain detection rules to RulesEngine
- [ ] Test with sample transactions
- [ ] Monitor false positive rate
- [ ] Tune thresholds based on results
- [ ] Set up database indexes for performance
- [ ] Document any custom configurations
