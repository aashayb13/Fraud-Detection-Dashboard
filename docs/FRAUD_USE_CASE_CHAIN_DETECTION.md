# Fraud Use Case: Complex Refund and Transfer Chains to Hide Origin

## Overview

This fraud detection use case identifies patterns where fraudsters use small fake credits followed by refunds and transfers designed to obscure where money came from. This is a form of transaction layering commonly used in money laundering operations.

## Fraud Pattern Description

### The Problem

Fraudsters attempt to hide the origin of illicit funds by creating complex transaction chains that make it difficult to trace the source. The typical pattern involves:

1. **Initial Credit**: Receiving illicit funds (often from fraud, theft, or other illegal activities)
2. **Refund Layer**: Issuing refunds, sometimes partial, to make the transactions appear legitimate ("oops, we made a mistake")
3. **Transfer Layer**: Moving funds to different accounts/parties to further obscure the trail

### Why It's Effective for Fraudsters

- Creates transaction "noise" that makes tracing difficult
- Refunds make transactions appear like legitimate business corrections
- Multiple counterparties obscure the relationship between source and destination
- Small amounts may fall below manual review thresholds
- Rapid sequences can exploit timing gaps in monitoring systems

## Technical Implementation

### Architecture

The implementation consists of three main components:

1. **ChainAnalyzer** (`app/services/chain_analyzer.py`)
   - Analyzes transaction history to identify suspicious chains
   - Detects three primary patterns:
     - Credit-Refund-Transfer chains
     - Layering patterns (multiple small credits → consolidation)
     - Rapid reversal patterns

2. **Enhanced ContextProvider** (`app/services/context_provider.py`)
   - Integrates chain analysis into transaction context
   - Can be enabled/disabled via `enable_chain_analysis` parameter

3. **Fraud Rules** (`app/services/fraud_rules.py`)
   - Five specialized rules for detecting chain-based fraud
   - Three pre-configured rule sets for different risk tolerances

### Detection Patterns

#### 1. Credit-Refund-Transfer Chains

**Pattern**: Credit → Refund → Transfer

**Example**:
```
Hour 0: +$500 credit from Party A (illicit funds)
Hour 2: -$300 refund to Party A (claiming error)
Hour 4: -$180 transfer to Party B (layering)
```

**Suspicion Factors**:
- Multiple counterparties involved
- Short time spans between transactions
- Mismatched amounts (not exact refunds)

#### 2. Layering Patterns

**Pattern**: Multiple Small Credits → Larger Transfer

**Example**:
```
Hour 0: +$25 from Party A
Hour 1: +$25 from Party B
Hour 2: +$25 from Party C
Hour 3: +$25 from Party D
Hour 5: -$90 transfer to Party E
```

**Suspicion Factors**:
- Multiple small incoming transactions (< $100)
- Consolidation into larger outgoing transfer
- Transfer amount roughly matches sum of credits (70-130%)

#### 3. Rapid Reversals

**Pattern**: Credit → Quick Refund (within 6 hours)

**Example**:
```
Hour 0: +$50 from Party A
Hour 1: -$45 refund to Party B (different party!)
```

**Suspicion Factors**:
- Very short time between credit and refund
- Different counterparties for credit vs refund
- Suggests testing or rapid movement

### Configuration

#### Chain Analyzer Settings

```python
# Default configuration in ChainAnalyzer class
CHAIN_LOOKBACK_HOURS = 72          # Look back 3 days
SMALL_TRANSACTION_THRESHOLD = 100   # Amounts below this are "small"
RAPID_TIMEFRAME_HOURS = 6          # Rapid = within 6 hours
MIN_CHAIN_LENGTH = 3               # Minimum chain length
```

#### Rule Sets

**High Security** (lowest false negative rate):
```python
from app.services.fraud_rules import get_high_security_chain_rules

rules = get_high_security_chain_rules()
# - Suspicion threshold: 0.6
# - Triggers on single instances of patterns
# - More false positives, catches more fraud
```

**Balanced** (recommended for most use cases):
```python
from app.services.fraud_rules import get_balanced_chain_rules

rules = get_balanced_chain_rules()
# - Suspicion threshold: 0.7
# - Balanced false positive/negative rates
# - Suitable for standard operations
```

**Permissive** (lowest false positive rate):
```python
from app.services.fraud_rules import get_permissive_chain_rules

rules = get_permissive_chain_rules()
# - Suspicion threshold: 0.8
# - Only triggers on high-confidence patterns
# - Fewer manual reviews, may miss some fraud
```

## Usage Examples

### Basic Setup

```python
from app.models.database import SessionLocal
from app.services.context_provider import ContextProvider
from app.services.rules_engine import RulesEngine
from app.services.fraud_rules import get_balanced_chain_rules

# Initialize database session
db = SessionLocal()

# Setup context provider with chain analysis
context_provider = ContextProvider(db, enable_chain_analysis=True)

# Setup rules engine with chain detection rules
rules_engine = RulesEngine()
for rule in get_balanced_chain_rules():
    rules_engine.add_rule(rule)

# Evaluate a transaction
transaction = {
    "transaction_id": "TX12345",
    "account_id": "ACC001",
    "amount": 50.0,
    "transaction_type": "TRANSFER",
    "counterparty_id": "COUNTER123"
}

context = context_provider.get_transaction_context(transaction)
triggered_rules = rules_engine.evaluate_all(transaction, context)

# Check results
if triggered_rules:
    print(f"⚠️ {len(triggered_rules)} rules triggered:")
    for rule_name, rule in triggered_rules.items():
        print(f"  - {rule.description}")
```

### Custom Rule Configuration

```python
from app.services.fraud_rules import (
    create_suspicious_chain_rule,
    create_credit_refund_transfer_rule
)

# Create custom rules with specific thresholds
rule1 = create_suspicious_chain_rule(
    suspicion_threshold=0.75,
    weight=3.0  # Higher weight for risk scoring
)

rule2 = create_credit_refund_transfer_rule(
    min_chain_count=2,  # Only trigger if 2+ chains detected
    weight=2.5
)

rules_engine.add_rule(rule1)
rules_engine.add_rule(rule2)
```

### Analyzing Chain Details

```python
# Get detailed chain analysis
context = context_provider.get_transaction_context(transaction)
chain_analysis = context.get("chain_analysis", {})

print(f"Total chains detected: {chain_analysis.get('chain_count', 0)}")
print(f"Max suspicion score: {chain_analysis.get('max_chain_suspicion', 0):.2f}")
print(f"Credit-refund chains: {chain_analysis.get('credit_refund_count', 0)}")
print(f"Layering patterns: {chain_analysis.get('layering_pattern_count', 0)}")
print(f"Rapid reversals: {chain_analysis.get('rapid_reversal_count', 0)}")

# Examine individual chains
for chain in chain_analysis.get('chains', []):
    print(f"\nChain: {chain['pattern_type']}")
    print(f"  Length: {chain['chain_length']} transactions")
    print(f"  Time span: {chain['time_span_hours']:.1f} hours")
    print(f"  Total amount: ${chain['total_amount']:.2f}")
    print(f"  Suspicion: {chain['suspicion_score']:.2f}")
    print(f"  Transactions: {', '.join(chain['transaction_ids'])}")
```

## Suspicion Scoring

The system calculates a suspicion score (0.0 to 1.0) for each detected chain based on:

| Factor | Score Adjustment | Reasoning |
|--------|------------------|-----------|
| Base pattern type | 0.6-0.8 | Credit-refund-transfer = 0.7, Layering = 0.8, Rapid = 0.6 |
| Chain length ≥ 4 | +0.1 | Longer chains more suspicious |
| Chain length ≥ 5 | +0.1 (additional) | Very long chains highly suspicious |
| Time span < 6 hours | +0.1 | Rapid execution suspicious |
| Time span < 2 hours | +0.1 (additional) | Very rapid highly suspicious |
| 3+ unique counterparties | +0.1 | More parties = more layering |
| 50%+ small transactions | +0.05 | Small amounts suggest testing |

**Example Calculation**:
- Base (layering pattern): 0.8
- Chain length 5: +0.2
- Time span 4 hours: +0.1
- 4 unique counterparties: +0.1
- **Total: 1.0 (capped at 1.0)**

## Integration with Risk Scoring

Chain detection integrates with the existing risk scoring system:

```python
from app.services.risk_scoring import RiskScorer
from app.services.decision_engine import DecisionEngine

# Chain rules have higher weights (1.5-2.5) vs standard rules (1.0)
# This ensures chain patterns significantly impact risk scores

risk_scorer = RiskScorer(rules_engine)
decision_engine = DecisionEngine(risk_scorer)

result = decision_engine.evaluate(transaction, context)

print(f"Risk score: {result['risk_assessment']['risk_score']:.2f}")
print(f"Decision: {result['decision']}")
if result['decision'] == 'manual_review':
    print(f"Reason: {result['review_reason']}")
```

## Testing

### Running Tests

Full integration tests (requires SQLAlchemy):
```bash
pytest tests/test_chain_detection.py -v
```

Logic validation (no external dependencies):
```bash
python tests/validate_chain_logic.py
```

Manual tests with database (requires SQLAlchemy):
```bash
python tests/test_chain_manual.py
```

### Test Coverage

The test suite includes:

1. **Unit Tests**: Individual pattern detection
   - Credit-refund-transfer chains
   - Layering patterns
   - Rapid reversals

2. **Integration Tests**: Component interaction
   - ChainAnalyzer with ContextProvider
   - Rules with RulesEngine
   - End-to-end transaction evaluation

3. **Scenario Tests**: Real-world cases
   - Complete fraud scenarios
   - Legitimate business transactions (false positive checks)

## Performance Considerations

### Database Queries

The ChainAnalyzer performs queries with:
- **Lookback period**: 72 hours (configurable)
- **Index requirements**: `account_id`, `timestamp`, `counterparty_id`
- **Query complexity**: O(n) where n = transactions in lookback period

### Optimization Tips

1. **Adjust lookback period** for high-volume accounts:
   ```python
   analyzer.CHAIN_LOOKBACK_HOURS = 24  # Reduce to 1 day
   ```

2. **Disable for low-risk transactions**:
   ```python
   if transaction['amount'] < 10:
       context_provider = ContextProvider(db, enable_chain_analysis=False)
   ```

3. **Cache analysis results** for batch processing:
   - Run chain analysis periodically
   - Store results in `metadata` field
   - Reference cached results instead of re-analyzing

## Monitoring and Tuning

### Key Metrics to Monitor

1. **False Positive Rate**:
   - Track manual review outcomes
   - Adjust rule thresholds if FP rate > 30%

2. **Detection Rate**:
   - Review confirmed fraud cases
   - Ensure chain patterns were detected

3. **Performance**:
   - Monitor query execution time
   - Alert if chain analysis takes > 500ms

### Tuning Recommendations

**If False Positive Rate is High**:
- Increase suspicion thresholds (0.7 → 0.8)
- Increase minimum chain counts (1 → 2)
- Switch to permissive rule set

**If Missing Fraud Cases**:
- Decrease suspicion thresholds (0.7 → 0.6)
- Decrease time windows to catch slower operations
- Switch to high-security rule set

**If Performance Issues**:
- Reduce lookback period (72h → 48h)
- Add database indexes on `timestamp`, `account_id`
- Implement caching for repeat analysis

## Limitations

1. **Legitimate Use Cases**: Some legitimate scenarios may trigger false positives:
   - Merchants with high refund rates
   - Payment processors handling multiple parties
   - Test environments with rapid transaction patterns

2. **Sophisticated Fraud**: May not detect:
   - Very slow layering (weeks/months)
   - Chains split across multiple accounts
   - External transfers outside the system

3. **Resource Usage**: Chain analysis adds:
   - Additional database queries per transaction
   - ~200-500ms processing time
   - Memory for storing chain data

## Future Enhancements

Potential improvements for future versions:

1. **Cross-Account Analysis**: Detect chains across multiple accounts
2. **Network Graph Analysis**: Build transaction network graphs
3. **ML-Based Pattern Recognition**: Use machine learning to identify new patterns
4. **Real-Time Alerts**: Immediate notifications for high-suspicion chains
5. **Historical Pattern Learning**: Adapt thresholds based on historical data

## References

- Transaction monitoring system documentation
- Rules engine documentation (`app/services/rules_engine.py`)
- Risk scoring documentation (`app/services/risk_scoring.py`)
- Decision engine documentation (`app/services/decision_engine.py`)

## Support

For questions or issues with chain detection:
1. Review the test cases in `tests/test_chain_detection.py`
2. Check configuration in `ChainAnalyzer` class
3. Examine triggered rules in dashboard
4. Review chain details in risk assessment metadata
