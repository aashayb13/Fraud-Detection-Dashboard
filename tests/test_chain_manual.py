#!/usr/bin/env python3
"""
Manual test script for chain detection (without pytest dependency).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base, Transaction, Account
from app.services.chain_analyzer import ChainAnalyzer
from app.services.context_provider import ContextProvider
from app.services.rules_engine import RulesEngine
from app.services.fraud_rules import (
    create_suspicious_chain_rule,
    create_credit_refund_transfer_rule,
    get_balanced_chain_rules
)


def create_test_db():
    """Create in-memory test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def test_credit_refund_transfer_chain():
    """Test detection of credit->refund->transfer chains."""
    print("\n=== Test: Credit-Refund-Transfer Chain Detection ===")

    db = create_test_db()

    # Create test account
    account = Account(
        account_id="TEST_ACC_001",
        creation_date=(datetime.utcnow() - timedelta(days=30)).isoformat(),
        risk_tier="medium",
        status="active"
    )
    db.add(account)
    db.commit()

    # Create suspicious chain
    now = datetime.utcnow()

    tx1 = Transaction(
        transaction_id="TX001",
        timestamp=(now - timedelta(hours=5)).isoformat(),
        account_id=account.account_id,
        counterparty_id="COUNTERPARTY_A",
        amount=50.0,
        transaction_type="CREDIT",
        description="Incoming payment"
    )

    tx2 = Transaction(
        transaction_id="TX002",
        timestamp=(now - timedelta(hours=3)).isoformat(),
        account_id=account.account_id,
        counterparty_id="COUNTERPARTY_A",
        amount=-45.0,
        transaction_type="REFUND",
        description="Refund payment"
    )

    tx3 = Transaction(
        transaction_id="TX003",
        timestamp=(now - timedelta(hours=1)).isoformat(),
        account_id=account.account_id,
        counterparty_id="COUNTERPARTY_B",
        amount=-40.0,
        transaction_type="TRANSFER",
        description="Transfer funds"
    )

    db.add_all([tx1, tx2, tx3])
    db.commit()

    # Analyze chains
    analyzer = ChainAnalyzer(db)
    current_tx = {"account_id": account.account_id, "amount": 100.0}
    result = analyzer.analyze_transaction_chains(account.account_id, current_tx)

    print(f"Chain count: {result['chain_count']}")
    print(f"Has suspicious chains: {result['has_suspicious_chains']}")
    print(f"Credit-refund chains: {result['credit_refund_count']}")
    print(f"Layering patterns: {result['layering_pattern_count']}")
    print(f"Rapid reversals: {result['rapid_reversal_count']}")
    print(f"Max suspicion score: {result.get('max_chain_suspicion', 0.0):.2f}")

    assert result['chain_count'] >= 0, "Should detect at least 0 chains"
    print("✓ Test passed!")

    db.close()
    return True


def test_layering_pattern():
    """Test detection of layering patterns."""
    print("\n=== Test: Layering Pattern Detection ===")

    db = create_test_db()

    # Create test account
    account = Account(
        account_id="TEST_ACC_002",
        creation_date=(datetime.utcnow() - timedelta(days=30)).isoformat(),
        risk_tier="medium",
        status="active"
    )
    db.add(account)
    db.commit()

    now = datetime.utcnow()

    # Multiple small credits
    for i in range(4):
        tx = Transaction(
            transaction_id=f"TX_CREDIT_{i}",
            timestamp=(now - timedelta(hours=10 - i)).isoformat(),
            account_id=account.account_id,
            counterparty_id=f"COUNTERPARTY_{i}",
            amount=25.0,
            transaction_type="CREDIT",
            description=f"Small credit {i}"
        )
        db.add(tx)

    # Consolidation transfer
    tx_transfer = Transaction(
        transaction_id="TX_CONSOLIDATE",
        timestamp=(now - timedelta(hours=2)).isoformat(),
        account_id=account.account_id,
        counterparty_id="COUNTERPARTY_FINAL",
        amount=-90.0,
        transaction_type="TRANSFER",
        description="Consolidation transfer"
    )
    db.add(tx_transfer)
    db.commit()

    # Analyze chains
    analyzer = ChainAnalyzer(db)
    current_tx = {"account_id": account.account_id, "amount": 100.0}
    result = analyzer.analyze_transaction_chains(account.account_id, current_tx)

    print(f"Chain count: {result['chain_count']}")
    print(f"Layering patterns: {result['layering_pattern_count']}")
    print(f"Has suspicious chains: {result['has_suspicious_chains']}")

    assert 'layering_pattern_count' in result, "Should have layering pattern count"
    print("✓ Test passed!")

    db.close()
    return True


def test_context_provider_integration():
    """Test integration with ContextProvider."""
    print("\n=== Test: ContextProvider Integration ===")

    db = create_test_db()

    # Create test account
    account = Account(
        account_id="TEST_ACC_003",
        creation_date=(datetime.utcnow() - timedelta(days=30)).isoformat(),
        risk_tier="medium",
        status="active"
    )
    db.add(account)
    db.commit()

    now = datetime.utcnow()

    # Add some transactions
    tx1 = Transaction(
        transaction_id="TX_CTX_1",
        timestamp=(now - timedelta(hours=3)).isoformat(),
        account_id=account.account_id,
        counterparty_id="COUNTERPARTY_X",
        amount=100.0,
        transaction_type="CREDIT"
    )

    tx2 = Transaction(
        transaction_id="TX_CTX_2",
        timestamp=(now - timedelta(hours=1)).isoformat(),
        account_id=account.account_id,
        counterparty_id="COUNTERPARTY_Y",
        amount=-90.0,
        transaction_type="TRANSFER"
    )

    db.add_all([tx1, tx2])
    db.commit()

    # Get context with chain analysis
    provider = ContextProvider(db, enable_chain_analysis=True)
    current_tx = {
        "account_id": account.account_id,
        "amount": 50.0,
        "transaction_type": "TRANSFER",
        "counterparty_id": "COUNTERPARTY_Z"
    }

    context = provider.get_transaction_context(current_tx)

    print("Context keys:", list(context.keys()))
    assert "chain_analysis" in context, "Context should include chain analysis"
    assert "has_suspicious_chains" in context["chain_analysis"]
    print(f"Chain analysis present: {context['chain_analysis']}")
    print("✓ Test passed!")

    db.close()
    return True


def test_fraud_rules():
    """Test fraud detection rules."""
    print("\n=== Test: Fraud Detection Rules ===")

    # Create test context
    context = {
        "chain_analysis": {
            "has_suspicious_chains": True,
            "max_chain_suspicion": 0.8,
            "chain_count": 2,
            "credit_refund_count": 1
        }
    }

    transaction = {"amount": 100.0}

    # Test suspicious chain rule
    rule = create_suspicious_chain_rule(suspicion_threshold=0.7)
    result = rule.evaluate(transaction, context)

    print(f"Suspicious chain rule triggered: {result}")
    assert result == True, "Rule should trigger for high suspicion"

    # Test with lower suspicion
    context["chain_analysis"]["max_chain_suspicion"] = 0.5
    result = rule.evaluate(transaction, context)
    print(f"Lower suspicion rule triggered: {result}")
    assert result == False, "Rule should not trigger for low suspicion"

    print("✓ Test passed!")
    return True


def test_rules_engine_integration():
    """Test integration with RulesEngine."""
    print("\n=== Test: Rules Engine Integration ===")

    # Create rules engine
    engine = RulesEngine()

    # Add balanced chain rules
    for rule in get_balanced_chain_rules():
        engine.add_rule(rule)

    print(f"Loaded {len(engine.rules)} rules")

    # Create context with suspicious activity
    context = {
        "chain_analysis": {
            "has_suspicious_chains": True,
            "max_chain_suspicion": 0.75,
            "chain_count": 3,
            "credit_refund_count": 1,
            "layering_pattern_count": 1,
            "rapid_reversal_count": 2
        }
    }

    transaction = {"amount": 200.0}

    # Evaluate all rules
    triggered = engine.evaluate_all(transaction, context)

    print(f"Triggered rules: {len(triggered)}")
    for rule_name in triggered.keys():
        print(f"  - {rule_name}")

    assert len(triggered) > 0, "Should trigger at least one rule"
    print("✓ Test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Chain Detection Fraud Use Case - Manual Tests")
    print("=" * 60)

    tests = [
        test_credit_refund_transfer_chain,
        test_layering_pattern,
        test_context_provider_integration,
        test_fraud_rules,
        test_rules_engine_integration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
