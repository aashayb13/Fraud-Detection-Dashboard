#!/usr/bin/env python3
"""
Validation script for chain detection logic (no external dependencies).
Tests the core logic without database operations.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    """Test that all modules can be imported."""
    print("\n=== Testing Module Imports ===")

    try:
        from app.services import chain_analyzer
        print("✓ chain_analyzer module imported")
    except Exception as e:
        print(f"✗ Failed to import chain_analyzer: {e}")
        return False

    try:
        from app.services import fraud_rules
        print("✓ fraud_rules module imported")
    except Exception as e:
        print(f"✗ Failed to import fraud_rules: {e}")
        return False

    return True


def test_fraud_rule_factories():
    """Test fraud rule factory functions."""
    print("\n=== Testing Fraud Rule Factories ===")

    try:
        from app.services.fraud_rules import (
            create_suspicious_chain_rule,
            create_credit_refund_transfer_rule,
            create_layering_pattern_rule,
            create_rapid_reversal_rule,
            create_complex_chain_rule,
            get_balanced_chain_rules,
            get_high_security_chain_rules,
            get_permissive_chain_rules
        )

        # Test rule creation
        rule1 = create_suspicious_chain_rule()
        assert rule1.name == "suspicious_chain_70"
        assert rule1.weight == 2.0
        print(f"✓ Created suspicious_chain_rule: {rule1.name}")

        rule2 = create_credit_refund_transfer_rule()
        assert rule2.name == "credit_refund_transfer_chain_1"
        assert rule2.weight == 2.5
        print(f"✓ Created credit_refund_transfer_rule: {rule2.name}")

        rule3 = create_layering_pattern_rule()
        assert rule3.name == "layering_pattern_1"
        print(f"✓ Created layering_pattern_rule: {rule3.name}")

        rule4 = create_rapid_reversal_rule()
        assert rule4.name == "rapid_reversals_2"
        print(f"✓ Created rapid_reversal_rule: {rule4.name}")

        rule5 = create_complex_chain_rule()
        assert rule5.name == "complex_chains_3"
        print(f"✓ Created complex_chain_rule: {rule5.name}")

        # Test rule sets
        balanced_rules = get_balanced_chain_rules()
        assert len(balanced_rules) == 5
        print(f"✓ Balanced rule set has {len(balanced_rules)} rules")

        high_security_rules = get_high_security_chain_rules()
        assert len(high_security_rules) == 5
        print(f"✓ High security rule set has {len(high_security_rules)} rules")

        permissive_rules = get_permissive_chain_rules()
        assert len(permissive_rules) == 5
        print(f"✓ Permissive rule set has {len(permissive_rules)} rules")

        return True

    except Exception as e:
        print(f"✗ Failed fraud rule factory tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rule_evaluation():
    """Test rule evaluation logic."""
    print("\n=== Testing Rule Evaluation Logic ===")

    try:
        from app.services.fraud_rules import (
            create_suspicious_chain_rule,
            create_credit_refund_transfer_rule
        )

        # Test suspicious chain rule
        rule = create_suspicious_chain_rule(suspicion_threshold=0.7)

        # Context with high suspicion
        context_high = {
            "chain_analysis": {
                "has_suspicious_chains": True,
                "max_chain_suspicion": 0.85
            }
        }
        transaction = {"amount": 100.0}

        result = rule.evaluate(transaction, context_high)
        assert result == True, "Should trigger for high suspicion"
        print("✓ Suspicious chain rule correctly triggers for high suspicion")

        # Context with low suspicion
        context_low = {
            "chain_analysis": {
                "has_suspicious_chains": False,
                "max_chain_suspicion": 0.3
            }
        }

        result = rule.evaluate(transaction, context_low)
        assert result == False, "Should not trigger for low suspicion"
        print("✓ Suspicious chain rule correctly ignores low suspicion")

        # Test credit-refund-transfer rule
        rule2 = create_credit_refund_transfer_rule(min_chain_count=2)

        context_chains = {
            "chain_analysis": {
                "credit_refund_count": 3
            }
        }

        result = rule2.evaluate(transaction, context_chains)
        assert result == True, "Should trigger when chain count exceeds threshold"
        print("✓ Credit-refund-transfer rule correctly triggers")

        context_no_chains = {
            "chain_analysis": {
                "credit_refund_count": 1
            }
        }

        result = rule2.evaluate(transaction, context_no_chains)
        assert result == False, "Should not trigger when chain count below threshold"
        print("✓ Credit-refund-transfer rule correctly ignores low counts")

        return True

    except Exception as e:
        print(f"✗ Failed rule evaluation tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chain_analyzer_classes():
    """Test ChainAnalyzer dataclasses and structures."""
    print("\n=== Testing ChainAnalyzer Classes ===")

    try:
        from app.services.chain_analyzer import TransactionNode, TransactionChain
        from datetime import datetime

        # Test TransactionNode
        node = TransactionNode(
            transaction_id="TX001",
            timestamp=datetime.utcnow(),
            account_id="ACC001",
            counterparty_id="COUNTER001",
            amount=100.0,
            transaction_type="CREDIT",
            description="Test transaction"
        )

        assert node.transaction_id == "TX001"
        assert node.amount == 100.0
        print("✓ TransactionNode created successfully")

        # Test TransactionChain
        chain = TransactionChain(
            nodes=[node],
            chain_length=1,
            total_amount=100.0,
            time_span_hours=2.0,
            pattern_type="test_pattern",
            suspicion_score=0.75
        )

        assert chain.chain_length == 1
        assert chain.suspicion_score == 0.75
        print("✓ TransactionChain created successfully")

        # Test to_dict conversion
        chain_dict = chain.to_dict()
        assert "chain_length" in chain_dict
        assert "suspicion_score" in chain_dict
        assert chain_dict["pattern_type"] == "test_pattern"
        print("✓ TransactionChain.to_dict() works correctly")

        return True

    except Exception as e:
        print(f"✗ Failed ChainAnalyzer class tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_provider_modifications():
    """Test ContextProvider modifications (syntax check)."""
    print("\n=== Testing ContextProvider Modifications ===")

    try:
        from app.services.context_provider import ContextProvider

        # Check that ContextProvider has the new parameter
        import inspect
        sig = inspect.signature(ContextProvider.__init__)
        params = list(sig.parameters.keys())

        assert 'enable_chain_analysis' in params, "Should have enable_chain_analysis parameter"
        print("✓ ContextProvider.__init__ has enable_chain_analysis parameter")

        # Check that ChainAnalyzer import exists
        import app.services.context_provider as cp_module
        source = inspect.getsource(cp_module)

        assert "ChainAnalyzer" in source, "Should import ChainAnalyzer"
        print("✓ ContextProvider imports ChainAnalyzer")

        assert "chain_analysis" in source, "Should reference chain_analysis"
        print("✓ ContextProvider references chain_analysis")

        return True

    except Exception as e:
        print(f"✗ Failed ContextProvider tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Chain Detection Implementation - Validation Tests")
    print("=" * 70)

    tests = [
        test_imports,
        test_fraud_rule_factories,
        test_rule_evaluation,
        test_chain_analyzer_classes,
        test_context_provider_modifications
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Validation Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✓ All validation tests passed!")
        print("The chain detection implementation is syntactically correct.")
        print("Database integration tests require SQLAlchemy installation.")
    else:
        print("\n✗ Some validation tests failed.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
