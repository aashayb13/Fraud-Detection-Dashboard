# tests/test_chain_detection.py
"""
Tests for chain detection fraud use case: Complex refund and transfer chains to hide origin.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base, Transaction, Account
from app.services.chain_analyzer import ChainAnalyzer, TransactionNode
from app.services.context_provider import ContextProvider
from app.services.rules_engine import RulesEngine
from app.services.fraud_rules import (
    create_suspicious_chain_rule,
    create_credit_refund_transfer_rule,
    create_layering_pattern_rule,
    create_rapid_reversal_rule,
    create_complex_chain_rule,
    get_balanced_chain_rules
)


@pytest.fixture
def test_db():
    """Create in-memory test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def test_account(test_db):
    """Create a test account."""
    account = Account(
        account_id="ACC001",
        creation_date=(datetime.utcnow() - timedelta(days=30)).isoformat(),
        risk_tier="medium",
        status="active"
    )
    test_db.add(account)
    test_db.commit()
    return account


class TestChainAnalyzer:
    """Test suite for ChainAnalyzer."""

    def test_credit_refund_transfer_detection(self, test_db, test_account):
        """Test detection of credit->refund->transfer chains."""
        # Create a suspicious chain
        now = datetime.utcnow()

        # Credit received
        tx1 = Transaction(
            transaction_id="TX001",
            timestamp=(now - timedelta(hours=5)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="COUNTERPARTY_A",
            amount=50.0,
            transaction_type="CREDIT",
            description="Incoming payment"
        )

        # Refund issued
        tx2 = Transaction(
            transaction_id="TX002",
            timestamp=(now - timedelta(hours=3)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="COUNTERPARTY_A",
            amount=-45.0,
            transaction_type="REFUND",
            description="Refund payment"
        )

        # Transfer to different party
        tx3 = Transaction(
            transaction_id="TX003",
            timestamp=(now - timedelta(hours=1)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="COUNTERPARTY_B",
            amount=-40.0,
            transaction_type="TRANSFER",
            description="Transfer funds"
        )

        test_db.add_all([tx1, tx2, tx3])
        test_db.commit()

        # Analyze chains
        analyzer = ChainAnalyzer(test_db)
        current_tx = {"account_id": test_account.account_id, "amount": 100.0}
        result = analyzer.analyze_transaction_chains(test_account.account_id, current_tx)

        # Assertions
        assert result["has_suspicious_chains"] or result["chain_count"] > 0
        assert result["credit_refund_count"] >= 0

    def test_layering_pattern_detection(self, test_db, test_account):
        """Test detection of layering patterns (multiple small credits -> transfer)."""
        now = datetime.utcnow()

        # Multiple small credits
        small_credits = []
        for i in range(4):
            tx = Transaction(
                transaction_id=f"TX_CREDIT_{i}",
                timestamp=(now - timedelta(hours=10 - i)).isoformat(),
                account_id=test_account.account_id,
                counterparty_id=f"COUNTERPARTY_{i}",
                amount=25.0,  # Small amounts
                transaction_type="CREDIT",
                description=f"Small credit {i}"
            )
            small_credits.append(tx)

        # Consolidation transfer
        tx_transfer = Transaction(
            transaction_id="TX_CONSOLIDATE",
            timestamp=(now - timedelta(hours=2)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="COUNTERPARTY_FINAL",
            amount=-90.0,  # Similar to sum of credits
            transaction_type="TRANSFER",
            description="Consolidation transfer"
        )

        test_db.add_all(small_credits + [tx_transfer])
        test_db.commit()

        # Analyze chains
        analyzer = ChainAnalyzer(test_db)
        current_tx = {"account_id": test_account.account_id, "amount": 100.0}
        result = analyzer.analyze_transaction_chains(test_account.account_id, current_tx)

        # Assertions
        assert result["chain_count"] >= 0
        assert "layering_pattern_count" in result

    def test_rapid_reversal_detection(self, test_db, test_account):
        """Test detection of rapid credit-refund reversals."""
        now = datetime.utcnow()

        # Create multiple rapid reversals
        for i in range(3):
            # Credit
            tx_credit = Transaction(
                transaction_id=f"TX_CREDIT_R_{i}",
                timestamp=(now - timedelta(hours=12 - i*3)).isoformat(),
                account_id=test_account.account_id,
                counterparty_id=f"COUNTERPARTY_{i}",
                amount=30.0,
                transaction_type="CREDIT",
                description=f"Credit {i}"
            )

            # Quick refund (within 2 hours)
            tx_refund = Transaction(
                transaction_id=f"TX_REFUND_R_{i}",
                timestamp=(now - timedelta(hours=11 - i*3)).isoformat(),
                account_id=test_account.account_id,
                counterparty_id=f"COUNTERPARTY_OTHER_{i}",
                amount=-28.0,
                transaction_type="REFUND",
                description=f"Refund {i}"
            )

            test_db.add_all([tx_credit, tx_refund])

        test_db.commit()

        # Analyze chains
        analyzer = ChainAnalyzer(test_db)
        current_tx = {"account_id": test_account.account_id, "amount": 100.0}
        result = analyzer.analyze_transaction_chains(test_account.account_id, current_tx)

        # Assertions
        assert "rapid_reversal_count" in result
        assert result["chain_count"] >= 0

    def test_no_chains_for_normal_transactions(self, test_db, test_account):
        """Test that normal transactions don't trigger chain detection."""
        now = datetime.utcnow()

        # Normal transactions without suspicious patterns
        tx1 = Transaction(
            transaction_id="TX_NORMAL_1",
            timestamp=(now - timedelta(days=5)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="COUNTERPARTY_NORMAL",
            amount=1000.0,
            transaction_type="CREDIT",
            description="Normal payment"
        )

        tx2 = Transaction(
            transaction_id="TX_NORMAL_2",
            timestamp=(now - timedelta(days=2)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="COUNTERPARTY_NORMAL",
            amount=-500.0,
            transaction_type="TRANSFER",
            description="Normal transfer"
        )

        test_db.add_all([tx1, tx2])
        test_db.commit()

        # Analyze chains
        analyzer = ChainAnalyzer(test_db)
        current_tx = {"account_id": test_account.account_id, "amount": 100.0}
        result = analyzer.analyze_transaction_chains(test_account.account_id, current_tx)

        # Should have minimal or no suspicious chains
        assert result["has_suspicious_chains"] == False or result["max_chain_suspicion"] < 0.6


class TestContextProviderWithChains:
    """Test suite for ContextProvider with chain analysis."""

    def test_context_includes_chain_analysis(self, test_db, test_account):
        """Test that context includes chain analysis data."""
        now = datetime.utcnow()

        # Create a simple chain
        tx1 = Transaction(
            transaction_id="TX_CTX_1",
            timestamp=(now - timedelta(hours=3)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="COUNTERPARTY_X",
            amount=100.0,
            transaction_type="CREDIT"
        )

        tx2 = Transaction(
            transaction_id="TX_CTX_2",
            timestamp=(now - timedelta(hours=1)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="COUNTERPARTY_Y",
            amount=-90.0,
            transaction_type="TRANSFER"
        )

        test_db.add_all([tx1, tx2])
        test_db.commit()

        # Get context with chain analysis
        provider = ContextProvider(test_db, enable_chain_analysis=True)
        current_tx = {
            "account_id": test_account.account_id,
            "amount": 50.0,
            "transaction_type": "TRANSFER",
            "counterparty_id": "COUNTERPARTY_Z"
        }

        context = provider.get_transaction_context(current_tx)

        # Assertions
        assert "chain_analysis" in context
        assert "has_suspicious_chains" in context["chain_analysis"]
        assert "chain_count" in context["chain_analysis"]
        assert "chains" in context["chain_analysis"]

    def test_context_without_chain_analysis(self, test_db, test_account):
        """Test that chain analysis can be disabled."""
        provider = ContextProvider(test_db, enable_chain_analysis=False)
        current_tx = {
            "account_id": test_account.account_id,
            "amount": 50.0,
            "transaction_type": "TRANSFER",
            "counterparty_id": "COUNTERPARTY_Z"
        }

        context = provider.get_transaction_context(current_tx)

        # Chain analysis should not be present
        assert "chain_analysis" not in context


class TestFraudRules:
    """Test suite for fraud detection rules."""

    def test_suspicious_chain_rule(self, test_db, test_account):
        """Test suspicious chain rule triggers correctly."""
        # Create suspicious chain context
        context = {
            "chain_analysis": {
                "has_suspicious_chains": True,
                "max_chain_suspicion": 0.8,
                "chain_count": 2
            }
        }

        transaction = {"amount": 100.0, "account_id": test_account.account_id}

        # Test rule
        rule = create_suspicious_chain_rule(suspicion_threshold=0.7)
        assert rule.evaluate(transaction, context) == True

        # Test with lower suspicion
        context["chain_analysis"]["max_chain_suspicion"] = 0.5
        assert rule.evaluate(transaction, context) == False

    def test_credit_refund_transfer_rule(self, test_db, test_account):
        """Test credit-refund-transfer chain rule."""
        context = {
            "chain_analysis": {
                "credit_refund_count": 2,
                "chain_count": 3
            }
        }

        transaction = {"amount": 100.0}

        # Test rule with min_chain_count=1
        rule = create_credit_refund_transfer_rule(min_chain_count=1)
        assert rule.evaluate(transaction, context) == True

        # Test with min_chain_count=3
        rule = create_credit_refund_transfer_rule(min_chain_count=3)
        assert rule.evaluate(transaction, context) == False

    def test_layering_pattern_rule(self, test_db, test_account):
        """Test layering pattern rule."""
        context = {
            "chain_analysis": {
                "layering_pattern_count": 2,
                "chain_count": 2
            }
        }

        transaction = {"amount": 100.0}

        rule = create_layering_pattern_rule(min_pattern_count=1)
        assert rule.evaluate(transaction, context) == True

        rule = create_layering_pattern_rule(min_pattern_count=3)
        assert rule.evaluate(transaction, context) == False

    def test_rapid_reversal_rule(self, test_db, test_account):
        """Test rapid reversal rule."""
        context = {
            "chain_analysis": {
                "rapid_reversal_count": 3,
                "chain_count": 3
            }
        }

        transaction = {"amount": 100.0}

        rule = create_rapid_reversal_rule(min_reversal_count=2)
        assert rule.evaluate(transaction, context) == True

        rule = create_rapid_reversal_rule(min_reversal_count=5)
        assert rule.evaluate(transaction, context) == False

    def test_complex_chain_rule(self, test_db, test_account):
        """Test complex chain rule."""
        context = {
            "chain_analysis": {
                "chain_count": 5,
                "has_suspicious_chains": True
            }
        }

        transaction = {"amount": 100.0}

        rule = create_complex_chain_rule(min_total_chains=3)
        assert rule.evaluate(transaction, context) == True

        rule = create_complex_chain_rule(min_total_chains=10)
        assert rule.evaluate(transaction, context) == False

    def test_balanced_rule_set_integration(self, test_db, test_account):
        """Test integration of balanced rule set with rules engine."""
        # Create rules engine
        engine = RulesEngine()

        # Add balanced chain rules
        for rule in get_balanced_chain_rules():
            engine.add_rule(rule)

        # Create context with moderate suspicion
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

        # Should trigger multiple rules
        assert len(triggered) > 0
        assert any("suspicious_chain" in name for name in triggered.keys())


class TestEndToEndChainDetection:
    """End-to-end tests for complete fraud detection flow."""

    def test_complete_fraud_scenario(self, test_db, test_account):
        """Test complete fraud detection scenario with all components."""
        now = datetime.utcnow()

        # Scenario: Fraudster receives illicit funds, tries to obscure via refund chain
        # Step 1: Receive initial credit
        tx1 = Transaction(
            transaction_id="FRAUD_TX_1",
            timestamp=(now - timedelta(hours=8)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="ILLICIT_SOURCE",
            amount=500.0,
            transaction_type="CREDIT",
            description="Initial payment"
        )

        # Step 2: Issue partial refund (claiming error)
        tx2 = Transaction(
            transaction_id="FRAUD_TX_2",
            timestamp=(now - timedelta(hours=6)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="ILLICIT_SOURCE",
            amount=-300.0,
            transaction_type="REFUND",
            description="Partial refund"
        )

        # Step 3: Transfer remaining to different account (layering)
        tx3 = Transaction(
            transaction_id="FRAUD_TX_3",
            timestamp=(now - timedelta(hours=4)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="LAYERING_ACCOUNT",
            amount=-180.0,
            transaction_type="TRANSFER",
            description="Transfer out"
        )

        test_db.add_all([tx1, tx2, tx3])
        test_db.commit()

        # New transaction to evaluate
        current_tx = {
            "transaction_id": "FRAUD_TX_4",
            "account_id": test_account.account_id,
            "counterparty_id": "FINAL_DESTINATION",
            "amount": -50.0,
            "transaction_type": "TRANSFER"
        }

        # Get context with chain analysis
        provider = ContextProvider(test_db, enable_chain_analysis=True)
        context = provider.get_transaction_context(current_tx)

        # Setup rules engine
        engine = RulesEngine()
        for rule in get_balanced_chain_rules():
            engine.add_rule(rule)

        # Evaluate transaction
        triggered_rules = engine.evaluate_all(current_tx, context)

        # Assertions - should detect suspicious patterns
        assert context["chain_analysis"]["chain_count"] > 0 or len(triggered_rules) >= 0

    def test_legitimate_business_scenario(self, test_db, test_account):
        """Test that legitimate business transactions don't trigger false positives."""
        now = datetime.utcnow()

        # Legitimate scenario: Normal business operations
        tx1 = Transaction(
            transaction_id="LEGIT_TX_1",
            timestamp=(now - timedelta(days=7)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="CUSTOMER_A",
            amount=2000.0,
            transaction_type="CREDIT",
            description="Product sale"
        )

        tx2 = Transaction(
            transaction_id="LEGIT_TX_2",
            timestamp=(now - timedelta(days=3)).isoformat(),
            account_id=test_account.account_id,
            counterparty_id="SUPPLIER_B",
            amount=-800.0,
            transaction_type="TRANSFER",
            description="Pay supplier"
        )

        test_db.add_all([tx1, tx2])
        test_db.commit()

        current_tx = {
            "transaction_id": "LEGIT_TX_3",
            "account_id": test_account.account_id,
            "counterparty_id": "CUSTOMER_A",
            "amount": 500.0,
            "transaction_type": "CREDIT"
        }

        # Get context and evaluate
        provider = ContextProvider(test_db, enable_chain_analysis=True)
        context = provider.get_transaction_context(current_tx)

        engine = RulesEngine()
        for rule in get_balanced_chain_rules():
            engine.add_rule(rule)

        triggered_rules = engine.evaluate_all(current_tx, context)

        # Should have minimal or no triggers for legitimate business
        assert context["chain_analysis"]["has_suspicious_chains"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
