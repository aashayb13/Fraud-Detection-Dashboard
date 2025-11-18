# tests/test_odd_hours_detection.py
"""
Tests for odd hours transaction fraud detection.

Tests the detection of suspicious transactions occurring at unusual times,
including:
- Transactions during late night/early morning hours
- Large amounts at odd hours
- Timing that deviates from customer's normal patterns
- Weekend and holiday transactions
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base, Transaction, Account
from app.services.context_provider import ContextProvider
from app.services.odd_hours_rules import (
    create_odd_hours_transaction_rule,
    create_large_odd_hours_transaction_rule,
    create_very_large_odd_hours_transaction_rule,
    create_deviates_from_pattern_rule,
    create_weekend_odd_hours_transaction_rule,
    create_new_counterparty_odd_hours_rule,
    create_first_odd_hours_transaction_rule,
    create_outgoing_transfer_odd_hours_rule,
    create_international_transfer_odd_hours_rule,
    initialize_odd_hours_rules
)
from config.settings import (
    ODD_HOURS_START,
    ODD_HOURS_END,
    ODD_HOURS_LARGE_TRANSACTION_THRESHOLD,
    ODD_HOURS_VERY_LARGE_THRESHOLD
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def context_provider(db_session):
    """Create a ContextProvider instance for testing."""
    return ContextProvider(db_session, enable_chain_analysis=False)


@pytest.fixture
def test_account(db_session):
    """Create a test account."""
    account = Account(
        account_id="ACC_001",
        creation_date=(datetime.utcnow() - timedelta(days=365)).isoformat(),
        risk_tier="standard"
    )
    db_session.add(account)
    db_session.commit()
    return account


def test_odd_hours_transaction_detection(db_session, context_provider, test_account):
    """Test basic odd hours transaction detection."""
    # Create transaction at 2 AM (odd hours)
    transaction = {
        "transaction_id": "TX_001",
        "account_id": "ACC_001",
        "amount": 1000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": datetime.utcnow().replace(hour=2, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Check context has odd hours information
    assert "is_odd_hours" in context
    assert context["is_odd_hours"] is True
    assert context["transaction_hour"] == 2

    # Test rule
    rule = create_odd_hours_transaction_rule()
    assert rule.evaluate(transaction, context) is True


def test_business_hours_transaction_not_flagged(db_session, context_provider, test_account):
    """Test that business hours transactions are not flagged as odd hours."""
    # Create transaction at 2 PM (business hours)
    transaction = {
        "transaction_id": "TX_002",
        "account_id": "ACC_001",
        "amount": 1000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": datetime.utcnow().replace(hour=14, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Check context
    assert context["is_odd_hours"] is False
    assert context["transaction_hour"] == 14

    # Test rule should not trigger
    rule = create_odd_hours_transaction_rule()
    assert rule.evaluate(transaction, context) is False


def test_large_odd_hours_transaction(db_session, context_provider, test_account):
    """Test detection of large transactions during odd hours."""
    # Create large transaction at 3 AM
    transaction = {
        "transaction_id": "TX_003",
        "account_id": "ACC_001",
        "amount": 10000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": datetime.utcnow().replace(hour=3, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Test large odd hours rule
    rule = create_large_odd_hours_transaction_rule()
    assert rule.evaluate(transaction, context) is True


def test_very_large_odd_hours_transaction(db_session, context_provider, test_account):
    """Test detection of very large transactions during odd hours."""
    # Create very large transaction at 1 AM
    transaction = {
        "transaction_id": "TX_004",
        "account_id": "ACC_001",
        "amount": 50000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": datetime.utcnow().replace(hour=1, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Test very large odd hours rule
    rule = create_very_large_odd_hours_transaction_rule()
    assert rule.evaluate(transaction, context) is True


def test_pattern_deviation_detection(db_session, context_provider, test_account):
    """Test detection of transactions that deviate from customer's normal pattern."""
    # Create historical transactions during business hours (9 AM - 5 PM)
    base_time = datetime.utcnow() - timedelta(days=30)

    for i in range(20):
        tx = Transaction(
            transaction_id=f"TX_HIST_{i}",
            account_id="ACC_001",
            amount=1000.0,
            direction="debit",
            transaction_type="WIRE_TRANSFER",
            timestamp=(base_time + timedelta(days=i)).replace(hour=14, minute=0).isoformat()
        )
        db_session.add(tx)
    db_session.commit()

    # Now create an odd hours transaction
    transaction = {
        "transaction_id": "TX_005",
        "account_id": "ACC_001",
        "amount": 5000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": datetime.utcnow().replace(hour=2, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Check pattern deviation is detected
    assert context.get("deviates_from_pattern", False) is True
    assert context.get("historical_business_hours_ratio", 0) > 0.8

    # Test pattern deviation rule
    rule = create_deviates_from_pattern_rule()
    assert rule.evaluate(transaction, context) is True


def test_weekend_odd_hours_transaction(db_session, context_provider, test_account):
    """Test detection of transactions during weekend odd hours."""
    # Create transaction on Saturday at 11 PM
    now = datetime.utcnow()

    # Find next Saturday
    days_until_saturday = (5 - now.weekday()) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7
    saturday = now + timedelta(days=days_until_saturday)

    transaction = {
        "transaction_id": "TX_006",
        "account_id": "ACC_001",
        "amount": 10000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": saturday.replace(hour=23, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Check weekend and odd hours flags
    assert context.get("is_weekend", False) is True
    assert context.get("is_odd_hours", False) is True

    # Test weekend odd hours rule
    rule = create_weekend_odd_hours_transaction_rule()
    assert rule.evaluate(transaction, context) is True


def test_new_counterparty_odd_hours(db_session, context_provider, test_account):
    """Test detection of transactions to new counterparties during odd hours."""
    transaction = {
        "transaction_id": "TX_007",
        "account_id": "ACC_001",
        "amount": 5000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "counterparty_id": "COUNTER_NEW",
        "timestamp": datetime.utcnow().replace(hour=3, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Should detect both odd hours and new counterparty
    assert context.get("is_odd_hours", False) is True
    assert context.get("is_new_counterparty", False) is True

    # Test new counterparty odd hours rule
    rule = create_new_counterparty_odd_hours_rule()
    assert rule.evaluate(transaction, context) is True


def test_first_odd_hours_transaction(db_session, context_provider, test_account):
    """Test detection of first odd hours transaction in recent period."""
    # Create historical transactions during business hours only
    base_time = datetime.utcnow() - timedelta(days=5)

    for i in range(10):
        tx = Transaction(
            transaction_id=f"TX_BH_{i}",
            account_id="ACC_001",
            amount=1000.0,
            direction="debit",
            transaction_type="WIRE_TRANSFER",
            timestamp=(base_time + timedelta(days=i)).replace(hour=14, minute=0).isoformat()
        )
        db_session.add(tx)
    db_session.commit()

    # Now create first odd hours transaction
    transaction = {
        "transaction_id": "TX_008",
        "account_id": "ACC_001",
        "amount": 3000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": datetime.utcnow().replace(hour=2, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Check no recent odd hours transactions
    assert context.get("recent_odd_hours_transaction_count", 1) == 0
    assert context.get("is_odd_hours", False) is True

    # Test first odd hours rule
    rule = create_first_odd_hours_transaction_rule()
    assert rule.evaluate(transaction, context) is True


def test_outgoing_transfer_odd_hours(db_session, context_provider, test_account):
    """Test detection of outgoing transfers during odd hours."""
    transaction = {
        "transaction_id": "TX_009",
        "account_id": "ACC_001",
        "amount": 5000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": datetime.utcnow().replace(hour=1, minute=30, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Test outgoing transfer odd hours rule
    rule = create_outgoing_transfer_odd_hours_rule()
    assert rule.evaluate(transaction, context) is True


def test_international_transfer_odd_hours(db_session, context_provider, test_account):
    """Test detection of international transfers during odd hours."""
    transaction = {
        "transaction_id": "TX_010",
        "account_id": "ACC_001",
        "amount": 20000.0,
        "direction": "debit",
        "transaction_type": "INTERNATIONAL_WIRE",
        "counterparty_country": "CN",
        "timestamp": datetime.utcnow().replace(hour=4, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Test international transfer odd hours rule
    rule = create_international_transfer_odd_hours_rule()
    assert rule.evaluate(transaction, context) is True


def test_odd_hours_boundary_conditions(db_session, context_provider, test_account):
    """Test boundary conditions for odd hours detection."""
    # Test at exact start of odd hours (10 PM)
    transaction_start = {
        "transaction_id": "TX_011",
        "account_id": "ACC_001",
        "amount": 1000.0,
        "timestamp": datetime.utcnow().replace(hour=ODD_HOURS_START, minute=0, second=0).isoformat()
    }

    context_start = context_provider.get_transaction_context(transaction_start)
    assert context_start["is_odd_hours"] is True

    # Test just before end of odd hours (5:59 AM if end is 6 AM)
    transaction_end = {
        "transaction_id": "TX_012",
        "account_id": "ACC_001",
        "amount": 1000.0,
        "timestamp": datetime.utcnow().replace(hour=ODD_HOURS_END - 1, minute=59, second=0).isoformat()
    }

    context_end = context_provider.get_transaction_context(transaction_end)
    assert context_end["is_odd_hours"] is True

    # Test at exact end of odd hours (6 AM)
    transaction_after = {
        "transaction_id": "TX_013",
        "account_id": "ACC_001",
        "amount": 1000.0,
        "timestamp": datetime.utcnow().replace(hour=ODD_HOURS_END, minute=0, second=0).isoformat()
    }

    context_after = context_provider.get_transaction_context(transaction_after)
    assert context_after["is_odd_hours"] is False


def test_insufficient_history(db_session, context_provider, test_account):
    """Test handling of accounts with insufficient transaction history."""
    # Create transaction with no historical data
    transaction = {
        "transaction_id": "TX_014",
        "account_id": "ACC_001",
        "amount": 5000.0,
        "direction": "debit",
        "transaction_type": "WIRE_TRANSFER",
        "timestamp": datetime.utcnow().replace(hour=2, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Should flag insufficient history
    assert context.get("insufficient_history", False) is True

    # Pattern deviation rule should not trigger without sufficient history
    rule = create_deviates_from_pattern_rule()
    assert rule.evaluate(transaction, context) is False


def test_all_rules_initialization():
    """Test that all rules can be initialized without errors."""
    rules = initialize_odd_hours_rules()

    # Check we got a list of rules
    assert isinstance(rules, list)
    assert len(rules) > 0

    # Check each item is a Rule
    for rule in rules:
        assert hasattr(rule, "name")
        assert hasattr(rule, "description")
        assert hasattr(rule, "weight")
        assert hasattr(rule, "evaluate")


def test_hour_distribution_analysis(db_session, context_provider, test_account):
    """Test that hour distribution is correctly analyzed."""
    # Create transactions at various hours
    base_time = datetime.utcnow() - timedelta(days=30)

    # Create 10 transactions at 2 PM, 5 transactions at 3 AM
    for i in range(10):
        tx = Transaction(
            transaction_id=f"TX_2PM_{i}",
            account_id="ACC_001",
            amount=1000.0,
            direction="debit",
            transaction_type="WIRE_TRANSFER",
            timestamp=(base_time + timedelta(days=i)).replace(hour=14, minute=0).isoformat()
        )
        db_session.add(tx)

    for i in range(5):
        tx = Transaction(
            transaction_id=f"TX_3AM_{i}",
            account_id="ACC_001",
            amount=1000.0,
            direction="debit",
            transaction_type="WIRE_TRANSFER",
            timestamp=(base_time + timedelta(days=i)).replace(hour=3, minute=0).isoformat()
        )
        db_session.add(tx)
    db_session.commit()

    # Create new transaction at 2 PM
    transaction = {
        "transaction_id": "TX_015",
        "account_id": "ACC_001",
        "amount": 1000.0,
        "timestamp": datetime.utcnow().replace(hour=14, minute=0, second=0).isoformat()
    }

    context = context_provider.get_transaction_context(transaction)

    # Check hour distribution was calculated
    assert "hour_distribution" in context
    assert len(context["hour_distribution"]) == 24
    assert context["hour_distribution"][14] >= 10  # At least 10 at 2 PM
    assert context["hour_distribution"][3] >= 5  # At least 5 at 3 AM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
