# tests/test_money_mule_detection.py
import unittest
from unittest.mock import Mock, MagicMock
import datetime
from app.services.rules_engine import create_money_mule_rule
from app.services.context_provider import ContextProvider


class TestMoneyMuleDetection(unittest.TestCase):
    """Test cases for money mule fraud detection."""

    def test_money_mule_rule_triggers_on_pattern(self):
        """Test that money mule rule triggers when pattern is detected."""
        # Create rule with default thresholds
        rule = create_money_mule_rule(
            min_incoming_count=5,
            max_avg_incoming=500.0,
            min_flow_through_ratio=0.7,
            max_transfer_hours=48.0,
            time_window_hours=72
        )

        # Transaction data (doesn't matter for this test)
        transaction = {
            "transaction_id": "tx_001",
            "account_id": "acc_123",
            "amount": 450.0,
            "direction": "debit"
        }

        # Context showing money mule pattern
        context = {
            "incoming_count_72h": 8,  # Many incoming transactions
            "avg_incoming_amount_72h": 350.0,  # Small average amounts
            "flow_through_ratio_72h": 0.85,  # 85% flows through
            "avg_hours_to_transfer": 24.0  # Quick turnaround (24 hours)
        }

        # Should trigger
        result = rule.evaluate(transaction, context)
        self.assertTrue(result, "Money mule rule should trigger on suspicious pattern")

    def test_money_mule_rule_no_trigger_insufficient_incoming(self):
        """Test that rule doesn't trigger with insufficient incoming transactions."""
        rule = create_money_mule_rule(
            min_incoming_count=5,
            max_avg_incoming=500.0,
            min_flow_through_ratio=0.7,
            max_transfer_hours=48.0
        )

        transaction = {"transaction_id": "tx_001", "account_id": "acc_123", "amount": 450.0}

        # Only 3 incoming transactions (below threshold of 5)
        context = {
            "incoming_count_72h": 3,
            "avg_incoming_amount_72h": 350.0,
            "flow_through_ratio_72h": 0.85,
            "avg_hours_to_transfer": 24.0
        }

        result = rule.evaluate(transaction, context)
        self.assertFalse(result, "Should not trigger with insufficient incoming transactions")

    def test_money_mule_rule_no_trigger_large_amounts(self):
        """Test that rule doesn't trigger with large incoming amounts."""
        rule = create_money_mule_rule(
            min_incoming_count=5,
            max_avg_incoming=500.0,
            min_flow_through_ratio=0.7,
            max_transfer_hours=48.0
        )

        transaction = {"transaction_id": "tx_001", "account_id": "acc_123", "amount": 1500.0}

        # Large average incoming amount (above threshold)
        context = {
            "incoming_count_72h": 8,
            "avg_incoming_amount_72h": 1200.0,  # Too large
            "flow_through_ratio_72h": 0.85,
            "avg_hours_to_transfer": 24.0
        }

        result = rule.evaluate(transaction, context)
        self.assertFalse(result, "Should not trigger with large incoming amounts")

    def test_money_mule_rule_no_trigger_low_flow_through(self):
        """Test that rule doesn't trigger with low flow-through ratio."""
        rule = create_money_mule_rule(
            min_incoming_count=5,
            max_avg_incoming=500.0,
            min_flow_through_ratio=0.7,
            max_transfer_hours=48.0
        )

        transaction = {"transaction_id": "tx_001", "account_id": "acc_123", "amount": 450.0}

        # Low flow-through ratio (money stays in account)
        context = {
            "incoming_count_72h": 8,
            "avg_incoming_amount_72h": 350.0,
            "flow_through_ratio_72h": 0.3,  # Only 30% flows through
            "avg_hours_to_transfer": 24.0
        }

        result = rule.evaluate(transaction, context)
        self.assertFalse(result, "Should not trigger with low flow-through ratio")

    def test_money_mule_rule_no_trigger_slow_transfers(self):
        """Test that rule doesn't trigger when transfers are slow."""
        rule = create_money_mule_rule(
            min_incoming_count=5,
            max_avg_incoming=500.0,
            min_flow_through_ratio=0.7,
            max_transfer_hours=48.0
        )

        transaction = {"transaction_id": "tx_001", "account_id": "acc_123", "amount": 450.0}

        # Slow transfer speed
        context = {
            "incoming_count_72h": 8,
            "avg_incoming_amount_72h": 350.0,
            "flow_through_ratio_72h": 0.85,
            "avg_hours_to_transfer": 120.0  # 5 days (too slow)
        }

        result = rule.evaluate(transaction, context)
        self.assertFalse(result, "Should not trigger with slow transfer times")

    def test_money_mule_rule_handles_missing_transfer_time(self):
        """Test that rule handles missing transfer time data gracefully."""
        rule = create_money_mule_rule(
            min_incoming_count=5,
            max_avg_incoming=500.0,
            min_flow_through_ratio=0.7,
            max_transfer_hours=48.0
        )

        transaction = {"transaction_id": "tx_001", "account_id": "acc_123", "amount": 450.0}

        # No transfer time data available (new account or no outgoing yet)
        context = {
            "incoming_count_72h": 8,
            "avg_incoming_amount_72h": 350.0,
            "flow_through_ratio_72h": 0.85,
            "avg_hours_to_transfer": None  # No data
        }

        # Should still trigger if other conditions met (transfer time check passes if None)
        result = rule.evaluate(transaction, context)
        self.assertTrue(result, "Should trigger even without transfer time data if other conditions met")

    def test_money_mule_rule_custom_parameters(self):
        """Test money mule rule with custom parameters."""
        # More strict parameters
        rule = create_money_mule_rule(
            min_incoming_count=10,  # Require 10+ incoming
            max_avg_incoming=300.0,  # Lower threshold
            min_flow_through_ratio=0.9,  # 90% flow-through
            max_transfer_hours=24.0,  # Faster transfers
            time_window_hours=168  # Weekly window
        )

        transaction = {"transaction_id": "tx_001", "account_id": "acc_123", "amount": 250.0}

        # Meets strict criteria
        context = {
            "incoming_count_168h": 12,
            "avg_incoming_amount_168h": 250.0,
            "flow_through_ratio_168h": 0.95,
            "avg_hours_to_transfer": 12.0
        }

        result = rule.evaluate(transaction, context)
        self.assertTrue(result, "Should trigger with custom strict parameters")

    def test_money_mule_rule_metadata(self):
        """Test that money mule rule has proper metadata."""
        rule = create_money_mule_rule()

        self.assertEqual(rule.name, "money_mule_72h")
        self.assertIn("Money mule pattern", rule.description)
        self.assertEqual(rule.weight, 2.0)  # High weight for serious fraud

    def test_context_provider_adds_money_mule_context(self):
        """Test that ContextProvider adds money mule detection context."""
        # Mock database session and query results
        mock_db = Mock()

        # Mock account query
        mock_account = Mock()
        mock_account.creation_date = datetime.datetime(2024, 1, 1).isoformat()
        mock_account.risk_tier = "medium"

        # Mock transaction queries for money mule context
        # Incoming transactions (credits)
        mock_incoming_txs = []
        for i in range(7):
            tx = Mock()
            tx.amount = 400.0 + (i * 10)  # Small amounts
            tx.direction = "credit"
            tx.timestamp = (datetime.datetime.utcnow() - datetime.timedelta(hours=i*10)).isoformat()
            mock_incoming_txs.append(tx)

        # Outgoing transactions (debits)
        mock_outgoing_txs = []
        for i in range(5):
            tx = Mock()
            tx.amount = 500.0
            tx.direction = "debit"
            tx.timestamp = (datetime.datetime.utcnow() - datetime.timedelta(hours=i*12 + 5)).isoformat()
            mock_outgoing_txs.append(tx)

        # Setup query mock behavior
        def query_side_effect(*args):
            mock_query = Mock()
            mock_filter = Mock()

            # Return different results based on query type
            if args[0] == Mock:  # Account query (simplified)
                mock_query.filter.return_value.first.return_value = mock_account
            else:
                # Transaction queries
                mock_filter.all.return_value = mock_incoming_txs if "credit" in str(args) else mock_outgoing_txs
                mock_filter.order_by.return_value.all.return_value = mock_incoming_txs if "credit" in str(args) else mock_outgoing_txs
                mock_filter.count.return_value = len(mock_incoming_txs) + len(mock_outgoing_txs)
                mock_filter.first.return_value = None
                mock_query.filter.return_value = mock_filter

            return mock_query

        mock_db.query.side_effect = query_side_effect

        # Create context provider
        provider = ContextProvider(mock_db)

        # Get context for a transaction
        transaction = {
            "account_id": "acc_123",
            "amount": 450.0,
            "transaction_type": "ACH",
            "counterparty_id": "cp_456"
        }

        context = provider.get_transaction_context(transaction)

        # Verify money mule context keys exist
        self.assertIn("incoming_count_24h", context)
        self.assertIn("incoming_count_72h", context)
        self.assertIn("outgoing_count_24h", context)
        self.assertIn("flow_through_ratio_72h", context)
        self.assertIn("avg_incoming_amount_72h", context)
        self.assertIn("avg_hours_to_transfer", context)


if __name__ == '__main__':
    unittest.main()
