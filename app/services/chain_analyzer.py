# app/services/chain_analyzer.py
"""
Chain Analyzer for detecting complex transaction chains used to obscure money origin.

This module detects patterns where small fake credits are followed by refunds and
transfers designed to hide the original source of funds (layering/money laundering).
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import Transaction
import json


@dataclass
class TransactionNode:
    """Represents a node in a transaction chain."""
    transaction_id: str
    timestamp: datetime
    account_id: str
    counterparty_id: str
    amount: float
    transaction_type: str
    description: Optional[str] = None

    @classmethod
    def from_db_transaction(cls, tx: Transaction) -> 'TransactionNode':
        """Create TransactionNode from database Transaction object."""
        return cls(
            transaction_id=tx.transaction_id,
            timestamp=datetime.fromisoformat(tx.timestamp),
            account_id=tx.account_id,
            counterparty_id=tx.counterparty_id,
            amount=tx.amount,
            transaction_type=tx.transaction_type,
            description=tx.description
        )


@dataclass
class TransactionChain:
    """Represents a chain of related transactions."""
    nodes: List[TransactionNode]
    chain_length: int
    total_amount: float
    time_span_hours: float
    pattern_type: str
    suspicion_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary for serialization."""
        return {
            "chain_length": self.chain_length,
            "total_amount": self.total_amount,
            "time_span_hours": self.time_span_hours,
            "pattern_type": self.pattern_type,
            "suspicion_score": self.suspicion_score,
            "transaction_ids": [node.transaction_id for node in self.nodes]
        }


class ChainAnalyzer:
    """
    Analyzes transaction chains to detect patterns that obscure money origin.

    Key patterns detected:
    1. Credit-Refund-Transfer chains
    2. Multiple small credits followed by consolidation transfers
    3. Rapid refund patterns after credits
    4. Circular transaction patterns
    """

    # Configuration constants
    CHAIN_LOOKBACK_HOURS = 72  # Look back 3 days for chain analysis
    SMALL_TRANSACTION_THRESHOLD = 100  # Amounts below this are "small"
    RAPID_TIMEFRAME_HOURS = 6  # Transactions within this are "rapid"
    MIN_CHAIN_LENGTH = 3  # Minimum chain length to be suspicious

    def __init__(self, db: Session):
        """
        Initialize chain analyzer with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def analyze_transaction_chains(self, account_id: str,
                                  current_tx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze transaction chains for an account and current transaction.

        Args:
            account_id: Account ID to analyze
            current_tx: Current transaction being evaluated

        Returns:
            Dictionary with chain analysis results
        """
        # Get recent transaction history
        recent_txs = self._get_recent_transactions(account_id)

        if not recent_txs:
            return {
                "has_suspicious_chains": False,
                "chain_count": 0,
                "chains": []
            }

        # Convert to TransactionNode objects
        nodes = [TransactionNode.from_db_transaction(tx) for tx in recent_txs]

        # Detect various chain patterns
        credit_refund_chains = self._detect_credit_refund_chains(nodes)
        layering_patterns = self._detect_layering_patterns(nodes)
        rapid_reversal_patterns = self._detect_rapid_reversals(nodes)

        # Combine all detected chains
        all_chains = credit_refund_chains + layering_patterns + rapid_reversal_patterns

        # Calculate aggregate metrics
        has_suspicious = any(chain.suspicion_score > 0.6 for chain in all_chains)
        max_suspicion = max([chain.suspicion_score for chain in all_chains], default=0.0)

        return {
            "has_suspicious_chains": has_suspicious,
            "chain_count": len(all_chains),
            "max_chain_suspicion": max_suspicion,
            "chains": [chain.to_dict() for chain in all_chains],
            "credit_refund_count": len(credit_refund_chains),
            "layering_pattern_count": len(layering_patterns),
            "rapid_reversal_count": len(rapid_reversal_patterns)
        }

    def _get_recent_transactions(self, account_id: str) -> List[Transaction]:
        """Get recent transactions for chain analysis."""
        lookback_time = (datetime.utcnow() -
                        timedelta(hours=self.CHAIN_LOOKBACK_HOURS)).isoformat()

        return self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > lookback_time
        ).order_by(Transaction.timestamp.desc()).all()

    def _detect_credit_refund_chains(self, nodes: List[TransactionNode]) -> List[TransactionChain]:
        """
        Detect credit->refund->transfer chains that obscure money origin.

        Pattern: Credit received -> Refund issued -> Transfer to different party
        This pattern is used to make illicit funds appear legitimate by claiming
        they were refunded, then moving them elsewhere.
        """
        chains = []

        for i in range(len(nodes) - 2):
            # Look for credit transactions
            if self._is_credit(nodes[i]):
                # Check if followed by refund within timeframe
                for j in range(i + 1, min(i + 10, len(nodes))):
                    if self._is_refund_or_debit(nodes[j]):
                        # Check if followed by transfer
                        for k in range(j + 1, min(j + 10, len(nodes))):
                            if self._is_transfer_or_debit(nodes[k]):
                                # Found a credit->refund->transfer chain
                                chain_nodes = [nodes[i], nodes[j], nodes[k]]
                                chain = self._build_chain(
                                    chain_nodes,
                                    "credit_refund_transfer"
                                )

                                if chain:
                                    chains.append(chain)

        return chains

    def _detect_layering_patterns(self, nodes: List[TransactionNode]) -> List[TransactionChain]:
        """
        Detect layering patterns: multiple small credits followed by transfers.

        Pattern: Multiple small incoming credits -> Larger outgoing transfer
        This obscures the source by breaking it into small pieces before consolidation.
        """
        chains = []

        # Find clusters of small credits
        small_credits = [n for n in nodes if self._is_credit(n) and
                        n.amount < self.SMALL_TRANSACTION_THRESHOLD]

        if len(small_credits) >= 2:
            # Look for transfers that might consolidate these credits
            for node in nodes:
                if self._is_transfer_or_debit(node):
                    # Check if this transfer follows the small credits
                    relevant_credits = [
                        c for c in small_credits
                        if c.timestamp < node.timestamp and
                        (node.timestamp - c.timestamp).total_seconds() / 3600 <
                        self.CHAIN_LOOKBACK_HOURS
                    ]

                    if len(relevant_credits) >= 2:
                        # Calculate if transfer amount is similar to sum of credits
                        total_credits = sum(c.amount for c in relevant_credits)

                        if 0.7 <= node.amount / max(total_credits, 0.01) <= 1.3:
                            # Found layering pattern
                            chain_nodes = relevant_credits + [node]
                            chain = self._build_chain(
                                chain_nodes,
                                "layering_consolidation"
                            )

                            if chain:
                                chains.append(chain)

        return chains

    def _detect_rapid_reversals(self, nodes: List[TransactionNode]) -> List[TransactionChain]:
        """
        Detect rapid credit-refund reversals that may indicate testing or layering.

        Pattern: Credit -> Quick refund to different party
        This can indicate testing of accounts or rapid movement to obscure origin.
        """
        chains = []

        for i in range(len(nodes) - 1):
            if self._is_credit(nodes[i]):
                # Look for rapid refund
                for j in range(i + 1, min(i + 5, len(nodes))):
                    if self._is_refund_or_debit(nodes[j]):
                        time_diff = (nodes[j].timestamp - nodes[i].timestamp)
                        hours_diff = time_diff.total_seconds() / 3600

                        if hours_diff <= self.RAPID_TIMEFRAME_HOURS:
                            # Check if counterparties differ (more suspicious)
                            chain_nodes = [nodes[i], nodes[j]]
                            chain = self._build_chain(
                                chain_nodes,
                                "rapid_reversal"
                            )

                            if chain:
                                chains.append(chain)

        return chains

    def _build_chain(self, nodes: List[TransactionNode],
                    pattern_type: str) -> Optional[TransactionChain]:
        """
        Build a TransactionChain object from nodes and calculate suspicion score.
        """
        if len(nodes) < 2:
            return None

        # Sort by timestamp
        sorted_nodes = sorted(nodes, key=lambda n: n.timestamp)

        # Calculate metrics
        chain_length = len(sorted_nodes)
        total_amount = sum(abs(n.amount) for n in sorted_nodes)
        time_span = (sorted_nodes[-1].timestamp - sorted_nodes[0].timestamp)
        time_span_hours = time_span.total_seconds() / 3600

        # Calculate suspicion score based on pattern characteristics
        suspicion_score = self._calculate_suspicion_score(
            sorted_nodes, pattern_type, time_span_hours
        )

        return TransactionChain(
            nodes=sorted_nodes,
            chain_length=chain_length,
            total_amount=total_amount,
            time_span_hours=time_span_hours,
            pattern_type=pattern_type,
            suspicion_score=suspicion_score
        )

    def _calculate_suspicion_score(self, nodes: List[TransactionNode],
                                  pattern_type: str,
                                  time_span_hours: float) -> float:
        """
        Calculate suspicion score for a chain (0.0 to 1.0).

        Higher scores indicate more suspicious patterns.
        """
        score = 0.0

        # Base score by pattern type
        pattern_base_scores = {
            "credit_refund_transfer": 0.7,
            "layering_consolidation": 0.8,
            "rapid_reversal": 0.6
        }
        score = pattern_base_scores.get(pattern_type, 0.5)

        # Adjust for chain length (longer chains are more suspicious)
        if len(nodes) >= 4:
            score += 0.1
        if len(nodes) >= 5:
            score += 0.1

        # Adjust for speed (faster chains are more suspicious)
        if time_span_hours < self.RAPID_TIMEFRAME_HOURS:
            score += 0.1
        if time_span_hours < 2:
            score += 0.1

        # Check for varying counterparties (more suspicious)
        unique_counterparties = len(set(n.counterparty_id for n in nodes))
        if unique_counterparties >= 3:
            score += 0.1

        # Check for small transaction amounts (testing behavior)
        small_tx_count = sum(1 for n in nodes
                           if n.amount < self.SMALL_TRANSACTION_THRESHOLD)
        if small_tx_count >= len(nodes) * 0.5:
            score += 0.05

        # Cap at 1.0
        return min(1.0, score)

    def _is_credit(self, node: TransactionNode) -> bool:
        """Check if transaction is an incoming credit."""
        return node.amount > 0 or node.transaction_type.upper() in ['CREDIT', 'DEPOSIT', 'INCOMING']

    def _is_refund_or_debit(self, node: TransactionNode) -> bool:
        """Check if transaction is a refund or debit."""
        tx_type = node.transaction_type.upper()
        return (tx_type in ['REFUND', 'DEBIT', 'OUTGOING', 'REVERSAL'] or
                (node.description and 'refund' in node.description.lower()))

    def _is_transfer_or_debit(self, node: TransactionNode) -> bool:
        """Check if transaction is a transfer or debit."""
        tx_type = node.transaction_type.upper()
        return tx_type in ['TRANSFER', 'WIRE', 'ACH', 'DEBIT', 'OUTGOING', 'PAYMENT']
