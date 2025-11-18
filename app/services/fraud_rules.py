# app/services/fraud_rules.py
"""
Fraud-specific rules for transaction monitoring.

This module contains rule factories for detecting various fraud patterns,
including complex refund/transfer chains designed to obscure money origin.
"""

from typing import Dict, Any
from app.services.rules_engine import Rule


def create_suspicious_chain_rule(
    suspicion_threshold: float = 0.7,
    rule_name: str = None,
    weight: float = 2.0
) -> Rule:
    """
    Create a rule that triggers on suspicious transaction chains.

    Detects complex refund and transfer chains used to hide money origin,
    including:
    - Credit -> Refund -> Transfer patterns
    - Multiple small credits consolidated into transfers (layering)
    - Rapid credit-refund reversals

    Args:
        suspicion_threshold: Minimum suspicion score to trigger (0.0-1.0)
        rule_name: Optional custom rule name
        weight: Rule importance weight (higher = more important)

    Returns:
        Rule object
    """
    name = rule_name or f"suspicious_chain_{int(suspicion_threshold*100)}"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        chain_analysis = ctx.get("chain_analysis")
        if not chain_analysis:
            return False

        return (chain_analysis.get("has_suspicious_chains", False) and
                chain_analysis.get("max_chain_suspicion", 0.0) >= suspicion_threshold)

    return Rule(
        name=name,
        description=f"Suspicious transaction chain detected (threshold: {suspicion_threshold})",
        condition_func=condition,
        weight=weight
    )


def create_credit_refund_transfer_rule(
    min_chain_count: int = 1,
    rule_name: str = None,
    weight: float = 2.5
) -> Rule:
    """
    Create a rule that triggers on credit-refund-transfer chains.

    This specific pattern is often used to obscure the origin of funds by:
    1. Receiving credit (potentially illicit)
    2. Issuing refund (claiming transaction was in error)
    3. Transferring to different party (layering the funds)

    Args:
        min_chain_count: Minimum number of such chains to trigger
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object
    """
    name = rule_name or f"credit_refund_transfer_chain_{min_chain_count}"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        chain_analysis = ctx.get("chain_analysis")
        if not chain_analysis:
            return False

        return chain_analysis.get("credit_refund_count", 0) >= min_chain_count

    return Rule(
        name=name,
        description=f"Credit-Refund-Transfer chain detected (min {min_chain_count} chains)",
        condition_func=condition,
        weight=weight
    )


def create_layering_pattern_rule(
    min_pattern_count: int = 1,
    rule_name: str = None,
    weight: float = 2.0
) -> Rule:
    """
    Create a rule that triggers on layering patterns.

    Detects when multiple small credits are followed by larger transfers,
    a classic money laundering technique to obscure the source of funds.

    Args:
        min_pattern_count: Minimum number of layering patterns to trigger
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object
    """
    name = rule_name or f"layering_pattern_{min_pattern_count}"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        chain_analysis = ctx.get("chain_analysis")
        if not chain_analysis:
            return False

        return chain_analysis.get("layering_pattern_count", 0) >= min_pattern_count

    return Rule(
        name=name,
        description=f"Layering pattern detected - multiple small credits consolidated (min {min_pattern_count} patterns)",
        condition_func=condition,
        weight=weight
    )


def create_rapid_reversal_rule(
    min_reversal_count: int = 2,
    rule_name: str = None,
    weight: float = 1.5
) -> Rule:
    """
    Create a rule that triggers on rapid credit-refund reversals.

    Rapid reversals can indicate:
    - Testing of compromised accounts
    - Attempts to confuse transaction monitoring
    - Quick movement of funds to obscure origin

    Args:
        min_reversal_count: Minimum number of rapid reversals to trigger
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object
    """
    name = rule_name or f"rapid_reversals_{min_reversal_count}"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        chain_analysis = ctx.get("chain_analysis")
        if not chain_analysis:
            return False

        return chain_analysis.get("rapid_reversal_count", 0) >= min_reversal_count

    return Rule(
        name=name,
        description=f"Rapid credit-refund reversals detected (min {min_reversal_count} reversals)",
        condition_func=condition,
        weight=weight
    )


def create_complex_chain_rule(
    min_total_chains: int = 3,
    rule_name: str = None,
    weight: float = 2.5
) -> Rule:
    """
    Create a rule that triggers when multiple chain patterns are present.

    When an account shows multiple types of suspicious chains, it indicates
    sophisticated attempts to obscure transaction origins.

    Args:
        min_total_chains: Minimum total number of chains to trigger
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object
    """
    name = rule_name or f"complex_chains_{min_total_chains}"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        chain_analysis = ctx.get("chain_analysis")
        if not chain_analysis:
            return False

        return chain_analysis.get("chain_count", 0) >= min_total_chains

    return Rule(
        name=name,
        description=f"Multiple suspicious transaction chains detected (min {min_total_chains} total)",
        condition_func=condition,
        weight=weight
    )


# Pre-configured rule sets for different risk tolerances

def get_high_security_chain_rules():
    """
    Get a conservative rule set for high-security environments.

    Triggers on lower thresholds to catch more potential fraud.
    """
    return [
        create_suspicious_chain_rule(suspicion_threshold=0.6, weight=2.0),
        create_credit_refund_transfer_rule(min_chain_count=1, weight=2.5),
        create_layering_pattern_rule(min_pattern_count=1, weight=2.0),
        create_rapid_reversal_rule(min_reversal_count=2, weight=1.5),
        create_complex_chain_rule(min_total_chains=2, weight=2.5)
    ]


def get_balanced_chain_rules():
    """
    Get a balanced rule set for standard fraud detection.

    Balanced between catching fraud and minimizing false positives.
    """
    return [
        create_suspicious_chain_rule(suspicion_threshold=0.7, weight=2.0),
        create_credit_refund_transfer_rule(min_chain_count=1, weight=2.5),
        create_layering_pattern_rule(min_pattern_count=1, weight=2.0),
        create_rapid_reversal_rule(min_reversal_count=3, weight=1.5),
        create_complex_chain_rule(min_total_chains=3, weight=2.5)
    ]


def get_permissive_chain_rules():
    """
    Get a permissive rule set for low false-positive tolerance.

    Only triggers on high-confidence fraud patterns.
    """
    return [
        create_suspicious_chain_rule(suspicion_threshold=0.8, weight=2.0),
        create_credit_refund_transfer_rule(min_chain_count=2, weight=2.5),
        create_layering_pattern_rule(min_pattern_count=2, weight=2.0),
        create_rapid_reversal_rule(min_reversal_count=4, weight=1.5),
        create_complex_chain_rule(min_total_chains=4, weight=2.5)
    ]
