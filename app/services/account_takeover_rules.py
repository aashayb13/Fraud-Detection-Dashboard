# app/services/account_takeover_rules.py
"""
Account takeover fraud detection rules.

This module implements detection rules for the following scenario:
An attacker gains control of an account, changes the phone number or device
to prevent security alerts, then performs unauthorized outgoing transfers.

Common attack pattern:
1. Attacker gains account access (phishing, credential stuffing, etc.)
2. Changes phone number/SIM to intercept 2FA codes and prevent alerts
3. Performs outgoing transfers before legitimate owner notices
"""
from typing import Dict, Any, List
from app.services.rules_engine import Rule


def create_phone_change_before_transfer_rule(
    max_hours: int = 48,
    rule_name: str = None,
    weight: float = 3.5
) -> Rule:
    """
    Detect outgoing transfers shortly after phone/device changes.

    This is a high-risk indicator: attackers often change phone numbers
    before initiating fraudulent transfers to prevent the legitimate user
    from receiving security alerts.

    Args:
        max_hours: Maximum hours between phone change and transfer to trigger
        rule_name: Optional custom rule name
        weight: Rule importance weight (high due to strong fraud indicator)

    Returns:
        Rule object
    """
    name = rule_name or f"phone_change_before_transfer_{max_hours}h"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Only check outgoing transfers
        if not ctx.get("is_outgoing_transfer", False):
            return False

        # Check if there was a recent phone change
        phone_changes = ctx.get(f"phone_changes_count_{max_hours}h", 0)
        if phone_changes == 0:
            return False

        # Check time since phone change
        hours_since_change = ctx.get("hours_since_phone_change")
        if hours_since_change is not None and hours_since_change <= max_hours:
            return True

        return False

    return Rule(
        name=name,
        description=f"Outgoing transfer within {max_hours} hours of phone/device change - possible account takeover",
        condition_func=condition,
        weight=weight
    )


def create_unverified_phone_change_transfer_rule(
    max_hours: int = 48,
    rule_name: str = None,
    weight: float = 4.5
) -> Rule:
    """
    Detect outgoing transfers after unverified phone changes.

    Extremely high risk: unverified phone changes followed by transfers
    are a strong indicator of account takeover fraud.

    Args:
        max_hours: Maximum hours to look back for phone changes
        rule_name: Optional custom rule name
        weight: Rule importance weight (very high - critical fraud indicator)

    Returns:
        Rule object
    """
    name = rule_name or f"unverified_phone_change_transfer_{max_hours}h"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Only check outgoing transfers
        if not ctx.get("is_outgoing_transfer", False):
            return False

        # Check for unverified phone changes
        unverified_count = ctx.get(f"unverified_phone_changes_{max_hours}h", 0)
        return unverified_count > 0

    return Rule(
        name=name,
        description=f"Outgoing transfer after unverified phone/device change - critical account takeover risk",
        condition_func=condition,
        weight=weight
    )


def create_suspicious_phone_change_transfer_rule(
    max_hours: int = 48,
    rule_name: str = None,
    weight: float = 4.0
) -> Rule:
    """
    Detect outgoing transfers after phone changes flagged as suspicious.

    Phone changes flagged by other systems (e.g., unusual location, device)
    followed by transfers warrant immediate review.

    Args:
        max_hours: Maximum hours to look back for phone changes
        rule_name: Optional custom rule name
        weight: Rule importance weight (very high - serious fraud indicator)

    Returns:
        Rule object
    """
    name = rule_name or f"suspicious_phone_change_transfer_{max_hours}h"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Only check outgoing transfers
        if not ctx.get("is_outgoing_transfer", False):
            return False

        # Check for suspicious phone changes
        suspicious_count = ctx.get(f"suspicious_phone_changes_{max_hours}h", 0)
        return suspicious_count > 0

    return Rule(
        name=name,
        description=f"Outgoing transfer after suspicious phone/device change - likely account takeover",
        condition_func=condition,
        weight=weight
    )


def create_first_transfer_after_phone_change_rule(
    rule_name: str = None,
    weight: float = 3.0
) -> Rule:
    """
    Detect the first outgoing transfer after a phone change.

    The first transfer after a phone change deserves extra scrutiny,
    especially if combined with other risk factors.

    Args:
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object
    """
    name = rule_name or "first_transfer_after_phone_change"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Only check outgoing transfers
        if not ctx.get("is_outgoing_transfer", False):
            return False

        # Check if this is the first transfer after phone change
        return ctx.get("is_first_transfer_after_phone_change", False)

    return Rule(
        name=name,
        description="First outgoing transfer after phone/device change - increased takeover risk",
        condition_func=condition,
        weight=weight
    )


def create_rapid_phone_change_rule(
    min_changes: int = 2,
    time_window_hours: int = 24,
    rule_name: str = None,
    weight: float = 3.5
) -> Rule:
    """
    Detect multiple phone/device changes in a short period.

    Multiple rapid phone changes can indicate:
    - Attacker trying different approaches
    - SIM swapping attempts
    - Testing of compromised account access

    Args:
        min_changes: Minimum number of changes to trigger
        time_window_hours: Time window in hours
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object
    """
    name = rule_name or f"rapid_phone_changes_{min_changes}_in_{time_window_hours}h"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Check for multiple phone changes
        changes_count = ctx.get(f"phone_changes_count_{time_window_hours}h", 0)
        return changes_count >= min_changes

    return Rule(
        name=name,
        description=f"Multiple phone/device changes ({min_changes}+) in {time_window_hours}h - possible SIM swap or account takeover",
        condition_func=condition,
        weight=weight
    )


def create_immediate_transfer_after_phone_change_rule(
    max_hours: int = 1,
    rule_name: str = None,
    weight: float = 5.0
) -> Rule:
    """
    Detect outgoing transfers immediately after phone change.

    Transfers within 1 hour of phone change are extremely suspicious
    and indicate automated fraud or urgent account takeover.

    Args:
        max_hours: Maximum hours between change and transfer (default 1)
        rule_name: Optional custom rule name
        weight: Rule importance weight (very high - critical indicator)

    Returns:
        Rule object
    """
    name = rule_name or f"immediate_transfer_after_phone_change_{max_hours}h"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Only check outgoing transfers
        if not ctx.get("is_outgoing_transfer", False):
            return False

        # Check if there was a phone change in the immediate window
        phone_changes = ctx.get(f"phone_changes_count_{max_hours}h", 0)
        if phone_changes == 0:
            return False

        # Verify the time is truly immediate
        hours_since_change = ctx.get("hours_since_phone_change")
        if hours_since_change is not None and hours_since_change <= max_hours:
            return True

        return False

    return Rule(
        name=name,
        description=f"Outgoing transfer within {max_hours} hour(s) of phone change - critical account takeover alert",
        condition_func=condition,
        weight=weight
    )


def create_large_transfer_after_phone_change_rule(
    min_amount: float = 5000.0,
    max_hours: int = 48,
    rule_name: str = None,
    weight: float = 4.0
) -> Rule:
    """
    Detect large outgoing transfers after phone changes.

    Large transfers combined with recent phone changes are high-risk.
    The combination of high value and phone change increases fraud likelihood.

    Args:
        min_amount: Minimum transfer amount to trigger
        max_hours: Maximum hours since phone change
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object
    """
    name = rule_name or f"large_transfer_after_phone_change_{int(min_amount)}"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Only check outgoing transfers
        if not ctx.get("is_outgoing_transfer", False):
            return False

        # Check transfer amount
        amount = tx.get("amount", 0)
        if amount < min_amount:
            return False

        # Check for recent phone change
        phone_changes = ctx.get(f"phone_changes_count_{max_hours}h", 0)
        return phone_changes > 0

    return Rule(
        name=name,
        description=f"Large transfer (>=${min_amount:,.2f}) within {max_hours}h of phone change - high-risk takeover",
        condition_func=condition,
        weight=weight
    )


def create_new_counterparty_after_phone_change_rule(
    max_hours: int = 48,
    rule_name: str = None,
    weight: float = 3.5
) -> Rule:
    """
    Detect transfers to new counterparties after phone changes.

    Transfers to new/unknown recipients combined with phone changes
    indicate possible account takeover with fund exfiltration.

    Args:
        max_hours: Maximum hours since phone change
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object
    """
    name = rule_name or f"new_counterparty_after_phone_change_{max_hours}h"

    def condition(tx: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Only check outgoing transfers
        if not ctx.get("is_outgoing_transfer", False):
            return False

        # Check if counterparty is new
        if not ctx.get("is_new_counterparty", False):
            return False

        # Check for recent phone change
        phone_changes = ctx.get(f"phone_changes_count_{max_hours}h", 0)
        return phone_changes > 0

    return Rule(
        name=name,
        description=f"Transfer to new counterparty within {max_hours}h of phone change - account takeover with fund diversion",
        condition_func=condition,
        weight=weight
    )


def initialize_account_takeover_rules() -> List[Rule]:
    """
    Initialize all account takeover fraud detection rules.

    Returns:
        List of configured Rule objects for account takeover detection
    """
    return [
        # Critical - immediate action indicators
        create_immediate_transfer_after_phone_change_rule(max_hours=1, weight=5.0),
        create_unverified_phone_change_transfer_rule(max_hours=48, weight=4.5),
        create_suspicious_phone_change_transfer_rule(max_hours=48, weight=4.0),
        create_large_transfer_after_phone_change_rule(min_amount=5000.0, max_hours=48, weight=4.0),

        # High priority - strong indicators
        create_phone_change_before_transfer_rule(max_hours=24, weight=3.5),
        create_phone_change_before_transfer_rule(max_hours=48, rule_name="phone_change_before_transfer_48h_extended", weight=3.0),
        create_rapid_phone_change_rule(min_changes=2, time_window_hours=24, weight=3.5),
        create_new_counterparty_after_phone_change_rule(max_hours=48, weight=3.5),

        # Medium priority - contextual indicators
        create_first_transfer_after_phone_change_rule(weight=3.0),
    ]


def get_high_security_takeover_rules() -> List[Rule]:
    """
    Get a conservative rule set for high-security environments.

    Triggers on lower thresholds to catch more potential account takeovers.
    """
    return [
        create_immediate_transfer_after_phone_change_rule(max_hours=1, weight=5.0),
        create_unverified_phone_change_transfer_rule(max_hours=72, weight=4.5),
        create_suspicious_phone_change_transfer_rule(max_hours=72, weight=4.0),
        create_large_transfer_after_phone_change_rule(min_amount=1000.0, max_hours=72, weight=4.0),
        create_phone_change_before_transfer_rule(max_hours=72, weight=3.5),
        create_rapid_phone_change_rule(min_changes=2, time_window_hours=48, weight=3.5),
        create_new_counterparty_after_phone_change_rule(max_hours=72, weight=3.5),
        create_first_transfer_after_phone_change_rule(weight=3.0),
    ]


def get_balanced_takeover_rules() -> List[Rule]:
    """
    Get a balanced rule set for standard account takeover detection.

    Balanced between catching fraud and minimizing false positives.
    """
    return initialize_account_takeover_rules()


def get_permissive_takeover_rules() -> List[Rule]:
    """
    Get a permissive rule set for low false-positive tolerance.

    Only triggers on high-confidence account takeover patterns.
    """
    return [
        create_immediate_transfer_after_phone_change_rule(max_hours=1, weight=5.0),
        create_unverified_phone_change_transfer_rule(max_hours=24, weight=4.5),
        create_suspicious_phone_change_transfer_rule(max_hours=24, weight=4.0),
        create_large_transfer_after_phone_change_rule(min_amount=10000.0, max_hours=24, weight=4.0),
        create_rapid_phone_change_rule(min_changes=3, time_window_hours=24, weight=3.5),
    ]
