"""
Fraud Detection Modules Catalog

Complete catalog of all 25+ fraud detection modules with descriptions,
categories, and what they detect.
"""

FRAUD_MODULES_CATALOG = {
    "transaction_history": {
        "name": "Transaction History Analysis",
        "category": "Behavioral Analysis",
        "description": "Analyzes historical transaction patterns to detect anomalies",
        "detects": [
            "Unusual transaction amounts compared to history",
            "Frequency anomalies",
            "Deviations from normal patterns"
        ],
        "examples": [
            "Unusually large transfer from a low-activity account",
            "Tiny test deposits used to validate accounts before fraud"
        ],
        "severity": "medium",
        "icon": "📊"
    },
    "new_counterparty": {
        "name": "New Counterparty Detection",
        "category": "Relationship Analysis",
        "description": "Flags transactions to previously unseen counterparties",
        "detects": [
            "First-time recipient payments",
            "New beneficiary relationships",
            "Unknown counterparty transactions"
        ],
        "severity": "low",
        "icon": "🆕"
    },
    "money_mule": {
        "name": "Money Mule Detection",
        "category": "Financial Crime",
        "description": "Identifies accounts being used for money laundering",
        "detects": [
            "Rapid in-and-out transaction patterns",
            "High-velocity money movement",
            "Layering behavior",
            "Structuring attempts"
        ],
        "examples": [
            "Accounts used as money mules (funds in, immediately out)",
            "Multiple small transactions followed by a large withdrawal",
            "Tiny test deposits followed by large fraudulent transfers"
        ],
        "severity": "high",
        "icon": "💰"
    },
    "beneficiary_fraud": {
        "name": "Beneficiary Fraud Detection",
        "category": "Payment Fraud",
        "description": "Detects fraudulent beneficiary additions and changes",
        "detects": [
            "Rapid beneficiary additions",
            "Compromised admin account patterns",
            "Bulk beneficiary uploads",
            "Suspicious payment to new beneficiaries"
        ],
        "examples": [
            "Payment to a newly added beneficiary immediately after a change request",
            "Rapid addition of many new beneficiaries followed by payments",
            "Social-engineering push payment (P2P/Zelle-style fraud)"
        ],
        "severity": "high",
        "icon": "👥"
    },
    "vendor_impersonation": {
        "name": "Vendor Impersonation (BEC)",
        "category": "Business Email Compromise",
        "description": "Detects Business Email Compromise and vendor impersonation",
        "detects": [
            "Banking detail changes before payment",
            "Same-day payment after detail change",
            "Unverified vendor changes",
            "Rapid account modifications"
        ],
        "examples": [
            "Social-engineering push payment (fraudster impersonates CEO/vendor)",
            "Payment after suspicious banking detail change"
        ],
        "severity": "critical",
        "icon": "🎭"
    },
    "chain_analysis": {
        "name": "Transaction Chain Analysis",
        "category": "Network Analysis",
        "description": "Analyzes transaction chains to detect fraud networks",
        "detects": [
            "Circular transaction patterns",
            "Multi-hop money movement",
            "Rapid sequential transactions",
            "Suspicious transaction chains"
        ],
        "examples": [
            "Complex refund and transfer chains to hide origin",
            "Multiple small transactions followed by a large withdrawal"
        ],
        "severity": "high",
        "icon": "🔗"
    },
    "account_takeover": {
        "name": "Account Takeover Detection",
        "category": "Identity Fraud",
        "description": "Detects compromised accounts through behavioral changes",
        "detects": [
            "Sudden account changes",
            "Device/location mismatches",
            "Unusual login patterns",
            "Behavioral anomalies"
        ],
        "examples": [
            "Account takeover accompanied by phone/SIM swap changes",
            "Payroll/direct deposit rerouting after account compromise"
        ],
        "severity": "critical",
        "icon": "🔐"
    },
    "odd_hours": {
        "name": "Odd Hours Transaction Detection",
        "category": "Temporal Analysis",
        "description": "Flags transactions during unusual hours (10 PM - 6 AM)",
        "detects": [
            "Late-night transactions",
            "Weekend activity",
            "Deviations from normal timing patterns",
            "Large transactions during odd hours"
        ],
        "examples": [
            "Large transactions initiated at odd hours (2 AM wire transfer)"
        ],
        "severity": "medium",
        "icon": "🌙"
    },
    "high_risk_times": {
        "name": "High-Risk Transaction Times",
        "category": "Temporal Analysis",
        "description": "Advanced timing analysis including holidays and end-of-month",
        "detects": [
            "Holiday transaction anomalies",
            "End-of-month fraud patterns",
            "Timing pattern deviations",
            "Non-business hours activity"
        ],
        "examples": [
            "Large transactions initiated at odd hours (holiday fraud)",
            "Weekend wire transfers when business is closed"
        ],
        "severity": "medium",
        "icon": "⏰"
    },
    "geographic_fraud": {
        "name": "Geographic Fraud Detection",
        "category": "Location Analysis",
        "description": "Detects unusual geographic transaction patterns",
        "detects": [
            "Payments to unusual countries",
            "Geographic inconsistencies",
            "Deviations from vendor locations",
            "Routing anomalies"
        ],
        "examples": [
            "Payments routed to unexpected or high-risk countries",
            "Wire transfer to jurisdiction inconsistent with business activity"
        ],
        "severity": "medium",
        "icon": "🌍"
    },
    "location_inconsistent": {
        "name": "Location-Inconsistent Transactions",
        "category": "Location Analysis",
        "description": "Detects impossible travel and location anomalies",
        "detects": [
            "Impossible travel patterns",
            "Simultaneous transactions from different locations",
            "Geographic velocity anomalies",
            "VPN/location spoofing"
        ],
        "severity": "high",
        "icon": "📍"
    },
    "blacklist": {
        "name": "Blacklist Detection",
        "category": "Watchlist Screening",
        "description": "Checks transactions against known fraudulent entities",
        "detects": [
            "Sanctioned entities",
            "Known fraudulent accounts",
            "Blacklisted merchants",
            "Flagged routing numbers"
        ],
        "severity": "critical",
        "icon": "⛔"
    },
    "device_fingerprint": {
        "name": "Device Fingerprinting",
        "category": "Digital Forensics",
        "description": "Tracks and analyzes device signatures",
        "detects": [
            "New device usage",
            "Device spoofing",
            "Unusual device patterns",
            "Multiple accounts from same device"
        ],
        "severity": "medium",
        "icon": "📱"
    },
    "vpn_proxy": {
        "name": "VPN/Proxy Detection",
        "category": "Digital Forensics",
        "description": "Identifies transactions from masked IP addresses",
        "detects": [
            "VPN usage",
            "Proxy servers",
            "Tor exit nodes",
            "IP masking attempts"
        ],
        "severity": "medium",
        "icon": "🕵️"
    },
    "geolocation": {
        "name": "Geo-Location Fraud",
        "category": "Location Analysis",
        "description": "Advanced geolocation fraud detection",
        "detects": [
            "High-risk country transactions",
            "Location spoofing",
            "Geographic fraud hotspots",
            "Sanctioned region activity"
        ],
        "examples": [
            "Payments routed to unexpected or high-risk countries"
        ],
        "severity": "high",
        "icon": "🗺️"
    },
    "behavioral_biometrics": {
        "name": "Behavioral Biometrics",
        "category": "Behavioral Analysis",
        "description": "Analyzes user interaction patterns",
        "detects": [
            "Typing pattern anomalies",
            "Mouse movement deviations",
            "Bot-like behavior",
            "Session timing anomalies"
        ],
        "severity": "medium",
        "icon": "👆"
    },
    "recipient_relationship": {
        "name": "Recipient Relationship Analysis",
        "category": "Relationship Analysis",
        "description": "Analyzes payment recipient patterns",
        "detects": [
            "Unusual recipient relationships",
            "Payment pattern anomalies",
            "New recipient risk",
            "Relationship trust score deviations"
        ],
        "severity": "low",
        "icon": "🤝"
    },
    "social_trust": {
        "name": "Social Trust Score",
        "category": "Reputation Analysis",
        "description": "Calculates trust scores based on social factors",
        "detects": [
            "Low social trust indicators",
            "Reputation anomalies",
            "Network trust score deviations",
            "Social graph fraud patterns"
        ],
        "severity": "low",
        "icon": "⭐"
    },
    "account_age": {
        "name": "Account Age Fraud",
        "category": "Account Analysis",
        "description": "Flags risky activity on new accounts",
        "detects": [
            "High-value transactions on new accounts",
            "Premature large transfers",
            "New account velocity abuse",
            "Account age inconsistencies"
        ],
        "examples": [
            "Unusually large transfer from a low-activity account",
            "New account immediately used for large wire transfer"
        ],
        "severity": "medium",
        "icon": "🆕"
    },
    "past_fraud_flags": {
        "name": "Past Fraudulent Behavior",
        "category": "Historical Analysis",
        "description": "Checks for previous fraud history",
        "detects": [
            "Repeat offenders",
            "Previous fraud flags",
            "Historical suspicious activity",
            "Pattern recurrence"
        ],
        "severity": "high",
        "icon": "📜"
    },
    "normalized_amount": {
        "name": "Normalized Transaction Amount",
        "category": "Amount Analysis",
        "description": "Analyzes transaction amounts relative to context",
        "detects": [
            "Structuring (just below reporting thresholds)",
            "Amount anomalies for account type",
            "Unusual amount patterns",
            "Context-relative fraud"
        ],
        "severity": "medium",
        "icon": "💵"
    },
    "payroll_fraud": {
        "name": "Payroll Rerouting Fraud",
        "category": "Payment Fraud",
        "description": "Detects payroll fraud and account rerouting",
        "detects": [
            "Unverified account changes before payroll",
            "Suspicious payroll redirects",
            "Rapid account changes",
            "Employee account takeover"
        ],
        "examples": [
            "Payroll/direct deposit rerouting (employee account details changed)",
            "Account takeover followed by immediate payroll redirect"
        ],
        "severity": "high",
        "icon": "💼"
    },
    "check_fraud": {
        "name": "Check Fraud Detection",
        "category": "Payment Fraud",
        "description": "Identifies check fraud patterns",
        "detects": [
            "Duplicate check numbers",
            "Out-of-sequence checks",
            "Unusual check amounts",
            "Check kiting patterns"
        ],
        "examples": [
            "Same check deposited more than once (duplicate deposit fraud)"
        ],
        "severity": "medium",
        "icon": "📝"
    },
    "merchant_category_mismatch": {
        "name": "Merchant Category Mismatch",
        "category": "Merchant Fraud",
        "description": "Detects MCC code violations",
        "detects": [
            "Out-of-category transactions",
            "MCC code abuse",
            "Merchant category fraud",
            "Business type mismatches"
        ],
        "severity": "low",
        "icon": "🏪"
    },
    "user_limit_exceeded": {
        "name": "User Limit Violations",
        "category": "Compliance",
        "description": "Detects transactions exceeding user limits",
        "detects": [
            "Daily limit breaches",
            "Transaction count violations",
            "Amount threshold exceeded",
            "Velocity limit abuse"
        ],
        "severity": "medium",
        "icon": "⚠️"
    },
    "high_value_flags": {
        "name": "High-Value Transaction Flags",
        "category": "Amount Analysis",
        "description": "Special scrutiny for large transactions",
        "detects": [
            "Unusually large transactions",
            "High-value fraud patterns",
            "Large transfer anomalies",
            "Suspicious big-ticket items"
        ],
        "severity": "high",
        "icon": "💎"
    }
}

def get_module_by_category():
    """Group modules by category"""
    categories = {}
    for module_id, module_info in FRAUD_MODULES_CATALOG.items():
        category = module_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append({
            "id": module_id,
            **module_info
        })
    return categories

def get_module_count():
    """Get total number of modules"""
    return len(FRAUD_MODULES_CATALOG)

def get_module_by_severity():
    """Group modules by severity"""
    severity_groups = {"critical": [], "high": [], "medium": [], "low": []}
    for module_id, module_info in FRAUD_MODULES_CATALOG.items():
        severity = module_info.get("severity", "medium")
        severity_groups[severity].append({
            "id": module_id,
            **module_info
        })
    return severity_groups
