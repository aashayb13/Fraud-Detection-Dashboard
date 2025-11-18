"""
Fraud Scenario Analysis

Deep-dive analysis of 13 fraud scenarios with detailed timelines and rule breakdowns.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from streamlit_app.theme import apply_master_theme, render_page_header, get_chart_colors
from streamlit_app.ai_recommendations import get_ai_engine, render_ai_insight
from streamlit_app.explainability import get_explainability_engine


# Complete fraud scenarios dataset (all 13 scenarios)
fraud_scenarios = {
    "1. Large Transfer - Low Activity": {
        "title": "Unusually Large Transfer from Low-Activity Account",
        "subtitle": "High-value transaction anomaly detection",
        "risk_score": 89,
        "outcome": "FRAUD CONFIRMED",
        "customer_profile": "Individual - Personal Banking",
        "transaction_type": "Wire Transfer",
        "timeline": [
            {"time": "Oct 15 - Nov 2", "event": "Normal activity: $50-150/week", "status": "normal"},
            {"time": "Nov 3, 2:30 PM", "event": "Sudden $15,000 wire transfer", "status": "critical"},
            {"time": "Nov 3, 2:31 PM", "event": "System flagged for review", "status": "flagged"},
            {"time": "Nov 3, 2:45 PM", "event": "Analyst rejected transaction", "status": "resolved"}
        ],
        "triggered_rules": [
            {"name": "Transaction Amount Anomalies", "weight": 32, "detail": "18,750% above 30-day average", "severity": "critical"},
            {"name": "Transaction Context Anomalies", "weight": 24, "detail": "No prior wire transfers", "severity": "critical"},
            {"name": "Normalized Transaction Amount", "weight": 18, "detail": "Exceeds 95th percentile for profile", "severity": "high"},
            {"name": "High-Risk Transaction Times", "weight": 9, "detail": "Outside typical banking hours", "severity": "medium"},
            {"name": "Recent High-Value Transaction Flags", "weight": 6, "detail": "First transaction over $1000", "severity": "medium"}
        ],
        "metrics": {
            "Average Transaction": "$87",
            "Current Transaction": "$15,000",
            "Standard Deviation": "172.4σ",
            "Account Age": "4 years",
            "Last Large Transaction": "Never"
        },
        "decision": {
            "recommendation": "REJECT",
            "confidence": 98,
            "reasoning": "Extreme deviation from established pattern with no prior authorization",
            "action": "Contact account holder for verification before release"
        },
        "visualization_data": {
            "amounts": [50, 75, 120, 85, 95, 110, 65, 130, 15000],
            "dates": ["Oct 15", "Oct 18", "Oct 21", "Oct 24", "Oct 27", "Oct 30", "Nov 1", "Nov 2", "Nov 3"]
        }
    },
    "2. Testing Pattern": {
        "title": "Multiple Small Transactions Followed by Large Withdrawal",
        "subtitle": "Testing pattern detection - Classic mule behavior",
        "risk_score": 92,
        "outcome": "FRAUD CONFIRMED",
        "customer_profile": "Individual - Checking Account",
        "transaction_type": "Multiple small deposits + Large withdrawal",
        "timeline": [
            {"time": "Nov 3, 10:15 AM", "event": "$1.00 test transaction - SUCCESS", "status": "warning"},
            {"time": "Nov 3, 10:18 AM", "event": "$0.50 test transaction - SUCCESS", "status": "warning"},
            {"time": "Nov 3, 10:22 AM", "event": "$2.00 test transaction - SUCCESS", "status": "warning"},
            {"time": "Nov 3, 10:35 AM", "event": "$8,500 withdrawal attempt", "status": "critical"},
            {"time": "Nov 3, 10:35 AM", "event": "System auto-blocked", "status": "blocked"}
        ],
        "triggered_rules": [
            {"name": "Transaction Frequency", "weight": 35, "detail": "4 transactions in 20 minutes", "severity": "critical"},
            {"name": "Transaction Amount Anomalies", "weight": 28, "detail": "Final amount 8,500x test amounts", "severity": "critical"},
            {"name": "Behavioral Biometrics", "weight": 15, "detail": "Automated script pattern detected", "severity": "high"},
            {"name": "Transaction Context Anomalies", "weight": 10, "detail": "Testing followed by exploitation", "severity": "high"},
            {"name": "Past Fraudulent Behavior Flags", "weight": 4, "detail": "Similar pattern 30 days ago", "severity": "medium"}
        ],
        "metrics": {
            "Test Transactions": "3",
            "Test Window": "7 minutes",
            "Exploit Amount": "$8,500",
            "Time Between": "13 minutes",
            "Pattern Match": "Known fraud signature"
        },
        "decision": {
            "recommendation": "AUTO-REJECT",
            "confidence": 99,
            "reasoning": "Classic account validation followed by exploitation pattern",
            "action": "Block account, initiate fraud investigation"
        },
        "visualization_data": {
            "amounts": [1, 0.5, 2, 8500],
            "times": ["10:15", "10:18", "10:22", "10:35"],
            "types": ["test", "test", "test", "exploit"]
        }
    },
    "3. Payroll Rerouting": {
        "title": "Payroll/Direct Deposit Rerouting",
        "subtitle": "Business Email Compromise (BEC) - HR impersonation",
        "risk_score": 85,
        "outcome": "FRAUD PREVENTED",
        "customer_profile": "Individual - Employee",
        "transaction_type": "Payroll Direct Deposit",
        "timeline": [
            {"time": "Oct 1 - Oct 31", "event": "Normal payroll to Account ***1234", "status": "normal"},
            {"time": "Nov 2, 3:45 PM", "event": "Account update request via email", "status": "warning"},
            {"time": "Nov 2, 4:10 PM", "event": "New routing added: Account ***9876", "status": "flagged"},
            {"time": "Nov 3, 9:00 AM", "event": "Payroll redirected to new account", "status": "critical"},
            {"time": "Nov 3, 9:01 AM", "event": "System flagged - Analyst review", "status": "review"}
        ],
        "triggered_rules": [
            {"name": "Recipient Verification Status", "weight": 26, "detail": "New beneficiary added yesterday", "severity": "critical"},
            {"name": "Time Since Last Transaction with Recipient", "weight": 24, "detail": "First transaction to this recipient", "severity": "critical"},
            {"name": "Social Trust Score", "weight": 18, "detail": "No relationship history", "severity": "high"},
            {"name": "Transaction Context Anomalies", "weight": 12, "detail": "Payroll pattern break", "severity": "high"},
            {"name": "High-Risk Transaction Times", "weight": 5, "detail": "Immediate payment after update", "severity": "medium"}
        ],
        "metrics": {
            "Recipient Age": "<24 hours",
            "Prior Transactions": "0",
            "Request Source": "Unverified email",
            "Payroll History": "18 months consistent",
            "Verification Status": "FAILED"
        },
        "decision": {
            "recommendation": "ESCALATE",
            "confidence": 94,
            "reasoning": "High-value recurring payment to unverified new recipient",
            "action": "Contact employee directly (not via email) to verify account change"
        }
    },
    "4. Money Mule": {
        "title": "Account Used as Money Mule",
        "subtitle": "Rapid in-out transaction pattern indicating money laundering",
        "risk_score": 94,
        "outcome": "FRAUD CONFIRMED",
        "customer_profile": "Individual - New Account",
        "transaction_type": "Multiple in/out transfers",
        "timeline": [
            {"time": "Nov 3, 8:15 AM", "event": "Incoming: $500 from Account A", "status": "warning"},
            {"time": "Nov 3, 8:42 AM", "event": "Incoming: $750 from Account B", "status": "warning"},
            {"time": "Nov 3, 9:10 AM", "event": "Incoming: $1,200 from Account C", "status": "warning"},
            {"time": "Nov 3, 9:55 AM", "event": "Outgoing: $2,380 to offshore account", "status": "critical"},
            {"time": "Nov 3, 10:20 AM", "event": "Incoming: $900 from Account D", "status": "critical"}
        ],
        "triggered_rules": [
            {"name": "Transaction Frequency", "weight": 32, "detail": "5 transactions in 2 hours", "severity": "critical"},
            {"name": "Transaction Context Anomalies", "weight": 28, "detail": "Multiple sources, single destination", "severity": "critical"},
            {"name": "Geo-Location Flags", "weight": 20, "detail": "Outbound to high-risk jurisdiction", "severity": "critical"},
            {"name": "Account Age", "weight": 8, "detail": "Account opened 12 days ago", "severity": "high"},
            {"name": "Social Trust Score", "weight": 6, "detail": "No established relationships", "severity": "medium"}
        ],
        "metrics": {
            "Incoming Transactions": "4",
            "Total Incoming": "$3,350",
            "Outgoing Transactions": "1",
            "Total Outgoing": "$2,380",
            "Retention Rate": "29%",
            "Account Age": "12 days"
        },
        "decision": {
            "recommendation": "REJECT & FREEZE",
            "confidence": 97,
            "reasoning": "Clear money mule behavior - rapid pass-through to offshore account",
            "action": "Freeze account, report to financial crimes unit"
        },
        "visualization_data": {
            "flow": {
                "incoming": [500, 750, 1200, 900],
                "outgoing": [2380],
                "sources": ["Account A", "Account B", "Account C", "Account D"],
                "destination": "Offshore Account"
            }
        }
    },
    "5. Account Takeover": {
        "title": "Account Takeover with Phone/SIM Changes",
        "subtitle": "Credential compromise with device manipulation",
        "risk_score": 96,
        "outcome": "FRAUD PREVENTED",
        "customer_profile": "Individual - Mobile Banking",
        "transaction_type": "Large transfer",
        "timeline": [
            {"time": "Oct 1 - Nov 2", "event": "Normal usage: iPhone 13, Dallas TX", "status": "normal"},
            {"time": "Nov 3, 1:45 AM", "event": "Phone number change request", "status": "warning"},
            {"time": "Nov 3, 2:10 AM", "event": "New device login: Android, Lagos", "status": "critical"},
            {"time": "Nov 3, 2:15 AM", "event": "VPN connection detected", "status": "critical"},
            {"time": "Nov 3, 2:18 AM", "event": "$12,000 transfer attempt - BLOCKED", "status": "blocked"}
        ],
        "triggered_rules": [
            {"name": "Device Fingerprinting", "weight": 30, "detail": "100% device profile change", "severity": "critical"},
            {"name": "VPN or Proxy Usage", "weight": 22, "detail": "Masked IP from Lagos, Nigeria", "severity": "critical"},
            {"name": "Geo-Location Flags", "weight": 20, "detail": "High-risk country access", "severity": "critical"},
            {"name": "Behavioral Biometrics", "weight": 15, "detail": "Typing pattern 87% different", "severity": "high"},
            {"name": "High-Risk Transaction Times", "weight": 9, "detail": "2 AM activity (never before)", "severity": "high"}
        ],
        "metrics": {
            "Device Change": "iPhone → Android",
            "Location Change": "Dallas → Lagos (6,147 mi)",
            "Time Gap": "25 minutes",
            "Typing Speed": "68 wpm → 23 wpm",
            "Phone Number Change": "Yes (1:45 AM)"
        },
        "decision": {
            "recommendation": "AUTO-REJECT",
            "confidence": 99,
            "reasoning": "All indicators of account takeover - SIM swap + credential access",
            "action": "Lock account, require in-person identity verification"
        },
        "visualization_data": {
            "device_comparison": {
                "normal": {"device": "iPhone 13", "location": "Dallas, TX", "vpn": "No", "typing_wpm": 68},
                "suspicious": {"device": "Android 12", "location": "Lagos, Nigeria", "vpn": "Yes", "typing_wpm": 23}
            }
        }
    },
    "6. Duplicate Check": {
        "title": "Same Check Deposited Multiple Times",
        "subtitle": "Check fraud - Duplicate deposit across institutions",
        "risk_score": 88,
        "outcome": "FRAUD CONFIRMED",
        "customer_profile": "Individual - Multiple bank accounts",
        "transaction_type": "Check deposit",
        "timeline": [
            {"time": "Nov 1, 10:00 AM", "event": "Check #4521 deposited - Bank A", "status": "normal"},
            {"time": "Nov 1, 10:15 AM", "event": "Check cleared - $3,200", "status": "normal"},
            {"time": "Nov 3, 2:30 PM", "event": "Same check #4521 deposited - Bank B", "status": "critical"},
            {"time": "Nov 3, 2:31 PM", "event": "Cross-bank duplicate detected", "status": "flagged"},
            {"time": "Nov 3, 2:32 PM", "event": "Deposit rejected automatically", "status": "blocked"}
        ],
        "triggered_rules": [
            {"name": "Past Fraudulent Behavior Flags", "weight": 35, "detail": "Check already cleared elsewhere", "severity": "critical"},
            {"name": "Transaction Context Anomalies", "weight": 28, "detail": "Identical check number/amount", "severity": "critical"},
            {"name": "Fraud Complaints Count", "weight": 12, "detail": "1 prior check fraud flag (2024)", "severity": "high"},
            {"name": "Transaction Frequency", "weight": 8, "detail": "Multiple deposit attempts", "severity": "medium"},
            {"name": "Account Age", "weight": 5, "detail": "Account opened 3 months ago", "severity": "medium"}
        ],
        "metrics": {
            "Check Number": "#4521",
            "Amount": "$3,200",
            "First Deposit": "Bank A (Nov 1)",
            "Second Deposit": "Bank B (Nov 3)",
            "Days Between": "2",
            "Check Status": "Already cleared"
        },
        "decision": {
            "recommendation": "AUTO-REJECT",
            "confidence": 100,
            "reasoning": "Definitive duplicate check - already cleared at another institution",
            "action": "Reject deposit, flag account for check fraud investigation"
        }
    },
    "7. Vendor Impersonation": {
        "title": "Payment to Newly Added Beneficiary (Vendor Impersonation)",
        "subtitle": "Invoice fraud - Fake supplier account substitution",
        "risk_score": 91,
        "outcome": "FRAUD PREVENTED",
        "customer_profile": "Small Business - Accounts Payable",
        "transaction_type": "Vendor payment",
        "timeline": [
            {"time": "Jan - Oct", "event": "Regular payments to Supplier XYZ (***1234)", "status": "normal"},
            {"time": "Nov 2, 4:20 PM", "event": "Email: 'Update bank account details'", "status": "warning"},
            {"time": "Nov 2, 4:45 PM", "event": "New beneficiary added: Supplier XYZ (***9999)", "status": "flagged"},
            {"time": "Nov 3, 9:00 AM", "event": "$45,000 payment to new account", "status": "critical"},
            {"time": "Nov 3, 9:01 AM", "event": "System holds for review", "status": "review"}
        ],
        "triggered_rules": [
            {"name": "Recipient Verification Status", "weight": 28, "detail": "Beneficiary added <24 hrs ago", "severity": "critical"},
            {"name": "Time Since Last Transaction with Recipient", "weight": 26, "detail": "First payment to this account", "severity": "critical"},
            {"name": "Social Trust Score", "weight": 18, "detail": "No transaction history", "severity": "high"},
            {"name": "Transaction Amount Anomalies", "weight": 14, "detail": "Above average vendor payment", "severity": "high"},
            {"name": "High-Risk Transaction Times", "weight": 5, "detail": "Same-day payment after update", "severity": "medium"}
        ],
        "metrics": {
            "Vendor History": "10 months, 24 payments",
            "Previous Account": "***1234 (Verified)",
            "New Account": "***9999 (Unverified)",
            "Change Request": "Email (unverified domain)",
            "Payment Timing": "<16 hours after change",
            "Amount": "$45,000"
        },
        "decision": {
            "recommendation": "ESCALATE - HIGH PRIORITY",
            "confidence": 96,
            "reasoning": "Classic vendor impersonation - immediate large payment to unverified account",
            "action": "Contact vendor via KNOWN phone number to verify account change"
        }
    },
    "8. High-Risk Country": {
        "title": "Payments to Unexpected High-Risk Countries",
        "subtitle": "Geographic anomaly - Vendor location switch",
        "risk_score": 87,
        "outcome": "FRAUD SUSPECTED",
        "customer_profile": "Small Business - International trade",
        "transaction_type": "Wire transfer",
        "timeline": [
            {"time": "Jan - Oct", "event": "All payments: US domestic accounts", "status": "normal"},
            {"time": "Nov 1, 10:00 AM", "event": "Invoice received from 'vendor'", "status": "warning"},
            {"time": "Nov 3, 11:30 AM", "event": "$28,000 payment to account in Belarus", "status": "critical"},
            {"time": "Nov 3, 11:31 AM", "event": "Geographic anomaly flagged", "status": "flagged"},
            {"time": "Nov 3, 11:45 AM", "event": "Analyst escalates to manager", "status": "escalated"}
        ],
        "triggered_rules": [
            {"name": "Geo-Location Flags", "weight": 32, "detail": "Payment to high-risk country (Belarus)", "severity": "critical"},
            {"name": "Transaction Context Anomalies", "weight": 25, "detail": "No prior international payments", "severity": "critical"},
            {"name": "Recipient Verification Status", "weight": 18, "detail": "New international beneficiary", "severity": "high"},
            {"name": "Transaction Amount Anomalies", "weight": 8, "detail": "Above average payment", "severity": "medium"},
            {"name": "Merchant Category Mismatch", "weight": 4, "detail": "Domestic vendor, international payment", "severity": "medium"}
        ],
        "metrics": {
            "Vendor Profile": "US Domestic (10 months)",
            "New Location": "Belarus (High-risk)",
            "Prior International": "0 payments",
            "Country Risk Score": "9/10",
            "Amount": "$28,000",
            "Sanctions Check": "Required"
        },
        "decision": {
            "recommendation": "HOLD FOR REVIEW",
            "confidence": 93,
            "reasoning": "Unexplained geographic shift to high-risk jurisdiction",
            "action": "Verify vendor authenticity, check sanctions compliance"
        },
        "visualization_data": {
            "geographic": {
                "historical": {"country": "United States", "count": 24, "risk": 1},
                "current": {"country": "Belarus", "count": 1, "risk": 9}
            }
        }
    },
    "9. Bulk Beneficiary": {
        "title": "Rapid Addition of Multiple Beneficiaries",
        "subtitle": "Scripted fraud - Mass beneficiary creation",
        "risk_score": 93,
        "outcome": "FRAUD CONFIRMED",
        "customer_profile": "Corporation - Treasury Management",
        "transaction_type": "Bulk payments",
        "timeline": [
            {"time": "Nov 3, 1:00 AM", "event": "11 new beneficiaries added", "status": "critical"},
            {"time": "Nov 3, 1:15 AM", "event": "Payment #1: $4,500 to Beneficiary A", "status": "critical"},
            {"time": "Nov 3, 1:18 AM", "event": "Payment #2: $3,800 to Beneficiary B", "status": "critical"},
            {"time": "Nov 3, 1:22 AM", "event": "System detects pattern - AUTO BLOCKS", "status": "blocked"},
            {"time": "Nov 3, 1:23 AM", "event": "Remaining 9 payments blocked", "status": "blocked"}
        ],
        "triggered_rules": [
            {"name": "Recipient Verification Status", "weight": 30, "detail": "11 beneficiaries in 15 minutes", "severity": "critical"},
            {"name": "Transaction Frequency", "weight": 28, "detail": "Automated script detected", "severity": "critical"},
            {"name": "High-Risk Transaction Times", "weight": 18, "detail": "1 AM bulk operations", "severity": "critical"},
            {"name": "Behavioral Biometrics", "weight": 12, "detail": "Non-human interaction pattern", "severity": "high"},
            {"name": "Social Trust Score", "weight": 5, "detail": "All new, unverified recipients", "severity": "medium"}
        ],
        "metrics": {
            "Beneficiaries Added": "11",
            "Time Window": "15 minutes",
            "Payments Attempted": "11",
            "Payments Blocked": "9",
            "Total At Risk": "$47,200",
            "Script Detection": "Confirmed"
        },
        "decision": {
            "recommendation": "AUTO-BLOCK & FREEZE",
            "confidence": 99,
            "reasoning": "Clear scripted attack - mass beneficiary creation + rapid fund distribution",
            "action": "Freeze account, reverse completed transactions, investigate credential compromise"
        }
    },
    "10. Odd Hours": {
        "title": "Large Transaction at Odd Hours",
        "subtitle": "Temporal anomaly - After-hours high-value transfer",
        "risk_score": 84,
        "outcome": "INVESTIGATED",
        "customer_profile": "Individual - Business owner",
        "transaction_type": "Wire transfer",
        "timeline": [
            {"time": "Regular hours", "event": "Normal activity 9 AM - 5 PM", "status": "normal"},
            {"time": "Nov 3, 3:17 AM", "event": "Login from usual device/location", "status": "warning"},
            {"time": "Nov 3, 3:22 AM", "event": "$22,000 wire transfer initiated", "status": "critical"},
            {"time": "Nov 3, 3:23 AM", "event": "System flags for review", "status": "flagged"},
            {"time": "Nov 3, 9:05 AM", "event": "Analyst contacts customer", "status": "review"}
        ],
        "triggered_rules": [
            {"name": "High-Risk Transaction Times", "weight": 28, "detail": "3:22 AM transaction (never before)", "severity": "critical"},
            {"name": "Transaction Amount Anomalies", "weight": 24, "detail": "Highest single transaction", "severity": "high"},
            {"name": "Transaction Context Anomalies", "weight": 16, "detail": "No prior 3 AM activity", "severity": "high"},
            {"name": "Behavioral Biometrics", "weight": 12, "detail": "Faster-than-usual navigation", "severity": "medium"},
            {"name": "Transaction Frequency", "weight": 4, "detail": "Immediate logout after transaction", "severity": "low"}
        ],
        "metrics": {
            "Transaction Time": "3:22 AM",
            "User Typical Hours": "9 AM - 6 PM",
            "Amount": "$22,000",
            "Previous Max Amount": "$8,500",
            "Session Duration": "6 min (vs avg 18 min)",
            "Device Match": "✓ Recognized"
        },
        "decision": {
            "recommendation": "HOLD - VERIFY",
            "confidence": 87,
            "reasoning": "Unusual timing with high amount, but device recognized",
            "action": "Contact customer immediately to verify transaction authenticity"
        }
    },
    "11. Social Engineering": {
        "title": "Social Engineering Push Payment (P2P Scam)",
        "subtitle": "Authorized push payment fraud - Victim manipulation",
        "risk_score": 76,
        "outcome": "FRAUD (VICTIM AUTHORIZED)",
        "customer_profile": "Individual - Elderly customer",
        "transaction_type": "P2P payment (Zelle/Venmo)",
        "timeline": [
            {"time": "Nov 3, 10:15 AM", "event": "User receives 'bank fraud alert' call", "status": "warning"},
            {"time": "Nov 3, 10:30 AM", "event": "User logs in and sends $5,000 to 'verify'", "status": "critical"},
            {"time": "Nov 3, 10:31 AM", "event": "System flags suspicious recipient", "status": "flagged"},
            {"time": "Nov 3, 10:45 AM", "event": "User reports scam - too late", "status": "reported"},
            {"time": "Nov 3, 11:00 AM", "event": "Analyst documents authorized fraud", "status": "logged"}
        ],
        "triggered_rules": [
            {"name": "Recipient Verification Status", "weight": 22, "detail": "Recipient created 2 days ago", "severity": "high"},
            {"name": "Social Trust Score", "weight": 20, "detail": "No relationship, not in contacts", "severity": "high"},
            {"name": "Time Since Last Transaction with Recipient", "weight": 18, "detail": "First transaction", "severity": "high"},
            {"name": "Transaction Context Anomalies", "weight": 10, "detail": "Unusual P2P pattern", "severity": "medium"},
            {"name": "Recipient Blacklist Status", "weight": 6, "detail": "Similar recipient reported 5 times", "severity": "medium"}
        ],
        "metrics": {
            "Recipient Age": "2 days",
            "Recipient Relationship": "None",
            "Fraud Reports": "47 similar this month",
            "User Initiated": "Yes (authorized)",
            "Amount": "$5,000",
            "Recovery Possibility": "Low"
        },
        "decision": {
            "recommendation": "FLAG - CANNOT BLOCK",
            "confidence": 82,
            "reasoning": "User authorized transaction despite red flags",
            "action": "Document as authorized fraud, add recipient to watchlist, user education"
        }
    },
    "12. Test Deposits": {
        "title": "Tiny Test Deposits for Account Validation",
        "subtitle": "Reconnaissance phase - Account probing before theft",
        "risk_score": 71,
        "outcome": "EARLY DETECTION",
        "customer_profile": "Individual - Checking account",
        "transaction_type": "Micro deposits",
        "timeline": [
            {"time": "Nov 3, 8:00 AM", "event": "$0.01 deposit from unknown source", "status": "warning"},
            {"time": "Nov 3, 8:05 AM", "event": "$0.02 deposit from unknown source", "status": "warning"},
            {"time": "Nov 3, 8:10 AM", "event": "$0.01 deposit from different source", "status": "flagged"},
            {"time": "Nov 3, 8:11 AM", "event": "Pattern detected - Monitoring activated", "status": "monitoring"},
            {"time": "Nov 3, 2:00 PM", "event": "Large withdrawal attempt - BLOCKED", "status": "blocked"}
        ],
        "triggered_rules": [
            {"name": "Transaction Frequency", "weight": 24, "detail": "3 micro-deposits in 10 minutes", "severity": "high"},
            {"name": "Transaction Context Anomalies", "weight": 20, "detail": "Penny testing pattern", "severity": "high"},
            {"name": "Recipient Verification Status", "weight": 14, "detail": "Unknown senders", "severity": "medium"},
            {"name": "Behavioral Biometrics", "weight": 8, "detail": "Automated validation pattern", "severity": "medium"},
            {"name": "Past Fraudulent Behavior Flags", "weight": 5, "detail": "Known fraud technique", "severity": "medium"}
        ],
        "metrics": {
            "Test Deposits": "3",
            "Total Amount": "$0.04",
            "Time Window": "10 minutes",
            "Exploit Attempt": "6 hours later",
            "Exploit Amount": "$7,500",
            "Outcome": "Blocked"
        },
        "decision": {
            "recommendation": "MONITOR & ALERT",
            "confidence": 88,
            "reasoning": "Classic testing pattern - likely precursor to larger fraud attempt",
            "action": "Enhanced monitoring for 48 hours, block large withdrawals, notify customer"
        },
        "visualization_data": {
            "test_pattern": {
                "phase1": [0.01, 0.02, 0.01],
                "phase2": 7500,
                "time_gap_hours": 6
            }
        }
    },
    "13. Refund Chain": {
        "title": "Complex Refund and Transfer Chains",
        "subtitle": "Money laundering - Obfuscation through transaction layering",
        "risk_score": 90,
        "outcome": "FRAUD CONFIRMED",
        "customer_profile": "Individual - E-commerce seller",
        "transaction_type": "Refunds and transfers",
        "timeline": [
            {"time": "Nov 1, 9:00 AM", "event": "Small payment received: $150", "status": "normal"},
            {"time": "Nov 1, 9:30 AM", "event": "Refund request: $175 to different account", "status": "warning"},
            {"time": "Nov 2, 2:00 PM", "event": "Transfer: $150 to Account B", "status": "warning"},
            {"time": "Nov 3, 10:00 AM", "event": "Refund: $200 to Account C", "status": "flagged"},
            {"time": "Nov 3, 10:15 AM", "event": "Complex chain detected - Flagged", "status": "critical"}
        ],
        "triggered_rules": [
            {"name": "Transaction Context Anomalies", "weight": 30, "detail": "Refund chain to different accounts", "severity": "critical"},
            {"name": "Transaction Frequency", "weight": 22, "detail": "7 transactions in unusual pattern", "severity": "high"},
            {"name": "Social Trust Score", "weight": 18, "detail": "No relationship between accounts", "severity": "high"},
            {"name": "Recipient Verification Status", "weight": 12, "detail": "Multiple unverified recipients", "severity": "medium"},
            {"name": "Transaction Amount Anomalies", "weight": 8, "detail": "Refund exceeds original payment", "severity": "medium"}
        ],
        "metrics": {
            "Total Transactions": "7",
            "Unique Accounts": "5",
            "Refund Percentage": "117% of received",
            "Time Span": "3 days",
            "Pattern Type": "Layering",
            "Complexity Score": "High"
        },
        "decision": {
            "recommendation": "FREEZE & INVESTIGATE",
            "confidence": 95,
            "reasoning": "Complex transaction chain designed to obscure fund origin",
            "action": "Freeze all accounts involved, investigate source of funds, report to AML unit"
        },
        "visualization_data": {
            "chain": [
                {"from": "Account X", "to": "Main", "amount": 150, "type": "payment"},
                {"from": "Main", "to": "Account A", "amount": 175, "type": "refund"},
                {"from": "Main", "to": "Account B", "amount": 150, "type": "transfer"},
                {"from": "Main", "to": "Account C", "amount": 200, "type": "refund"}
            ]
        }
    }
}


def render():
    """Render the Fraud Scenario Analysis page"""

    # Apply theme
    apply_master_theme()

    # Header
    render_page_header(
        title="Attack Pattern Simulator",
        subtitle="Comprehensive Analysis of 13 Fraud Detection Scenarios",
        show_logo=False
    )

    # Get standardized chart colors
    colors = get_chart_colors()

    # --- Top-of-page scenario selector (main area) ---
    st.markdown("### 🔽 Scenario Selector")
    scenario_key_top = st.selectbox(
        "Select a fraud scenario to analyze:",
        options=list(fraud_scenarios.keys()),
        format_func=lambda x: fraud_scenarios[x]['title'],
        key="scenario_key_top",
    )
    st.markdown("---")

    # Sidebar for scenario selection
    with st.sidebar:
        st.markdown("### 🔍 Select Fraud Scenario")

        scenario_key = st.selectbox(
            "Choose a scenario to analyze:",
            options=list(fraud_scenarios.keys()),
            format_func=lambda x: fraud_scenarios[x]['title']
        )

        st.markdown("---")
        st.markdown("### 📊 Display Options")
        show_visualizations = st.checkbox("Show Advanced Visualizations", value=True)
        show_metrics = st.checkbox("Show Detailed Metrics", value=True)
        show_timeline = st.checkbox("Show Timeline", value=True)

    # Prefer the top dropdown selection if present; fall back to sidebar
    active_scenario_key = st.session_state.get("scenario_key_top", None) or scenario_key
    scenario = fraud_scenarios[active_scenario_key]

    # Risk Score Header
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"### {scenario['title']}")
        st.caption(scenario['subtitle'])
    with col2:
        risk_color = "🔴" if scenario['risk_score'] >= 90 else "🟠" if scenario['risk_score'] >= 75 else "🟡"
        st.metric("Risk Score", f"{scenario['risk_score']}/100 {risk_color}")
    with col3:
        st.metric("Profile", scenario['customer_profile'].split(' - ')[0])
    with col4:
        st.metric("Outcome", scenario['outcome'])
            # Key Information Row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Transaction Type:** {scenario['transaction_type']}")
    with col2:
        st.markdown(f"**Customer Profile:** {scenario['customer_profile']}")

    st.markdown("---")

    st.markdown("### 🎯 Analyst Decision & Recommendation")

    decision_col1, decision_col2 = st.columns([2, 1])

    with decision_col1:
        st.markdown(f"**Recommendation:** `{scenario['decision']['recommendation']}`")
        st.markdown(f"**Confidence Level:** {scenario['decision']['confidence']}%")
        
        # Confidence bar
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scenario['decision']['confidence'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10b981" if scenario['decision']['confidence'] >= 90 else "#f97316"},
                'steps': [
                    {'range': [0, 60], 'color': "#fee2e2"},
                    {'range': [60, 80], 'color': "#fef3c7"},
                    {'range': [80, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_conf.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_conf, use_container_width=True)
        
        st.markdown(f"**Reasoning:** {scenario['decision']['reasoning']}")
        st.markdown(f"**Recommended Action:** {scenario['decision']['action']}")

        # AI Analysis
        st.markdown("---")
        st.markdown("#### 🤖 AI Analysis")

        ai_engine = get_ai_engine()
        scenario_insight = ai_engine.get_pattern_insight(
            pattern_type="fraud_scenario",
            pattern_data={
                "scenario_type": scenario['title'],
                "risk_score": scenario['risk_score'],
                "outcome": scenario['outcome'],
                "rules_triggered": len(scenario['triggered_rules']),
                "confidence": scenario['decision']['confidence']
            }
        )

        st.info(scenario_insight)

    # Timeline Section
    if show_timeline:
        st.markdown("### ⏱️ Detection Timeline")
        
        timeline_df = pd.DataFrame(scenario['timeline'])
        
        for idx, row in timeline_df.iterrows():
            status_class = {
                'normal': '🟢', 'warning': '🟡', 'flagged': '🟠', 'critical': '🔴',
                'blocked': '🟣', 'review': '🔵', 'resolved': '⚫', 'escalated': '🔴',
                'reported': '🟠', 'monitoring': '🔵', 'logged': '⚫'
            }.get(row['status'], '⚪')
            
            st.markdown(f"{status_class} **{row['time']}** - {row['event']}")
        
        st.markdown("---")

    # Triggered Rules Section
    st.markdown("### 🚨 Triggered Rules & Risk Contribution")

    # Create rule contribution chart
    rule_df = pd.DataFrame(scenario['triggered_rules'])
    rule_df = rule_df.sort_values('weight', ascending=True)

    # Enhanced hover texts with explainability
    rule_hover_texts = []
    total_weight = rule_df['weight'].sum()

    for _, row in rule_df.iterrows():
        rule_name = row['name']
        weight = row['weight']
        detail = row['detail']
        severity = row['severity']

        # Calculate contribution percentage
        contribution_pct = (weight / total_weight) * 100 if total_weight > 0 else 0

        # Severity assessment
        severity_info = {
            'critical': {
                'badge': '🔴 CRITICAL',
                'color': '#dc2626',
                'impact': 'Major fraud indicator - Extremely suspicious behavior',
                'action': 'This alone warrants investigation'
            },
            'high': {
                'badge': '🟠 HIGH',
                'color': '#f59e0b',
                'impact': 'Strong fraud signal - Significant risk factor',
                'action': 'Important contributor to overall risk'
            },
            'medium': {
                'badge': '🟡 MODERATE',
                'color': '#eab308',
                'impact': 'Notable concern - Adds to risk profile',
                'action': 'Supporting evidence for fraud detection'
            },
            'low': {
                'badge': '🔵 LOW',
                'color': '#3b82f6',
                'impact': 'Minor flag - Supplementary indicator',
                'action': 'Minimal contribution to risk score'
            }
        }

        sev_info = severity_info.get(severity, severity_info['medium'])

        # Impact explanation
        if weight >= 30:
            impact_level = "DOMINANT FACTOR"
            impact_note = f"This rule alone accounts for {contribution_pct:.0f}% of the risk score"
        elif weight >= 20:
            impact_level = "MAJOR CONTRIBUTOR"
            impact_note = f"Significant {contribution_pct:.0f}% contribution to total risk"
        elif weight >= 10:
            impact_level = "MODERATE IMPACT"
            impact_note = f"Notable {contribution_pct:.0f}% of the risk assessment"
        else:
            impact_level = "SUPPORTING EVIDENCE"
            impact_note = f"Adds {contribution_pct:.0f}% to overall risk picture"

        hover_text = (
            f"<b style='font-size:14px'>{rule_name}</b><br><br>"
            f"<b style='color:{sev_info['color']}'>{sev_info['badge']} SEVERITY</b><br>"
            f"{sev_info['impact']}<br><br>"
            f"<b>📊 Risk Contribution:</b><br>"
            f"• Risk Points: <b>+{weight}</b><br>"
            f"• Percentage of Total: <b>{contribution_pct:.1f}%</b><br>"
            f"• Impact Level: <b>{impact_level}</b><br><br>"
            f"<b>🔍 Detection Detail:</b><br>"
            f"{detail}<br><br>"
            f"<b>💡 What This Means:</b><br>"
            f"{impact_note}<br><br>"
            f"<b>🎯 Analysis Impact:</b><br>"
            f"{sev_info['action']}<br><br>"
            f"<b>📈 Cumulative Effect:</b><br>"
            f"Without this rule, score would be <b>{scenario['risk_score'] - weight}</b> instead of <b>{scenario['risk_score']}</b>"
        )
        rule_hover_texts.append(hover_text)

    fig_rules = go.Figure()
    fig_rules.add_trace(go.Bar(
        y=rule_df['name'],
        x=rule_df['weight'],
        orientation='h',
        marker=dict(
            color=rule_df['severity'].map({
                'critical': '#ef4444', 'high': '#f97316', 'medium': '#eab308', 'low': '#3b82f6'
            }),
            line=dict(color='white', width=1)
        ),
        text=rule_df['weight'],
        textposition='outside',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=rule_hover_texts
    ))

    fig_rules.update_layout(
        title="Rule Weight Contribution to Risk Score",
        xaxis_title="Risk Points Added",
        yaxis_title="",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_rules, use_container_width=True)

    # Detailed rules table
    st.markdown("#### Detailed Rule Analysis")
    for idx, rule in enumerate(scenario['triggered_rules']):
        severity_emoji = {
            'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🔵'
        }.get(rule['severity'], '⚪')
        
        with st.expander(f"{severity_emoji} {rule['name']} (+{rule['weight']} points)"):
            st.markdown(f"**Severity:** {rule['severity'].upper()}")
            st.markdown(f"**Detail:** {rule['detail']}")
            st.markdown(f"**Risk Contribution:** {rule['weight']} points")

    st.markdown("---")

    # Metrics Section
    if show_metrics:
        st.markdown("### 📈 Key Detection Metrics")
        
        metrics_cols = st.columns(len(scenario['metrics']))
        for idx, (key, value) in enumerate(scenario['metrics'].items()):
            with metrics_cols[idx]:
                st.metric(key, value)
        
        st.markdown("---")

    # Advanced Visualizations
    if show_visualizations and 'visualization_data' in scenario:
        st.markdown("### 📊 Advanced Analysis Visualizations")
        
        viz_data = scenario['visualization_data']
        
        # Scenario 1: Transaction Amount Timeline
        if 'amounts' in viz_data and 'dates' in viz_data:
            fig_amount = go.Figure()

            amounts = viz_data['amounts']
            dates = viz_data['dates']
            avg = sum(amounts[:-1]) / len(amounts[:-1])

            # Enhanced hover for normal transactions
            normal_hover_texts = []
            for date, amount in zip(dates[:-1], amounts[:-1]):
                deviation_pct = ((amount - avg) / avg * 100) if avg > 0 else 0

                if abs(deviation_pct) < 20:
                    status = "🟢 Normal"
                    assessment = "Within expected range"
                elif abs(deviation_pct) < 50:
                    status = "🟡 Slight Variation"
                    assessment = "Minor deviation from average"
                else:
                    status = "🟠 Notable"
                    assessment = "Larger than typical but not alarming"

                hover_text = (
                    f"<b>Date:</b> {date}<br>"
                    f"<b>Amount:</b> ${amount}<br><br>"
                    f"<b>Status:</b> {status}<br>"
                    f"<b>vs Average:</b> ${avg:.0f}<br>"
                    f"<b>Deviation:</b> {deviation_pct:+.1f}%<br><br>"
                    f"<b>💡 Assessment:</b> {assessment}"
                )
                normal_hover_texts.append(hover_text)

            # Normal transactions
            fig_amount.add_trace(go.Scatter(
                x=dates[:-1],
                y=amounts[:-1],
                mode='lines+markers',
                name='Normal Activity',
                line=dict(color='#10b981', width=2),
                marker=dict(size=8),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=normal_hover_texts
            ))

            # Enhanced hover for flagged transaction
            flagged_amount = amounts[-1]
            flagged_date = dates[-1]
            increase_pct = ((flagged_amount - avg) / avg * 100) if avg > 0 else 0
            std_deviations = (flagged_amount - avg) / (sum([(x-avg)**2 for x in amounts[:-1]]) / len(amounts[:-1]))**0.5

            flagged_hover = (
                f"<b style='font-size:14px; color:#dc2626'>🚨 FLAGGED TRANSACTION</b><br><br>"
                f"<b>Date:</b> {flagged_date}<br>"
                f"<b>Amount:</b> <b style='color:#dc2626'>${flagged_amount:,}</b><br><br>"
                f"<b>📊 Anomaly Metrics:</b><br>"
                f"• Average Transaction: <b>${avg:.0f}</b><br>"
                f"• This Transaction: <b>${flagged_amount:,}</b><br>"
                f"• Increase: <b>+{increase_pct:.0f}%</b><br>"
                f"• Standard Deviations: <b>{std_deviations:.1f}σ</b><br><br>"
                f"<b>🔴 Why This Was Flagged:</b><br>"
                f"This transaction is <b>{flagged_amount/avg:.1f}x</b> larger than normal<br>"
                f"activity, representing a <b>{increase_pct:.0f}%</b> spike that is<br>"
                f"<b>{std_deviations:.0f}</b> standard deviations from typical behavior.<br><br>"
                f"<b>🎯 Risk Assessment:</b><br>"
                f"Extreme deviation from established spending pattern.<br>"
                f"This level of anomaly warrants immediate investigation.<br><br>"
                f"<b>💡 Context:</b><br>"
                f"Sudden large transfers from dormant or low-activity<br>"
                f"accounts are classic indicators of account takeover."
            )

            # Flagged transaction
            fig_amount.add_trace(go.Scatter(
                x=[dates[-1]],
                y=[amounts[-1]],
                mode='markers',
                name='Flagged Transaction',
                marker=dict(size=20, color='#ef4444', symbol='star'),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=[flagged_hover]
            ))

            # Average line
            fig_amount.add_hline(y=avg, line_dash="dash", line_color="gray",
                                 annotation_text=f"Average: ${avg:.0f}")

            fig_amount.update_layout(
                title="Transaction Amount Over Time",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                height=400,
                hovermode='closest'
            )

            st.plotly_chart(fig_amount, use_container_width=True)
        
        # Scenario 2: Testing Pattern
        if 'times' in viz_data and 'types' in viz_data:
            fig_test = go.Figure()

            colors = ['#fbbf24' if t == 'test' else '#ef4444' for t in viz_data['types']]

            # Enhanced hover for testing pattern
            test_hover_texts = []
            for idx, (time, amount, tx_type) in enumerate(zip(viz_data['times'], viz_data['amounts'], viz_data['types'])):
                if tx_type == 'test':
                    status = "🟡 TEST TRANSACTION"
                    status_color = "#f59e0b"
                    insight = "Small transaction testing system limits"
                    action = "Fraudster validating stolen credentials"
                else:
                    status = "🔴 EXPLOITATION"
                    status_color = "#ef4444"
                    insight = "Large fraudulent transaction after successful test"
                    action = "Actual fraud execution - stolen funds"

                hover_text = (
                    f"<b style='font-size:14px'>{time}</b><br><br>"
                    f"<b style='color:{status_color}'>{status}</b><br><br>"
                    f"<b>📊 Transaction Details:</b><br>"
                    f"• Amount: <b>${amount}</b><br>"
                    f"• Type: <b>{tx_type.upper()}</b><br>"
                    f"• Sequence: <b>#{idx+1}</b> of {len(viz_data['times'])}<br><br>"
                    f"<b>💡 Fraud Pattern:</b><br>"
                    f"{insight}<br><br>"
                    f"<b>🎯 Assessment:</b><br>"
                    f"{action}"
                )
                test_hover_texts.append(hover_text)

            fig_test.add_trace(go.Bar(
                x=viz_data['times'],
                y=viz_data['amounts'],
                marker=dict(color=colors),
                text=[f"${a}" for a in viz_data['amounts']],
                textposition='outside',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=test_hover_texts
            ))

            fig_test.update_layout(
                title="Testing Pattern: Small Tests → Large Exploitation",
                xaxis_title="Time",
                yaxis_title="Amount ($)",
                yaxis_type="log",
                height=400
            )

            st.plotly_chart(fig_test, use_container_width=True)
        
        # Scenario 4: Money Mule Flow
        if 'flow' in viz_data:
            st.markdown("#### Money Flow Diagram")
            
            flow = viz_data['flow']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📥 Incoming**")
                for idx, (source, amount) in enumerate(zip(flow['sources'], flow['incoming'])):
                    st.markdown(f"• {source}: ${amount}")
                st.markdown(f"**Total Incoming:** ${sum(flow['incoming'])}")
            
            with col2:
                st.markdown("**📤 Outgoing**")
                st.markdown(f"• {flow['destination']}: ${flow['outgoing'][0]}")
                st.markdown(f"**Total Outgoing:** ${sum(flow['outgoing'])}")
                retention = sum(flow['incoming']) - sum(flow['outgoing'])
                st.markdown(f"**Retained:** ${retention} ({retention/sum(flow['incoming'])*100:.1f}%)")
        
        # Scenario 5: Device Comparison
        if 'device_comparison' in viz_data:
            st.markdown("#### Device & Access Profile Comparison")
            
            comparison = viz_data['device_comparison']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Normal Profile**")
                st.markdown(f"📱 Device: {comparison['normal']['device']}")
                st.markdown(f"🌍 Location: {comparison['normal']['location']}")
                st.markdown(f"🔒 VPN: {comparison['normal']['vpn']}")
                st.markdown(f"⌨️ Typing Speed: {comparison['normal']['typing_wpm']} wpm")
            
            with col2:
                st.markdown("**⚠️ Suspicious Activity**")
                st.markdown(f"📱 Device: {comparison['suspicious']['device']} 🔴")
                st.markdown(f"🌍 Location: {comparison['suspicious']['location']} 🔴")
                st.markdown(f"🔓 VPN: {comparison['suspicious']['vpn']} 🔴")
                st.markdown(f"⌨️ Typing Speed: {comparison['suspicious']['typing_wpm']} wpm 🔴")
        
        # Scenario 13: Refund Chain
        if 'chain' in viz_data:
            st.markdown("#### Transaction Chain Visualization")
            
            chain_df = pd.DataFrame(viz_data['chain'])
            
            for idx, row in chain_df.iterrows():
                emoji = "📥" if row['type'] == 'payment' else "📤"
                st.markdown(f"{emoji} {row['from']} → {row['to']}: **${row['amount']}** ({row['type']})")

    st.markdown("---")

    # Decision Section
 

    with decision_col2:
        st.markdown("### Action Buttons")
        
        if st.button("🔴 REJECT TRANSACTION", type="primary"):
            st.error("Transaction rejected and account flagged for review")
        
        if st.button("🟡 ESCALATE TO MANAGER"):
            st.warning("Case escalated to senior analyst")
        
        if st.button("🟢 CLEAR TRANSACTION"):
            st.success("Transaction cleared - customer notified")
        
        if st.button("📞 CONTACT CUSTOMER"):
            st.info("Verification call initiated")

    st.markdown("---")

    # Summary Statistics
    st.markdown("### 📊 Scenario Summary Statistics")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric("Total Rules Triggered", len(scenario['triggered_rules']))
    with summary_col2:
        critical_rules = sum(1 for r in scenario['triggered_rules'] if r['severity'] == 'critical')
        st.metric("Critical Rules", critical_rules)
    with summary_col3:
        total_weight = sum(r['weight'] for r in scenario['triggered_rules'])
        st.metric("Total Risk Weight", total_weight)
    with summary_col4:
        st.metric("Detection Time", "Real-time" if scenario['risk_score'] >= 85 else "< 1 min")

    st.markdown("---")
    st.caption("💡 **Note:** All scenarios are based on real fraud patterns and detection methodologies used in financial institutions.")
    st.caption(f"**System Status:** 🟢 Active | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    render()
