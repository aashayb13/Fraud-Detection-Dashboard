# config/settings.py
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./transaction_monitoring.db")

# Risk scoring thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8
}

# Decision engine settings
AUTO_APPROVE_THRESHOLD = 0.3
MANUAL_REVIEW_THRESHOLD = 0.6
DEFAULT_AUTO_APPROVE_THRESHOLD = 0.3
DEFAULT_MANUAL_REVIEW_THRESHOLD = 0.6
DEFAULT_HIGH_VALUE_THRESHOLD = 10000.00

# Cost-benefit parameters
HOURLY_REVIEW_COST = 75.00  # Cost per hour for manual review
AVG_REVIEW_TIME_MINUTES = 15  # Average time to review a transaction

# Payroll fraud detection settings
PAYROLL_SUSPICIOUS_CHANGE_WINDOW_DAYS = 30  # Days before payroll to flag account changes
PAYROLL_RAPID_CHANGE_THRESHOLD = 2  # Number of changes that trigger suspicion
PAYROLL_RAPID_CHANGE_WINDOW_DAYS = 90  # Window to count rapid changes
PAYROLL_VERIFICATION_REQUIRED_THRESHOLD = 5000.00  # Payroll amount requiring verification

# Geographic fraud detection settings
GEOGRAPHIC_LOOKBACK_DAYS = 365  # Days to look back for vendor payment history
MIN_HISTORICAL_TRANSACTIONS = 3  # Minimum transactions needed to establish pattern
DOMESTIC_COUNTRY_CODE = "US"  # Default domestic country code

# Beneficiary fraud detection settings - Rapid Addition (compromised admin scenario)
BENEFICIARY_RAPID_ADDITION_THRESHOLD = 5  # Number of beneficiaries added to trigger alert
BENEFICIARY_RAPID_ADDITION_WINDOW_HOURS = 24  # Time window for rapid additions
BENEFICIARY_BULK_ADDITION_THRESHOLD = 10  # Threshold for bulk/scripted additions
BENEFICIARY_BULK_ADDITION_WINDOW_HOURS = 72  # Extended window for bulk detection
BENEFICIARY_RECENT_ADDITION_HOURS = 48  # Hours to consider beneficiary as "newly added"
BENEFICIARY_NEW_BENEFICIARY_PAYMENT_RATIO = 0.7  # Ratio of payments to new beneficiaries (70%+)

# Beneficiary fraud detection settings - Vendor Impersonation/BEC
BENEFICIARY_SAME_DAY_PAYMENT_HOURS = 24  # Hours since change to flag as same-day
BENEFICIARY_CRITICAL_CHANGE_WINDOW_DAYS = 7  # Days since change - critical risk window
BENEFICIARY_SUSPICIOUS_CHANGE_WINDOW_DAYS = 30  # Days since change - elevated risk window
BENEFICIARY_RAPID_CHANGE_THRESHOLD = 2  # Number of changes that trigger suspicion
BENEFICIARY_RAPID_CHANGE_WINDOW_DAYS = 60  # Window to count rapid changes
BENEFICIARY_HIGH_VALUE_THRESHOLD = 10000.00  # Payment amount requiring extra verification
BENEFICIARY_NEW_VENDOR_DAYS = 90  # Days since registration to consider "new"

# Odd hours transaction fraud detection settings
ODD_HOURS_START = 22  # 10 PM - start of odd hours window (24-hour format)
ODD_HOURS_END = 6  # 6 AM - end of odd hours window (24-hour format)
ODD_HOURS_LARGE_TRANSACTION_THRESHOLD = 5000.00  # Amount to flag as "large" transaction
ODD_HOURS_VERY_LARGE_THRESHOLD = 25000.00  # Amount to flag as "very large" transaction
ODD_HOURS_DEVIATION_FROM_PATTERN = 0.8  # Ratio threshold for unusual timing (80% of historical transactions during day)
ODD_HOURS_MIN_HISTORICAL_TRANSACTIONS = 5  # Minimum transactions needed to establish timing pattern
ODD_HOURS_LOOKBACK_DAYS = 90  # Days to analyze historical transaction patterns
