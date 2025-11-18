# app/models/database.py
from sqlalchemy import create_engine, Column, String, Float, ForeignKey, Integer, Text, Boolean, DateTime, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
from config.settings import DATABASE_URL

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class Account(Base):
    __tablename__ = "accounts"

    account_id = Column(String, primary_key=True, index=True)
    creation_date = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())
    risk_tier = Column(String, default="medium")
    status = Column(String, default="active")

    # Relationships
    transactions = relationship("Transaction", back_populates="account")
    employees = relationship("Employee", back_populates="account")
    account_changes = relationship("AccountChangeHistory", back_populates="account")
    
class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True, index=True)
    timestamp = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())
    account_id = Column(String, ForeignKey("accounts.account_id"))
    counterparty_id = Column(String, index=True)
    amount = Column(Float)
    direction = Column(String, default="credit")  # "credit" (incoming) or "debit" (outgoing)
    transaction_type = Column(String)  # "ACH", "WIRE", etc.
    description = Column(Text, nullable=True)
    tx_metadata = Column(Text, nullable=True)  # JSON string (renamed from 'metadata' to avoid SQLAlchemy conflict)

    # Relationships
    account = relationship("Account", back_populates="transactions")
    risk_assessment = relationship("RiskAssessment", back_populates="transaction", uselist=False)
    
class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    assessment_id = Column(String, primary_key=True, index=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"))
    risk_score = Column(Float)
    triggered_rules = Column(Text)  # JSON string of rule names
    decision = Column(String)  # "auto_approve", "manual_review"
    review_status = Column(String, default="pending")  # "pending", "approved", "rejected"
    review_notes = Column(Text, nullable=True)
    review_timestamp = Column(String)
    reviewer_id = Column(String, nullable=True)

    # Relationships
    transaction = relationship("Transaction", back_populates="risk_assessment")

class Employee(Base):
    __tablename__ = "employees"

    employee_id = Column(String, primary_key=True, index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"))
    name = Column(String)
    email = Column(String, index=True)
    department = Column(String, nullable=True)
    hire_date = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())
    employment_status = Column(String, default="active")  # "active", "terminated", "suspended"

    # Payroll information
    payroll_account_number = Column(String)  # Bank account for payroll
    payroll_routing_number = Column(String)
    payroll_bank_name = Column(String, nullable=True)
    payroll_frequency = Column(String, default="biweekly")  # "weekly", "biweekly", "monthly"
    last_payroll_date = Column(String, nullable=True)

    # Relationships
    account = relationship("Account", back_populates="employees")
    account_changes = relationship("AccountChangeHistory", back_populates="employee")

class AccountChangeHistory(Base):
    __tablename__ = "account_change_history"

    change_id = Column(String, primary_key=True, index=True)
    employee_id = Column(String, ForeignKey("employees.employee_id"), index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"))
    timestamp = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())

    # Change details
    change_type = Column(String)  # "account_number", "routing_number", "bank_name", "email", "phone"
    old_value = Column(String, nullable=True)
    new_value = Column(String)

    # Change metadata
    change_source = Column(String)  # "employee_portal", "hr_system", "phone_request", "email_request"
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    verification_method = Column(String, nullable=True)  # "2fa", "email_confirmation", "phone_call", "none"
    verified = Column(Boolean, default=False)
    verified_by = Column(String, nullable=True)  # HR staff ID
    verification_timestamp = Column(String, nullable=True)

    # Risk indicators
    flagged_as_suspicious = Column(Boolean, default=False)
    suspicious_reason = Column(Text, nullable=True)

    # Relationships
    employee = relationship("Employee", back_populates="account_changes")
    account = relationship("Account", back_populates="account_changes")

class Beneficiary(Base):
    """
    Tracks beneficiaries/payees that can receive payments from an account.
    Used to detect:
    1. Rapid beneficiary addition fraud patterns (compromised admin)
    2. Vendor impersonation/BEC attacks (changed banking details)
    """
    __tablename__ = "beneficiaries"

    beneficiary_id = Column(String, primary_key=True, index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"), index=True)
    counterparty_id = Column(String, index=True, nullable=True)  # External identifier for the beneficiary

    # Beneficiary details
    name = Column(String)
    beneficiary_type = Column(String)  # "supplier", "vendor", "contractor", "partner", "individual", "business", "payroll"
    email = Column(String, index=True, nullable=True)
    phone = Column(String, nullable=True)

    # Bank account information
    bank_account_number = Column(String)
    bank_routing_number = Column(String)
    bank_name = Column(String, nullable=True)
    bank_account_type = Column(String, default="checking")  # "checking", "savings"

    # Registration/Addition metadata
    registration_date = Column(String, default=lambda: datetime.datetime.utcnow().isoformat(), index=True)
    added_by = Column(String, nullable=True)  # User/admin ID who added the beneficiary
    addition_source = Column(String, nullable=True)  # "admin_portal", "api", "bulk_upload", "mobile_app", "ap_portal", "erp_system"
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)

    # Payment history
    last_payment_date = Column(String, nullable=True)
    total_payments_received = Column(Integer, default=0)
    total_amount_received = Column(Float, default=0.0)

    # Status
    status = Column(String, default="active")  # "active", "suspended", "inactive", "removed"

    # Verification status
    verified = Column(Boolean, default=False)
    verification_method = Column(String, nullable=True)  # "micro_deposit", "manual_review", "instant_verification", "callback", "email_confirmation"
    verification_date = Column(String, nullable=True)
    risk_level = Column(String, default="medium")  # "low", "medium", "high"

    # Risk indicators
    flagged_as_suspicious = Column(Boolean, default=False)
    suspicious_reason = Column(Text, nullable=True)

    # Relationships
    account = relationship("Account")
    change_history = relationship("BeneficiaryChangeHistory", back_populates="beneficiary")

class BeneficiaryChangeHistory(Base):
    __tablename__ = "beneficiary_change_history"

    change_id = Column(String, primary_key=True, index=True)
    beneficiary_id = Column(String, ForeignKey("beneficiaries.beneficiary_id"), index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"))
    timestamp = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())

    # Change details
    change_type = Column(String)  # "account_number", "routing_number", "bank_name", "email", "phone", "address"
    old_value = Column(String, nullable=True)
    new_value = Column(String)

    # Change metadata
    change_source = Column(String)  # "ap_portal", "erp_system", "phone_request", "email_request", "fax", "manual_entry"
    requestor_name = Column(String, nullable=True)  # Person who requested the change
    requestor_email = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)

    # Verification details
    verification_method = Column(String, nullable=True)  # "callback", "email_confirmation", "portal_verification", "in_person", "none"
    verified = Column(Boolean, default=False)
    verified_by = Column(String, nullable=True)  # AP staff ID
    verification_timestamp = Column(String, nullable=True)
    verification_notes = Column(Text, nullable=True)

    # Risk indicators
    flagged_as_suspicious = Column(Boolean, default=False)
    suspicious_reason = Column(Text, nullable=True)

    # Relationships
    beneficiary = relationship("Beneficiary", back_populates="change_history")
    account = relationship("Account")

class Blacklist(Base):
    """
    Tracks blacklisted entities (accounts, merchants, UPI IDs) to prevent fraudulent transactions.
    Used to detect and block transactions to/from known fraudulent or high-risk entities.
    """
    __tablename__ = "blacklist"

    blacklist_id = Column(String, primary_key=True, index=True)

    # Entity identifiers (at least one must be provided)
    entity_type = Column(String, index=True)  # "account", "merchant", "upi", "routing_number", "email", "phone"
    entity_value = Column(String, index=True)  # The actual ID/value being blacklisted

    # Entity details
    entity_name = Column(String, nullable=True)  # Name of the blacklisted entity
    counterparty_id = Column(String, index=True, nullable=True)  # If it's a counterparty

    # Blacklist metadata
    reason = Column(Text)  # Reason for blacklisting
    severity = Column(String, default="high")  # "low", "medium", "high", "critical"
    added_date = Column(String, default=lambda: datetime.datetime.utcnow().isoformat(), index=True)
    added_by = Column(String, nullable=True)  # Admin/system ID that added the entry
    source = Column(String, nullable=True)  # "internal_investigation", "law_enforcement", "industry_report", "user_report"

    # Status
    status = Column(String, default="active", index=True)  # "active", "removed", "expired"
    expiry_date = Column(String, nullable=True)  # Optional expiry for temporary blacklists
    removed_date = Column(String, nullable=True)
    removed_by = Column(String, nullable=True)
    removal_reason = Column(Text, nullable=True)

    # Additional context
    related_case_id = Column(String, nullable=True)  # Link to investigation case
    notes = Column(Text, nullable=True)
    external_reference = Column(String, nullable=True)  # Reference to external blacklist systems

class DeviceSession(Base):
    """
    Tracks device fingerprints and session information for fraud detection.
    Used to detect account takeover through device/location mismatches.
    """
    __tablename__ = "device_sessions"

    session_id = Column(String, primary_key=True, index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"), index=True)
    timestamp = Column(String, default=lambda: datetime.datetime.utcnow().isoformat(), index=True)

    # Device fingerprinting
    device_id = Column(String, index=True, nullable=True)  # Unique device identifier/fingerprint
    device_name = Column(String, nullable=True)  # User-friendly device name
    device_type = Column(String, nullable=True)  # "mobile", "desktop", "tablet"

    # Browser information
    browser = Column(String, nullable=True)  # "Chrome", "Firefox", "Safari", etc.
    browser_version = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)  # Full user agent string

    # Operating system
    os = Column(String, nullable=True)  # "Windows", "macOS", "iOS", "Android", etc.
    os_version = Column(String, nullable=True)

    # Network information
    ip_address = Column(String, index=True, nullable=True)
    ip_country = Column(String, nullable=True)
    ip_city = Column(String, nullable=True)
    isp = Column(String, nullable=True)

    # Session context
    session_type = Column(String, default="transaction")  # "login", "transaction", "account_change"
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"), nullable=True, index=True)

    # Trust indicators
    is_trusted_device = Column(Boolean, default=False)
    first_seen = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())
    last_seen = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())
    session_count = Column(Integer, default=1)  # How many times this device has been used

    # Risk flags
    flagged_as_suspicious = Column(Boolean, default=False)
    suspicious_reason = Column(Text, nullable=True)

    # Relationships
    account = relationship("Account")
    transaction = relationship("Transaction")

class VPNProxyIP(Base):
    """
    Tracks known VPN and proxy IP addresses/ranges for fraud detection.
    Used to flag transactions from masked IP addresses.
    """
    __tablename__ = "vpn_proxy_ips"

    entry_id = Column(String, primary_key=True, index=True)

    # IP information
    ip_address = Column(String, index=True, nullable=True)  # Specific IP address
    ip_range_start = Column(String, nullable=True)  # Start of IP range (CIDR notation or start IP)
    ip_range_end = Column(String, nullable=True)  # End of IP range
    subnet = Column(String, index=True, nullable=True)  # CIDR subnet (e.g., "192.168.1.0/24")

    # VPN/Proxy details
    service_type = Column(String, index=True)  # "vpn", "proxy", "tor", "datacenter", "hosting"
    service_name = Column(String, nullable=True)  # Name of VPN/proxy service (e.g., "NordVPN", "Tor Exit Node")
    provider = Column(String, nullable=True)  # Provider/company name

    # Location information
    country = Column(String, nullable=True)
    city = Column(String, nullable=True)
    datacenter = Column(String, nullable=True)  # Datacenter/hosting provider

    # Risk assessment
    risk_level = Column(String, default="medium")  # "low", "medium", "high", "critical"
    is_residential_proxy = Column(Boolean, default=False)  # Residential proxies harder to detect
    is_mobile_proxy = Column(Boolean, default=False)

    # Metadata
    added_date = Column(String, default=lambda: datetime.datetime.utcnow().isoformat(), index=True)
    added_by = Column(String, nullable=True)  # Admin/system that added this entry
    source = Column(String, nullable=True)  # "manual", "api", "threat_intelligence", "ip2proxy", "maxmind"
    last_verified = Column(String, nullable=True)  # When this was last verified as VPN/proxy

    # Status
    status = Column(String, default="active", index=True)  # "active", "removed", "expired"
    confidence = Column(Float, default=1.0)  # Confidence score (0.0 to 1.0)

    # Additional context
    notes = Column(Text, nullable=True)
    external_reference = Column(String, nullable=True)

class HighRiskLocation(Base):
    """
    Tracks high-risk countries, cities, and regions for fraud detection.
    Used to flag transactions from unusual or high-risk geolocations.
    """
    __tablename__ = "high_risk_locations"

    location_id = Column(String, primary_key=True, index=True)

    # Location identifiers
    country_code = Column(String, index=True, nullable=True)  # ISO 2-letter code (e.g., "NG", "RU")
    country_name = Column(String, nullable=True)
    region = Column(String, nullable=True)  # State/province
    city = Column(String, index=True, nullable=True)

    # Continent/geographic grouping
    continent = Column(String, nullable=True)  # "Africa", "Asia", "Europe", etc.

    # Risk assessment
    risk_level = Column(String, default="medium", index=True)  # "low", "medium", "high", "critical"
    risk_category = Column(String, nullable=True)  # "fraud_hotspot", "sanctioned", "high_crime", "data_breach_origin"
    risk_score = Column(Float, default=0.5)  # 0.0 (safe) to 1.0 (very risky)

    # Risk reasons
    fraud_rate = Column(Float, nullable=True)  # Percentage of fraud from this location
    reason = Column(Text, nullable=True)  # Why this location is flagged

    # Specific risk types
    is_sanctioned = Column(Boolean, default=False)  # Under economic sanctions
    is_embargoed = Column(Boolean, default=False)  # Trade embargo
    has_high_fraud_rate = Column(Boolean, default=False)
    has_high_cybercrime_rate = Column(Boolean, default=False)
    is_tax_haven = Column(Boolean, default=False)
    lacks_kyc_regulations = Column(Boolean, default=False)  # Weak KYC/AML regulations

    # Metadata
    added_date = Column(String, default=lambda: datetime.datetime.utcnow().isoformat(), index=True)
    added_by = Column(String, nullable=True)
    source = Column(String, nullable=True)  # "ofac", "fincen", "interpol", "manual", "threat_intelligence"
    last_updated = Column(String, nullable=True)

    # Status
    status = Column(String, default="active", index=True)  # "active", "removed", "under_review"

    # Action recommendations
    requires_manual_review = Column(Boolean, default=False)
    requires_enhanced_verification = Column(Boolean, default=False)
    block_by_default = Column(Boolean, default=False)

    # Additional context
    notes = Column(Text, nullable=True)
    external_reference = Column(String, nullable=True)

class BehavioralBiometric(Base):
    """
    Tracks user behavioral biometrics for fraud detection.
    Monitors interaction patterns like typing speed, mouse movement, click patterns.
    Used to detect account takeover through behavioral deviations.
    """
    __tablename__ = "behavioral_biometrics"

    biometric_id = Column(String, primary_key=True, index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"), index=True)
    session_id = Column(String, ForeignKey("device_sessions.session_id"), nullable=True, index=True)
    timestamp = Column(String, default=lambda: datetime.datetime.utcnow().isoformat(), index=True)

    # Session context
    session_type = Column(String, default="transaction")  # "login", "transaction", "form_fill"
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"), nullable=True, index=True)

    # Typing behavior metrics
    avg_typing_speed_wpm = Column(Float, nullable=True)  # Words per minute
    avg_key_hold_time_ms = Column(Float, nullable=True)  # Average time key is held down
    avg_key_interval_ms = Column(Float, nullable=True)  # Time between key presses
    typing_rhythm_variance = Column(Float, nullable=True)  # Variance in typing rhythm
    backspace_frequency = Column(Float, nullable=True)  # Backspace usage rate
    typing_errors_count = Column(Integer, default=0)  # Number of typing errors

    # Mouse/pointer behavior
    avg_mouse_speed_px_sec = Column(Float, nullable=True)  # Average mouse movement speed
    mouse_movement_smoothness = Column(Float, nullable=True)  # How smooth/jerky movements are (0-1)
    click_accuracy = Column(Float, nullable=True)  # How accurately user clicks targets (0-1)
    double_click_speed_ms = Column(Float, nullable=True)  # Average double-click timing
    mouse_idle_time_sec = Column(Float, nullable=True)  # Time mouse is idle
    mouse_distance_traveled_px = Column(Float, nullable=True)  # Total mouse movement

    # Touch behavior (mobile)
    avg_touch_pressure = Column(Float, nullable=True)  # Touch pressure (0-1)
    avg_touch_size = Column(Float, nullable=True)  # Touch contact size
    swipe_speed = Column(Float, nullable=True)  # Average swipe speed
    pinch_zoom_frequency = Column(Float, nullable=True)  # How often user pinches to zoom

    # Interaction timing patterns
    session_duration_sec = Column(Float, nullable=True)  # How long the session lasted
    time_to_first_action_sec = Column(Float, nullable=True)  # Time from load to first action
    action_count = Column(Integer, default=0)  # Number of actions in session
    actions_per_minute = Column(Float, nullable=True)  # Activity rate

    # Navigation patterns
    page_sequence = Column(Text, nullable=True)  # JSON array of pages visited
    navigation_method = Column(String, nullable=True)  # "keyboard", "mouse", "touch", "mixed"
    uses_shortcuts = Column(Boolean, default=False)  # Uses keyboard shortcuts
    uses_autofill = Column(Boolean, default=False)  # Uses browser autofill

    # Copy-paste behavior
    paste_frequency = Column(Float, nullable=True)  # How often user pastes
    paste_fields = Column(Text, nullable=True)  # JSON array of fields with pasted content

    # Device orientation (mobile)
    device_orientation = Column(String, nullable=True)  # "portrait", "landscape"
    orientation_changes = Column(Integer, default=0)  # Number of orientation changes

    # Behavioral profile metadata
    is_baseline = Column(Boolean, default=False)  # Is this a baseline/normal behavior
    confidence_score = Column(Float, default=0.5)  # Confidence in measurement (0.0-1.0)
    sample_size = Column(Integer, default=1)  # Number of interactions sampled

    # Anomaly detection
    is_anomalous = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=True)  # How anomalous (0.0-1.0)
    anomaly_reasons = Column(Text, nullable=True)  # JSON array of reasons

    # Relationships
    account = relationship("Account")
    session = relationship("DeviceSession")
    transaction = relationship("Transaction")

class FraudFlag(Base):
    """
    Model for tracking fraudulent behavior flags and fraud history.

    Stores historical fraud incidents, flags, and dispositions for accounts
    and beneficiaries to enable repeat fraud detection and risk assessment.
    """
    __tablename__ = "fraud_flags"

    id = Column(Integer, primary_key=True, index=True)

    # Entity information
    entity_type = Column(String(50), nullable=False, index=True)  # "account" or "beneficiary"
    entity_id = Column(String(255), nullable=False, index=True)  # account_id or beneficiary_id

    # Fraud details
    fraud_type = Column(String(100), nullable=False, index=True)  # Type of fraud detected
    fraud_category = Column(String(50), nullable=False, index=True)  # Category classification
    severity = Column(String(20), nullable=False)  # "low", "medium", "high", "critical"

    # Incident details
    incident_date = Column(DateTime, nullable=False, index=True)
    detection_date = Column(DateTime, nullable=False)
    reported_date = Column(DateTime, nullable=True)

    # Status and disposition
    status = Column(String(50), nullable=False, index=True)  # "active", "resolved", "disputed", "confirmed"
    disposition = Column(String(50), nullable=True)  # "confirmed_fraud", "false_positive", "pending_investigation"

    # Associated data
    related_transaction_id = Column(String(255), nullable=True, index=True)
    amount_involved = Column(Numeric(precision=15, scale=2), nullable=True)

    # Investigation details
    investigator_id = Column(String(255), nullable=True)
    investigation_notes = Column(Text, nullable=True)

    # Resolution
    resolution_date = Column(DateTime, nullable=True)
    resolution_action = Column(String(100), nullable=True)  # "account_closed", "funds_recovered", "warning_issued", etc.

    # Additional metadata
    risk_score_at_time = Column(Integer, nullable=True)  # Risk score when fraud occurred
    additional_data = Column(Text, nullable=True)  # JSON for additional context

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    account = relationship("Account", foreign_keys=[entity_id],
                          primaryjoin="and_(FraudFlag.entity_id==Account.account_id, FraudFlag.entity_type=='account')",
                          overlaps="fraud_flags")
    beneficiary = relationship("Beneficiary", foreign_keys=[entity_id],
                              primaryjoin="and_(FraudFlag.entity_id==Beneficiary.beneficiary_id, FraudFlag.entity_type=='beneficiary')",
                              overlaps="fraud_flags,account")

class FraudComplaint(Base):
    """
    Model for tracking fraud complaints and reports.

    Stores complaints linked to UPI IDs, devices, accounts, and beneficiaries
    to enable repeat offender detection and fraud network analysis.
    """
    __tablename__ = "fraud_complaints"

    id = Column(Integer, primary_key=True, index=True)

    # Complaint identifier
    complaint_id = Column(String(255), unique=True, nullable=False, index=True)

    # Entity information (what the complaint is about)
    entity_type = Column(String(50), nullable=False, index=True)  # "account", "beneficiary", "device", "upi_id", "transaction"
    entity_id = Column(String(255), nullable=False, index=True)  # Corresponding ID

    # Complainant information
    complainant_account_id = Column(String(255), nullable=True, index=True)
    complainant_name = Column(String(255), nullable=True)
    complainant_contact = Column(String(100), nullable=True)

    # Complaint details
    complaint_type = Column(String(100), nullable=False, index=True)  # "unauthorized_transaction", "account_takeover", "phishing", "scam", etc.
    complaint_category = Column(String(50), nullable=False, index=True)  # "financial_fraud", "identity_theft", "scam", "technical_fraud"
    severity = Column(String(20), nullable=False, index=True)  # "low", "medium", "high", "critical"

    # Description and evidence
    description = Column(Text, nullable=True)
    evidence_provided = Column(Boolean, default=False)
    evidence_details = Column(Text, nullable=True)  # JSON for file references, screenshots, etc.

    # Related transaction information
    related_transaction_id = Column(String(255), nullable=True, index=True)
    amount_involved = Column(Numeric(precision=15, scale=2), nullable=True)
    transaction_date = Column(DateTime, nullable=True)

    # Device and UPI information
    device_id = Column(String(255), nullable=True, index=True)
    device_fingerprint = Column(Text, nullable=True)
    upi_id = Column(String(255), nullable=True, index=True)

    # Status and resolution
    status = Column(String(50), nullable=False, index=True)  # "submitted", "under_review", "investigating", "resolved", "closed", "rejected"
    resolution = Column(String(100), nullable=True)  # "confirmed_fraud", "false_positive", "no_evidence", "resolved_with_refund"
    resolution_notes = Column(Text, nullable=True)

    # Investigation details
    assigned_investigator = Column(String(255), nullable=True)
    investigation_priority = Column(String(20), nullable=True)  # "low", "medium", "high", "urgent"

    # Action taken
    action_taken = Column(String(100), nullable=True)  # "account_suspended", "funds_frozen", "refund_issued", "no_action"
    refund_amount = Column(Numeric(precision=15, scale=2), nullable=True)

    # Timestamps
    complaint_date = Column(DateTime, nullable=False, index=True)
    review_date = Column(DateTime, nullable=True)
    resolution_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Fraud network analysis
    is_part_of_fraud_ring = Column(Boolean, default=False)
    fraud_ring_id = Column(String(255), nullable=True, index=True)
    related_complaints = Column(Text, nullable=True)  # JSON array of related complaint IDs

    # Verification
    verified = Column(Boolean, default=False)
    verification_date = Column(DateTime, nullable=True)
    verification_method = Column(String(100), nullable=True)

    # Additional metadata
    source = Column(String(100), nullable=True)  # "customer_service", "automated_system", "bank_report", "user_report"
    channel = Column(String(50), nullable=True)  # "app", "web", "phone", "email", "branch"
    additional_data = Column(Text, nullable=True)  # JSON for extra fields

class MerchantProfile(Base):
    """Model for tracking merchant profiles and their registered business categories"""
    __tablename__ = "merchant_profiles"

    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(String(255), unique=True, nullable=False, index=True)
    merchant_name = Column(String(255), nullable=False)

    # Merchant Category Code (MCC) - Primary business category
    registered_mcc = Column(String(10), nullable=False, index=True)  # 4-digit MCC code
    registered_category = Column(String(100), nullable=False, index=True)  # Human-readable category name

    # Business type and industry
    business_type = Column(String(100), nullable=True)  # "retail", "restaurant", "online", "service", "wholesale", "healthcare"
    industry = Column(String(100), nullable=True)  # "food_beverage", "clothing", "electronics", "travel", "entertainment"
    sub_industry = Column(String(100), nullable=True)

    # Registration information
    registration_date = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    registration_country = Column(String(10), nullable=True)  # ISO country code
    business_license_number = Column(String(255), nullable=True)
    tax_id = Column(String(100), nullable=True)

    # Status
    status = Column(String(50), nullable=False, default="active", index=True)  # "active", "suspended", "terminated", "under_review"
    risk_level = Column(String(20), nullable=False, default="low")  # "low", "medium", "high", "critical"

    # Transaction patterns
    avg_transaction_amount = Column(Float, nullable=True)
    max_transaction_amount = Column(Float, nullable=True)
    total_transactions = Column(Integer, default=0)
    total_volume = Column(Float, default=0.0)

    # Category compliance tracking
    category_mismatch_count = Column(Integer, default=0)  # Number of out-of-category transactions
    category_mismatch_rate = Column(Float, default=0.0)  # Percentage of mismatched transactions
    last_mismatch_date = Column(DateTime, nullable=True)

    # Historical MCCs - Track if merchant has changed categories
    previous_mccs = Column(Text, nullable=True)  # JSON array of previous MCC codes with dates
    mcc_change_count = Column(Integer, default=0)
    last_mcc_change_date = Column(DateTime, nullable=True)

    # Fraud indicators
    is_high_risk_merchant = Column(Boolean, default=False)
    is_flagged_for_fraud = Column(Boolean, default=False)
    fraud_flag_reason = Column(Text, nullable=True)
    fraud_flag_date = Column(DateTime, nullable=True)

    # Verification
    verified = Column(Boolean, default=False)
    verification_date = Column(DateTime, nullable=True)
    verification_method = Column(String(100), nullable=True)  # "manual_review", "document_verification", "third_party_check"

    # Contact information
    contact_email = Column(String(255), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    business_address = Column(Text, nullable=True)

    # Allowed secondary categories (some merchants legitimately operate in multiple categories)
    allowed_secondary_mccs = Column(Text, nullable=True)  # JSON array of allowed MCC codes

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    last_transaction_date = Column(DateTime, nullable=True, index=True)

    # Additional metadata
    notes = Column(Text, nullable=True)
    additional_data = Column(Text, nullable=True)  # JSON for extra fields

class AccountLimit(Base):
    """Model for tracking account transaction limits and thresholds"""
    __tablename__ = "account_limits"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(255), nullable=False, index=True)

    # Limit types and values
    # Daily limits
    daily_transaction_count_limit = Column(Integer, nullable=True)  # Max number of transactions per day
    daily_transaction_amount_limit = Column(Float, nullable=True)  # Max total amount per day
    daily_debit_limit = Column(Float, nullable=True)  # Max total debit amount per day
    daily_credit_limit = Column(Float, nullable=True)  # Max total credit amount per day

    # Single transaction limits
    single_transaction_limit = Column(Float, nullable=True)  # Max amount for a single transaction
    single_debit_limit = Column(Float, nullable=True)  # Max amount for a single debit
    single_credit_limit = Column(Float, nullable=True)  # Max amount for a single credit

    # Weekly/Monthly limits (optional)
    weekly_transaction_amount_limit = Column(Float, nullable=True)
    monthly_transaction_amount_limit = Column(Float, nullable=True)

    # Per-transaction-type limits
    ach_daily_limit = Column(Float, nullable=True)
    wire_daily_limit = Column(Float, nullable=True)
    card_daily_limit = Column(Float, nullable=True)
    upi_daily_limit = Column(Float, nullable=True)

    # Per-counterparty limits
    per_beneficiary_daily_limit = Column(Float, nullable=True)  # Max amount to single beneficiary per day
    per_beneficiary_transaction_limit = Column(Float, nullable=True)  # Max amount per transaction to beneficiary

    # Limit status
    status = Column(String(50), nullable=False, default="active", index=True)  # "active", "suspended", "override"
    is_custom_limit = Column(Boolean, default=False)  # Custom vs default limit
    risk_based_adjustment = Column(Float, default=1.0)  # Multiplier for risk-based limit adjustment (0.5 = 50% of normal)

    # Effective dates
    effective_date = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    expiration_date = Column(DateTime, nullable=True, index=True)

    # Limit override tracking
    override_enabled = Column(Boolean, default=False)
    override_reason = Column(Text, nullable=True)
    override_approved_by = Column(String(255), nullable=True)
    override_expiration = Column(DateTime, nullable=True)

    # Violation tracking
    total_violations = Column(Integer, default=0)
    last_violation_date = Column(DateTime, nullable=True, index=True)
    consecutive_violations = Column(Integer, default=0)

    # Limit modification history
    previous_limit_value = Column(Text, nullable=True)  # JSON of previous limit values
    limit_change_count = Column(Integer, default=0)
    last_limit_change_date = Column(DateTime, nullable=True)
    last_limit_change_reason = Column(Text, nullable=True)

    # Compliance and regulatory
    regulatory_limit = Column(Boolean, default=False)  # True if limit is regulatory requirement
    compliance_notes = Column(Text, nullable=True)

    # Usage tracking (for current period)
    current_daily_count = Column(Integer, default=0)
    current_daily_amount = Column(Float, default=0.0)
    current_period_reset_time = Column(DateTime, nullable=True)  # When current period counters reset

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Additional metadata
    notes = Column(Text, nullable=True)
    additional_data = Column(Text, nullable=True)  # JSON for extra fields

# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()