"""
FastAPI Application for Transaction Monitoring Dashboard

Provides REST API endpoints for the Streamlit dashboard to consume.
Leverages existing DashboardData and TransactionMonitor classes.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.database import get_db
from dashboard.main import DashboardData
from run import TransactionMonitor
from api.auth import authenticate_user, create_access_token, decode_token, Token, ACCESS_TOKEN_EXPIRE_MINUTES
from api.fraud_modules_catalog import FRAUD_MODULES_CATALOG, get_module_by_category, get_module_count, get_module_by_severity

# Initialize FastAPI
app = FastAPI(
    title="Transaction Monitoring API",
    description="REST API for fraud detection dashboard",
    version="1.0.0"
)

# CORS middleware (allow Streamlit to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# ==================== Response Models ====================

class OverviewStatsResponse(BaseModel):
    time_window_hours: int
    total_transactions: int
    total_value: float
    auto_approved: int
    manual_review: int
    blocked: int
    average_risk_score: float
    review_rate: float

class AlertItem(BaseModel):
    assessment_id: str
    transaction_id: str
    amount: float
    transaction_type: str
    risk_score: float
    triggered_rules: List[str]
    timestamp: str

class RuleStats(BaseModel):
    name: str
    description: str
    count: int
    weight: float

class AccountChangeItem(BaseModel):
    change_id: str
    employee_id: str
    change_type: str
    change_source: str
    verified: bool
    flagged: bool
    timestamp: str

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    database: str

# ==================== Authentication ====================

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify JWT authentication token.

    Returns user information if token is valid.
    """
    token = credentials.credentials
    token_data = decode_token(token)

    if not token_data or not token_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": token_data.user_id,
        "role": token_data.role
    }

# ==================== Endpoints ====================

@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected"
    }

@app.post("/api/v1/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint - authenticates user and returns JWT token.

    Test credentials:
    - Username: analyst, Password: analyst123 (Role: analyst)
    - Username: manager, Password: manager123 (Role: manager)
    - Username: investigator, Password: investigator123 (Role: investigator)
    - Username: admin, Password: admin123 (Role: admin)
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT token
    access_token = create_access_token(
        data={"sub": user.user_id, "role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "user_id": user.user_id
    }

@app.get("/api/v1/overview", response_model=OverviewStatsResponse)
async def get_overview_stats(
    time_window_hours: int = 24,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get overview statistics for the dashboard.

    Args:
        time_window_hours: Time window for statistics (default 24 hours)

    Returns:
        Overview statistics including transaction counts, risk scores, etc.
    """
    dashboard = DashboardData(db)
    stats = dashboard.get_overview_stats(time_window_hours)
    return stats

@app.get("/api/v1/alerts/live", response_model=List[AlertItem])
async def get_live_alerts(
    limit: int = 100,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get live fraud alerts (manual review queue).

    Args:
        limit: Maximum number of alerts to return

    Returns:
        List of pending fraud alerts sorted by risk score (descending)
    """
    dashboard = DashboardData(db)
    queue = dashboard.get_manual_review_queue()
    return queue[:limit]

@app.get("/api/v1/rules/top", response_model=List[RuleStats])
async def get_top_triggered_rules(
    limit: int = 10,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get most frequently triggered fraud detection rules.

    Args:
        limit: Number of top rules to return

    Returns:
        List of rules sorted by trigger count
    """
    dashboard = DashboardData(db)
    rules = dashboard.get_top_triggered_rules(limit)
    return rules

@app.get("/api/v1/scenarios/breakdown")
async def get_scenario_breakdown(
    time_window_hours: int = 24,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get fraud activity breakdown by scenario type.

    Args:
        time_window_hours: Time window for analysis

    Returns:
        Breakdown of fraud activity by scenario (payroll, beneficiary, etc.)
    """
    dashboard = DashboardData(db)
    scenarios = dashboard.get_scenario_breakdown(time_window_hours)
    return scenarios

@app.get("/api/v1/account-changes/recent", response_model=List[AccountChangeItem])
async def get_recent_account_changes(
    limit: int = 20,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get recent account changes (for payroll fraud monitoring).

    Args:
        limit: Number of changes to return

    Returns:
        List of recent account changes
    """
    dashboard = DashboardData(db)
    changes = dashboard.get_recent_account_changes(limit)
    return changes

@app.get("/api/v1/transaction/{transaction_id}")
async def get_transaction_details(
    transaction_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get detailed information about a specific transaction.

    Args:
        transaction_id: Transaction ID to lookup

    Returns:
        Detailed transaction and risk assessment information
    """
    from app.models.database import Transaction, RiskAssessment
    import json

    # Get transaction
    tx = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Get risk assessment
    assessment = db.query(RiskAssessment).filter(
        RiskAssessment.transaction_id == transaction_id
    ).first()

    result = {
        "transaction_id": tx.transaction_id,
        "account_id": tx.account_id,
        "amount": tx.amount,
        "direction": tx.direction,
        "transaction_type": tx.transaction_type,
        "description": tx.description,
        "timestamp": tx.timestamp,
        "counterparty_id": tx.counterparty_id
    }

    if assessment:
        triggered_rules = json.loads(assessment.triggered_rules) if assessment.triggered_rules else {}
        result["risk_assessment"] = {
            "assessment_id": assessment.assessment_id,
            "risk_score": assessment.risk_score,
            "decision": assessment.decision,
            "review_status": assessment.review_status,
            "triggered_rules": triggered_rules,
            "review_notes": assessment.review_notes
        }

    return result

@app.post("/api/v1/alert/{assessment_id}/action")
async def update_alert_status(
    assessment_id: str,
    action: str,
    notes: Optional[str] = None,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Update alert status (approve, reject, escalate).

    Args:
        assessment_id: Risk assessment ID
        action: Action to take (approved, rejected, escalated)
        notes: Optional review notes

    Returns:
        Updated assessment status
    """
    from app.models.database import RiskAssessment

    # Get assessment
    assessment = db.query(RiskAssessment).filter(
        RiskAssessment.assessment_id == assessment_id
    ).first()

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    # Update status
    if action == "approved":
        assessment.review_status = "approved"
    elif action == "rejected":
        assessment.review_status = "rejected"
    elif action == "escalated":
        assessment.review_status = "escalated"
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

    assessment.review_notes = notes
    assessment.reviewer_id = user["user_id"]
    assessment.review_timestamp = datetime.utcnow().isoformat()

    db.commit()

    return {
        "assessment_id": assessment_id,
        "status": assessment.review_status,
        "reviewer": user["user_id"]
    }

@app.get("/api/v1/metrics/time-series")
async def get_time_series_metrics(
    time_range: str = "24h",
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get time-series metrics for trend analysis.

    Args:
        time_range: Time range (1h, 24h, 7d, 30d)

    Returns:
        Time-series data for charts
    """
    from app.models.database import RiskAssessment, Transaction

    # Parse time range
    time_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = time_map.get(time_range, 24)

    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    # Get assessments
    assessments = db.query(RiskAssessment).filter(
        RiskAssessment.review_timestamp > cutoff
    ).all()

    # Group by hour
    hourly_data = {}
    for assessment in assessments:
        hour = assessment.review_timestamp[:13]  # YYYY-MM-DDTHH
        if hour not in hourly_data:
            hourly_data[hour] = {
                "timestamp": hour,
                "count": 0,
                "avg_risk": 0,
                "total_risk": 0,
                "high_risk_count": 0,
                "manual_review_count": 0,
                "auto_approve_count": 0
            }

        hourly_data[hour]["count"] += 1
        hourly_data[hour]["total_risk"] += assessment.risk_score
        if assessment.risk_score > 0.6:
            hourly_data[hour]["high_risk_count"] += 1

        if assessment.decision == "manual_review":
            hourly_data[hour]["manual_review_count"] += 1
        elif assessment.decision == "auto_approve":
            hourly_data[hour]["auto_approve_count"] += 1

    # Calculate averages
    for data in hourly_data.values():
        if data["count"] > 0:
            data["avg_risk"] = data["total_risk"] / data["count"]

    return {
        "time_range": time_range,
        "data": sorted(hourly_data.values(), key=lambda x: x["timestamp"])
    }

@app.get("/api/v1/analytics/risk-distribution")
async def get_risk_distribution(
    time_range: str = "24h",
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get risk score distribution for histogram.

    Returns:
        Risk score bins and counts
    """
    from app.models.database import RiskAssessment

    time_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = time_map.get(time_range, 24)
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    assessments = db.query(RiskAssessment).filter(
        RiskAssessment.review_timestamp > cutoff
    ).all()

    risk_scores = [a.risk_score for a in assessments]

    return {
        "risk_scores": risk_scores,
        "total_count": len(risk_scores),
        "avg_risk": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
        "max_risk": max(risk_scores) if risk_scores else 0,
        "min_risk": min(risk_scores) if risk_scores else 0
    }

@app.get("/api/v1/analytics/money-saved")
async def get_money_saved(
    time_range: str = "24h",
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Calculate money saved by blocking fraudulent transactions.

    Returns:
        Amount saved, blocked count, etc.
    """
    from app.models.database import RiskAssessment, Transaction
    import json

    time_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = time_map.get(time_range, 24)
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    # Get high-risk transactions that were flagged
    assessments = db.query(RiskAssessment).filter(
        RiskAssessment.review_timestamp > cutoff,
        RiskAssessment.risk_score >= 0.6  # High risk threshold
    ).all()

    total_saved = 0
    blocked_count = 0
    prevented_fraud_count = 0

    for assessment in assessments:
        tx = db.query(Transaction).filter(
            Transaction.transaction_id == assessment.transaction_id
        ).first()

        if tx:
            # Count as prevented if manual review or high risk
            if assessment.decision == "manual_review" or assessment.review_status == "rejected":
                total_saved += tx.amount
                prevented_fraud_count += 1

            if assessment.review_status == "rejected":
                blocked_count += 1

    return {
        "total_amount_saved": total_saved,
        "blocked_transaction_count": blocked_count,
        "prevented_fraud_count": prevented_fraud_count,
        "high_risk_flagged": len(assessments),
        "time_range": time_range
    }

@app.get("/api/v1/analytics/module-performance")
async def get_module_performance(
    time_range: str = "24h",
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get performance metrics for each fraud detection module.

    Returns:
        Module statistics including trigger rates, accuracy, etc.
    """
    from app.models.database import RiskAssessment
    import json

    time_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = time_map.get(time_range, 24)
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    assessments = db.query(RiskAssessment).filter(
        RiskAssessment.review_timestamp > cutoff
    ).all()

    module_stats = {}

    for assessment in assessments:
        triggered = json.loads(assessment.triggered_rules) if assessment.triggered_rules else {}

        for rule_name, rule_info in triggered.items():
            if rule_name not in module_stats:
                module_stats[rule_name] = {
                    "name": rule_name,
                    "description": rule_info.get("description", ""),
                    "trigger_count": 0,
                    "total_weight": 0,
                    "avg_weight": 0,
                    "high_risk_triggers": 0,
                    "confirmed_fraud": 0
                }

            module_stats[rule_name]["trigger_count"] += 1
            module_stats[rule_name]["total_weight"] += rule_info.get("weight", 0)

            if assessment.risk_score >= 0.6:
                module_stats[rule_name]["high_risk_triggers"] += 1

            if assessment.review_status == "rejected":
                module_stats[rule_name]["confirmed_fraud"] += 1

    # Calculate averages
    for stats in module_stats.values():
        if stats["trigger_count"] > 0:
            stats["avg_weight"] = stats["total_weight"] / stats["trigger_count"]
            stats["precision"] = stats["confirmed_fraud"] / stats["trigger_count"] if stats["trigger_count"] > 0 else 0

    return {
        "modules": list(module_stats.values()),
        "total_modules": len(module_stats)
    }

@app.get("/api/v1/investigation/search-transactions")
async def search_transactions(
    transaction_id: Optional[str] = None,
    account_id: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Search transactions with multiple filters.

    Args:
        transaction_id: Specific transaction ID
        account_id: Filter by account
        min_amount: Minimum transaction amount
        max_amount: Maximum transaction amount
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        risk_level: Filter by risk level (low/medium/high)
        limit: Max results to return

    Returns:
        List of matching transactions with risk assessments
    """
    from app.models.database import Transaction, RiskAssessment
    import json

    query = db.query(Transaction).join(
        RiskAssessment,
        Transaction.transaction_id == RiskAssessment.transaction_id,
        isouter=True
    )

    # Apply filters
    if transaction_id:
        query = query.filter(Transaction.transaction_id.contains(transaction_id))
    if account_id:
        query = query.filter(Transaction.account_id == account_id)
    if min_amount is not None:
        query = query.filter(Transaction.amount >= min_amount)
    if max_amount is not None:
        query = query.filter(Transaction.amount <= max_amount)
    if start_date:
        query = query.filter(Transaction.timestamp >= start_date)
    if end_date:
        query = query.filter(Transaction.timestamp <= end_date)
    if risk_level:
        risk_map = {"low": (0, 0.3), "medium": (0.3, 0.6), "high": (0.6, 1.0)}
        if risk_level in risk_map:
            min_risk, max_risk = risk_map[risk_level]
            query = query.filter(
                RiskAssessment.risk_score >= min_risk,
                RiskAssessment.risk_score < max_risk
            )

    # Execute query
    results = query.limit(limit).all()

    transactions = []
    for tx in results:
        assessment = db.query(RiskAssessment).filter(
            RiskAssessment.transaction_id == tx.transaction_id
        ).first()

        tx_data = {
            "transaction_id": tx.transaction_id,
            "account_id": tx.account_id,
            "amount": tx.amount,
            "direction": tx.direction,
            "transaction_type": tx.transaction_type,
            "description": tx.description,
            "timestamp": tx.timestamp,
            "counterparty_id": tx.counterparty_id
        }

        if assessment:
            triggered_rules = json.loads(assessment.triggered_rules) if assessment.triggered_rules else {}
            tx_data["risk_score"] = assessment.risk_score
            tx_data["decision"] = assessment.decision
            tx_data["review_status"] = assessment.review_status
            tx_data["triggered_rules_count"] = len(triggered_rules)

        transactions.append(tx_data)

    return {
        "transactions": transactions,
        "count": len(transactions),
        "limit": limit
    }

@app.get("/api/v1/investigation/account/{account_id}")
async def get_account_investigation(
    account_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get comprehensive account information for investigation.

    Returns:
        Account details, transaction history, risk profile, employee info
    """
    from app.models.database import Account, Transaction, RiskAssessment, Employee
    import json

    # Get account
    account = db.query(Account).filter(Account.account_id == account_id).first()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    # Get all transactions
    transactions = db.query(Transaction).filter(
        Transaction.account_id == account_id
    ).order_by(Transaction.timestamp.desc()).limit(100).all()

    # Get risk assessments
    risk_assessments = db.query(RiskAssessment).join(
        Transaction,
        RiskAssessment.transaction_id == Transaction.transaction_id
    ).filter(Transaction.account_id == account_id).all()

    # Calculate stats
    total_transactions = len(transactions)
    total_value = sum(tx.amount for tx in transactions)
    avg_risk = sum(a.risk_score for a in risk_assessments) / len(risk_assessments) if risk_assessments else 0
    high_risk_count = sum(1 for a in risk_assessments if a.risk_score >= 0.6)

    # Get employees (if applicable)
    employees = db.query(Employee).filter(Employee.account_id == account_id).all()

    employee_data = [{
        "employee_id": emp.employee_id,
        "name": emp.name,
        "email": emp.email,
        "department": emp.department,
        "employment_status": emp.employment_status
    } for emp in employees]

    return {
        "account_id": account_id,
        "creation_date": account.creation_date,
        "risk_tier": account.risk_tier,
        "status": account.status,
        "statistics": {
            "total_transactions": total_transactions,
            "total_value": total_value,
            "average_risk_score": avg_risk,
            "high_risk_count": high_risk_count,
            "high_risk_rate": high_risk_count / total_transactions if total_transactions > 0 else 0
        },
        "employees": employee_data,
        "recent_transactions": [
            {
                "transaction_id": tx.transaction_id,
                "amount": tx.amount,
                "timestamp": tx.timestamp,
                "transaction_type": tx.transaction_type
            } for tx in transactions[:10]
        ]
    }

@app.get("/api/v1/investigation/transaction/{transaction_id}/modules")
async def get_transaction_module_breakdown(
    transaction_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get detailed breakdown of all fraud detection modules for a transaction.

    Shows which of the 25 modules triggered and why.
    """
    from app.models.database import Transaction, RiskAssessment
    import json

    # Get transaction
    tx = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Get risk assessment
    assessment = db.query(RiskAssessment).filter(
        RiskAssessment.transaction_id == transaction_id
    ).first()

    if not assessment:
        return {
            "transaction_id": transaction_id,
            "modules_triggered": [],
            "total_modules": 0,
            "risk_score": 0
        }

    triggered_rules = json.loads(assessment.triggered_rules) if assessment.triggered_rules else {}

    modules = []
    for rule_name, rule_info in triggered_rules.items():
        modules.append({
            "name": rule_name,
            "description": rule_info.get("description", ""),
            "weight": rule_info.get("weight", 0),
            "category": rule_info.get("category", "general"),
            "severity": "high" if rule_info.get("weight", 0) >= 0.3 else "medium" if rule_info.get("weight", 0) >= 0.15 else "low"
        })

    # Sort by weight
    modules.sort(key=lambda x: x["weight"], reverse=True)

    return {
        "transaction_id": transaction_id,
        "risk_score": assessment.risk_score,
        "decision": assessment.decision,
        "review_status": assessment.review_status,
        "modules_triggered": modules,
        "total_modules_triggered": len(modules),
        "total_modules_available": 25
    }

@app.get("/api/v1/modules/catalog")
async def get_fraud_modules_catalog(
    group_by: Optional[str] = None,
    user: dict = Depends(verify_token)
):
    """
    Get comprehensive fraud detection modules catalog.

    Shows all 25+ fraud detection modules with detailed information about
    what each module detects, severity levels, and categories.

    Args:
        group_by: Optional grouping (category, severity)

    Returns:
        Complete catalog of fraud detection modules
    """
    if group_by == "category":
        modules = get_module_by_category()
        return {
            "grouped_by": "category",
            "total_modules": get_module_count(),
            "data": modules
        }
    elif group_by == "severity":
        modules = get_module_by_severity()
        return {
            "grouped_by": "severity",
            "total_modules": get_module_count(),
            "data": modules
        }
    else:
        # Return flat list with module IDs
        modules = []
        for module_id, module_info in FRAUD_MODULES_CATALOG.items():
            modules.append({
                "id": module_id,
                **module_info
            })
        return {
            "total_modules": get_module_count(),
            "modules": modules
        }

@app.get("/api/v1/analytics/geographic-fraud")
async def get_geographic_fraud_data(
    time_range: str = "24h",
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get geographic fraud data for heatmap visualization.

    Returns:
        Geographic distribution of fraud activity
    """
    from app.models.database import Transaction, RiskAssessment, HighRiskLocation
    import json
    import random

    time_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = time_map.get(time_range, 24)
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    # Get transactions with risk assessments
    assessments = db.query(RiskAssessment).filter(
        RiskAssessment.review_timestamp > cutoff
    ).all()

    # Get high-risk locations from database
    high_risk_locations = db.query(HighRiskLocation).all()

    # Create geographic data
    # In a real system, transactions would have location data
    # For demo purposes, we'll generate sample geographic data
    geographic_data = []

    # Use high-risk locations from database
    country_data = {}
    for location in high_risk_locations:
        country_data[location.country] = {
            "country": location.country,
            "country_code": location.country_code,
            "risk_level": location.risk_level,
            "transaction_count": 0,
            "fraud_count": 0,
            "total_amount": 0,
            "avg_risk_score": 0,
            "total_risk": 0
        }

    # Add some popular countries if not in high-risk list
    common_countries = [
        {"country": "United States", "code": "US", "risk": "low"},
        {"country": "United Kingdom", "code": "GB", "risk": "low"},
        {"country": "Canada", "code": "CA", "risk": "low"},
        {"country": "Germany", "code": "DE", "risk": "low"},
        {"country": "France", "code": "FR", "risk": "low"},
        {"country": "Japan", "code": "JP", "risk": "low"},
        {"country": "Australia", "code": "AU", "risk": "low"},
    ]

    for country in common_countries:
        if country["country"] not in country_data:
            country_data[country["country"]] = {
                "country": country["country"],
                "country_code": country["code"],
                "risk_level": country["risk"],
                "transaction_count": 0,
                "fraud_count": 0,
                "total_amount": 0,
                "avg_risk_score": 0,
                "total_risk": 0
            }

    # Simulate location data for transactions (in real app, this would come from transaction data)
    for assessment in assessments:
        # Assign random country (in real system, this would be from transaction data)
        country_name = random.choice(list(country_data.keys()))
        country_info = country_data[country_name]

        # Get transaction amount
        tx = db.query(Transaction).filter(
            Transaction.transaction_id == assessment.transaction_id
        ).first()

        if tx:
            country_info["transaction_count"] += 1
            country_info["total_amount"] += tx.amount
            country_info["total_risk"] += assessment.risk_score

            if assessment.risk_score >= 0.6:
                country_info["fraud_count"] += 1

    # Calculate averages
    for data in country_data.values():
        if data["transaction_count"] > 0:
            data["avg_risk_score"] = data["total_risk"] / data["transaction_count"]
            data["fraud_rate"] = data["fraud_count"] / data["transaction_count"]

        geographic_data.append(data)

    # Filter out countries with no transactions
    geographic_data = [g for g in geographic_data if g["transaction_count"] > 0]

    return {
        "time_range": time_range,
        "total_countries": len(geographic_data),
        "data": geographic_data
    }

@app.get("/api/v1/analytics/high-value-transactions")
async def get_high_value_transactions(
    threshold: float = 10000.0,
    time_range: str = "24h",
    limit: int = 100,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get high-value transactions for monitoring.

    Args:
        threshold: Minimum amount to consider high-value (default 10000)
        time_range: Time range to analyze
        limit: Maximum transactions to return

    Returns:
        High-value transactions with risk analysis
    """
    from app.models.database import Transaction, RiskAssessment
    import json

    time_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = time_map.get(time_range, 24)
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    # Get high-value transactions
    transactions = db.query(Transaction).filter(
        Transaction.amount >= threshold,
        Transaction.timestamp > cutoff
    ).order_by(Transaction.amount.desc()).limit(limit).all()

    results = []
    total_amount = 0
    high_risk_count = 0
    flagged_amount = 0

    for tx in transactions:
        # Get risk assessment
        assessment = db.query(RiskAssessment).filter(
            RiskAssessment.transaction_id == tx.transaction_id
        ).first()

        risk_score = assessment.risk_score if assessment else 0
        decision = assessment.decision if assessment else "unknown"
        review_status = assessment.review_status if assessment else "pending"

        triggered_rules = []
        if assessment and assessment.triggered_rules:
            triggered_data = json.loads(assessment.triggered_rules)
            triggered_rules = list(triggered_data.keys())

        total_amount += tx.amount

        is_high_risk = risk_score >= 0.6
        if is_high_risk:
            high_risk_count += 1
            flagged_amount += tx.amount

        results.append({
            "transaction_id": tx.transaction_id,
            "account_id": tx.account_id,
            "amount": tx.amount,
            "direction": tx.direction,
            "transaction_type": tx.transaction_type,
            "description": tx.description,
            "timestamp": tx.timestamp,
            "counterparty_id": tx.counterparty_id,
            "risk_score": risk_score,
            "decision": decision,
            "review_status": review_status,
            "triggered_rules": triggered_rules,
            "is_high_risk": is_high_risk
        })

    return {
        "time_range": time_range,
        "threshold": threshold,
        "total_transactions": len(results),
        "total_amount": total_amount,
        "high_risk_count": high_risk_count,
        "high_risk_rate": high_risk_count / len(results) if results else 0,
        "flagged_amount": flagged_amount,
        "transactions": results
    }

@app.get("/api/v1/analytics/limit-violations")
async def get_limit_violations(
    time_range: str = "24h",
    severity: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get account limit violations.

    Args:
        time_range: Time range to analyze
        severity: Filter by severity (low, medium, high, critical)
        limit: Maximum violations to return

    Returns:
        List of limit violations with account details
    """
    from app.models.database import Transaction, RiskAssessment, Account, AccountLimit
    import json

    time_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = time_map.get(time_range, 24)
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    # Get all accounts with limits
    accounts_with_limits = db.query(Account).join(
        AccountLimit,
        Account.account_id == AccountLimit.account_id
    ).all()

    violations = []
    total_violation_amount = 0

    for account in accounts_with_limits:
        # Get account limits
        limits = db.query(AccountLimit).filter(
            AccountLimit.account_id == account.account_id
        ).all()

        for limit_obj in limits:
            # Get transactions in the period
            transactions = db.query(Transaction).filter(
                Transaction.account_id == account.account_id,
                Transaction.timestamp > cutoff
            ).all()

            # Check for violations based on limit type
            for tx in transactions:
                violation = None
                severity_level = "low"

                # Check single transaction limit
                if limit_obj.single_transaction_limit and tx.amount > limit_obj.single_transaction_limit:
                    violation = {
                        "type": "single_transaction",
                        "limit": limit_obj.single_transaction_limit,
                        "actual": tx.amount,
                        "excess": tx.amount - limit_obj.single_transaction_limit
                    }
                    severity_level = "high" if violation["excess"] > limit_obj.single_transaction_limit * 0.5 else "medium"

                # Check daily limit
                if limit_obj.daily_limit:
                    # Calculate daily total
                    day_start = tx.timestamp[:10]  # YYYY-MM-DD
                    daily_txs = [t for t in transactions if t.timestamp.startswith(day_start)]
                    daily_total = sum(t.amount for t in daily_txs)

                    if daily_total > limit_obj.daily_limit:
                        violation = {
                            "type": "daily_limit",
                            "limit": limit_obj.daily_limit,
                            "actual": daily_total,
                            "excess": daily_total - limit_obj.daily_limit
                        }
                        severity_level = "critical" if violation["excess"] > limit_obj.daily_limit * 0.5 else "high"

                if violation:
                    # Get risk assessment
                    assessment = db.query(RiskAssessment).filter(
                        RiskAssessment.transaction_id == tx.transaction_id
                    ).first()

                    total_violation_amount += violation["excess"]

                    violations.append({
                        "violation_id": f"VIO-{tx.transaction_id[:8]}",
                        "transaction_id": tx.transaction_id,
                        "account_id": account.account_id,
                        "timestamp": tx.timestamp,
                        "violation": violation,
                        "severity": severity_level,
                        "risk_score": assessment.risk_score if assessment else 0,
                        "review_status": assessment.review_status if assessment else "pending"
                    })

    # Filter by severity if specified
    if severity:
        violations = [v for v in violations if v["severity"] == severity]

    # Sort by severity and excess amount
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    violations.sort(key=lambda x: (severity_order.get(x["severity"], 4), -x["violation"]["excess"]))

    return {
        "time_range": time_range,
        "total_violations": len(violations),
        "total_violation_amount": total_violation_amount,
        "severity_breakdown": {
            "critical": len([v for v in violations if v["severity"] == "critical"]),
            "high": len([v for v in violations if v["severity"] == "high"]),
            "medium": len([v for v in violations if v["severity"] == "medium"]),
            "low": len([v for v in violations if v["severity"] == "low"])
        },
        "violations": violations[:limit]
    }

@app.get("/api/v1/analytics/account-risk-timeline/{account_id}")
async def get_account_risk_timeline(
    account_id: str,
    time_range: str = "7d",
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Get risk score timeline for a specific account.

    Shows how an account's risk has evolved over time.

    Args:
        account_id: Account ID to analyze
        time_range: Time range for timeline

    Returns:
        Time-series risk score data for the account
    """
    from app.models.database import Transaction, RiskAssessment
    import json

    time_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = time_map.get(time_range, 24)
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    # Get all transactions for the account
    transactions = db.query(Transaction).filter(
        Transaction.account_id == account_id,
        Transaction.timestamp > cutoff
    ).order_by(Transaction.timestamp).all()

    timeline_data = []
    cumulative_risk = []

    for tx in transactions:
        # Get risk assessment
        assessment = db.query(RiskAssessment).filter(
            RiskAssessment.transaction_id == tx.transaction_id
        ).first()

        if assessment:
            risk_score = assessment.risk_score
            cumulative_risk.append(risk_score)

            # Calculate moving average (last 5 transactions)
            moving_avg = sum(cumulative_risk[-5:]) / min(len(cumulative_risk), 5)

            triggered_rules = []
            if assessment.triggered_rules:
                triggered_data = json.loads(assessment.triggered_rules)
                triggered_rules = list(triggered_data.keys())

            timeline_data.append({
                "timestamp": tx.timestamp,
                "transaction_id": tx.transaction_id,
                "amount": tx.amount,
                "risk_score": risk_score,
                "moving_average": moving_avg,
                "decision": assessment.decision,
                "review_status": assessment.review_status,
                "triggered_rules_count": len(triggered_rules),
                "transaction_type": tx.transaction_type
            })

    # Calculate statistics
    risk_scores = [d["risk_score"] for d in timeline_data]

    return {
        "account_id": account_id,
        "time_range": time_range,
        "total_transactions": len(timeline_data),
        "statistics": {
            "average_risk": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
            "max_risk": max(risk_scores) if risk_scores else 0,
            "min_risk": min(risk_scores) if risk_scores else 0,
            "current_risk": risk_scores[-1] if risk_scores else 0,
            "risk_trend": "increasing" if len(risk_scores) >= 2 and risk_scores[-1] > risk_scores[0] else "decreasing",
            "high_risk_count": len([r for r in risk_scores if r >= 0.6])
        },
        "timeline": timeline_data
    }

# ==================== Run Application ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
