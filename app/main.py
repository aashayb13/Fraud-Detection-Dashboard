# app/utils/main.py
import uuid
import json
import datetime
from typing import Dict, Any, List, Union, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("transaction_monitoring.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("transaction_monitoring")

def generate_id() -> str:
    """Generate a unique ID string for transactions and assessments."""
    return str(uuid.uuid4())

def current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.datetime.utcnow().isoformat()

def parse_timestamp(timestamp_str: str) -> datetime.datetime:
    """Parse ISO format timestamp to datetime object."""
    return datetime.datetime.fromisoformat(timestamp_str)

def format_currency(amount: float) -> str:
    """Format a number as currency."""
    return f"${amount:,.2f}"

def json_serialize(obj: Dict[str, Any]) -> str:
    """Convert object to JSON string."""
    return json.dumps(obj, default=json_serializer)

def json_deserialize(json_str: str) -> Dict[str, Any]:
    """Convert JSON string to object."""
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to deserialize JSON: {json_str}")
        return {}

def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def calculate_time_difference(timestamp: str, unit: str = "hours") -> float:
    """
    Calculate time difference between now and a timestamp.
    
    Args:
        timestamp: ISO format timestamp string
        unit: "hours", "days", "minutes", or "seconds"
        
    Returns:
        Time difference in the specified unit
    """
    now = datetime.datetime.utcnow()
    timestamp_dt = parse_timestamp(timestamp)
    diff = now - timestamp_dt
    
    if unit == "days":
        return diff.total_seconds() / (60 * 60 * 24)
    elif unit == "hours":
        return diff.total_seconds() / (60 * 60)
    elif unit == "minutes":
        return diff.total_seconds() / 60
    elif unit == "seconds":
        return diff.total_seconds()
    else:
        raise ValueError(f"Unknown time unit: {unit}")

def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and validate input data.
    
    - Ensures required fields exist
    - Converts types where necessary
    - Removes potentially dangerous values
    
    Args:
        data: Input data dictionary
        
    Returns:
        Sanitized data dictionary
    """
    # Make a copy to avoid modifying the original
    sanitized = data.copy()
    
    # Handle amount - ensure it's a positive float
    if "amount" in sanitized:
        try:
            sanitized["amount"] = float(sanitized["amount"])
            if sanitized["amount"] < 0:
                sanitized["amount"] = abs(sanitized["amount"])
                logger.warning("Converted negative amount to positive")
        except (ValueError, TypeError):
            sanitized["amount"] = 0.0
            logger.error(f"Invalid amount value: {data.get('amount')}")
    
    # Ensure transaction type is uppercase
    if "transaction_type" in sanitized and sanitized["transaction_type"]:
        sanitized["transaction_type"] = sanitized["transaction_type"].upper()
    
    # Ensure metadata is a valid JSON string
    if "metadata" in sanitized and isinstance(sanitized["metadata"], dict):
        sanitized["metadata"] = json_serialize(sanitized["metadata"])
    elif "metadata" in sanitized and not isinstance(sanitized["metadata"], str):
        sanitized["metadata"] = "{}"
    
    return sanitized

def group_by(items: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group a list of dictionaries by a key.
    
    Args:
        items: List of dictionaries
        key: Key to group by
        
    Returns:
        Dictionary of grouped items
    """
    result = {}
    for item in items:
        key_value = item.get(key)
        if key_value not in result:
            result[key_value] = []
        result[key_value].append(item)
    return result

def filter_dict(data: Dict[str, Any], allowed_keys: List[str]) -> Dict[str, Any]:
    """
    Filter a dictionary to include only allowed keys.
    
    Args:
        data: Dictionary to filter
        allowed_keys: List of keys to keep
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in data.items() if k in allowed_keys}

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Default value if denominator is zero
        
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default

def mask_sensitive_data(data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
    """
    Mask sensitive data in a dictionary.
    
    Args:
        data: Dictionary containing data
        sensitive_fields: List of sensitive field names
        
    Returns:
        Dictionary with sensitive fields masked
    """
    result = data.copy()
    for field in sensitive_fields:
        if field in result:
            value = str(result[field])
            if len(value) > 4:
                # Mask all but last 4 characters
                result[field] = '*' * (len(value) - 4) + value[-4:]
            else:
                # Mask everything for very short values
                result[field] = '*' * len(value)
    return result