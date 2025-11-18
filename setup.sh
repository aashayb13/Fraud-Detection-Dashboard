#!/bin/bash
# Setup script for transaction-monitoring project

echo "========================================="
echo "Transaction Monitoring - Setup Script"
echo "========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "WARNING: requirements.txt not found"
fi
echo ""

# Git configuration
echo "Configuring Git..."
git config pull.rebase false  # Use merge strategy for pulls
echo "✓ Git configured"
echo ""

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment manually, run:"
echo "  source venv/bin/activate"
echo ""
echo "VS Code should automatically activate the virtual environment when you open a terminal."
