#!/bin/bash
# Setup script for Deeter Interview Project

echo "=== Setting up Deeter Interview Environment ==="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Jupyter kernel
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=deeter-interview --display-name="Deeter Interview"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter, run:"
echo "  jupyter notebook Interview.ipynb"
echo ""
echo "Or open in VSCode and select the 'deeter-interview' kernel"
