#!/bin/bash

# Clinical Decision Support System - Launch Script
# This script activates the virtual environment and runs the Flask application

echo "🏥 Starting Clinical Decision Support System..."
echo "📁 Working directory: $(pwd)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment and run the app
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Run the Flask app using the virtual environment Python
echo "🚀 Starting Flask web application..."
echo "📋 Access the Clinical Decision Support System at: http://localhost:5001"
echo ""
echo "Features available:"
echo "  • Patient Data Upload (FHIR JSON or Manual Entry)"
echo "  • AI-Powered Clinical Query Processing"
echo "  • Evidence-Based Care Recommendations"
echo "  • Medical Literature Citations"
echo "  • Exportable Clinical Reports"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"

# Run the Flask app using the virtual environment Python
./venv/bin/python app.py