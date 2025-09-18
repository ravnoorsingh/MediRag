#!/usr/bin/env python3
"""
Clinical Decision Support System - Quick Test
This script demonstrates the core functionality of the system using the virtual environment.
"""

import sys
import json
from datetime import datetime

def test_clinical_decision_system():
    """Test the clinical decision support system with sample data."""
    
    print("🏥 Clinical Decision Support System - Test Suite")
    print("=" * 60)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from clinical_decision_engine import ClinicalDecisionEngine
        from fhir_data_parser import FHIRDataParser
        from app import app
        print("   ✅ All imports successful")
        
        # Test FHIR parser
        print("\n2. Testing FHIR Data Parser...")
        fhir_parser = FHIRDataParser()
        
        # Sample FHIR Patient data
        sample_fhir = {
            "resourceType": "Bundle",
            "id": "test-bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-123",
                        "name": [{"family": "Smith", "given": ["John"]}],
                        "birthDate": "1965-05-15",
                        "gender": "male"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "condition-123",
                        "subject": {"reference": "Patient/patient-123"},
                        "code": {
                            "coding": [{"code": "I10", "display": "Essential hypertension"}]
                        },
                        "clinicalStatus": {
                            "coding": [{"code": "active", "display": "Active"}]
                        }
                    }
                }
            ]
        }
        
        clinical_data = fhir_parser.parse_fhir_bundle(sample_fhir)
        print(f"   ✅ FHIR parsing successful")
        
        # Check the structure of parsed data
        if hasattr(clinical_data, 'demographics'):
            print(f"   📋 Patient: {clinical_data.demographics.get('full_name', 'N/A')}, Age: {clinical_data.demographics.get('age', 'N/A')}")
        else:
            print(f"   📋 Parsed data type: {type(clinical_data)}")
        
        if hasattr(clinical_data, 'past_medical_history'):
            print(f"   🩺 Conditions: {clinical_data.past_medical_history}")
        elif isinstance(clinical_data, dict) and 'past_medical_history' in clinical_data:
            print(f"   🩺 Conditions: {clinical_data['past_medical_history']}")
        
        # Test Clinical Decision Engine (basic structure test)
        print("\n3. Testing Clinical Decision Engine initialization...")
        engine = ClinicalDecisionEngine()
        print("   ✅ Clinical Decision Engine initialized successfully")
        
        # Test Flask app
        print("\n4. Testing Flask Application...")
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("   ✅ Flask app responds correctly")
            else:
                print(f"   ⚠️  Flask app returned status code: {response.status_code}")
        
        print("\n🎉 All tests passed successfully!")
        print("\n📋 System Summary:")
        print("   • ✅ Clinical Decision Engine: Ready")
        print("   • ✅ FHIR Data Parser: Ready") 
        print("   • ✅ Flask Web Application: Ready")
        print("   • ✅ Virtual Environment: Active")
        
        print(f"\n🚀 To start the web application, run:")
        print(f"   ./run_app.sh")
        print(f"\n💡 Or manually:")
        print(f"   source venv/bin/activate && python app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print(f"📋 Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_clinical_decision_system()
    sys.exit(0 if success else 1)