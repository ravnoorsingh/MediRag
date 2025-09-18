#!/usr/bin/env python3
"""
Simple Web Application Test
Test the Flask web application endpoints directly
"""

import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_web_app():
    """Test the web application directly"""
    print("🌐 Testing Web Application Integration")
    print("-" * 40)
    
    try:
        # Import the Flask app
        from app import app, clinical_engine, fhir_parser
        
        # Check if components are initialized
        if not clinical_engine:
            print("❌ Clinical engine not initialized in web app")
            return False
        
        if not fhir_parser:
            print("❌ FHIR parser not initialized in web app")
            return False
        
        print("✅ All web app components initialized")
        
        # Test with app context
        with app.app_context():
            # Create test client
            client = app.test_client()
            
            # Test main page
            print("🔍 Testing main page...")
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Main page accessible")
            else:
                print(f"❌ Main page failed: {response.status_code}")
                return False
            
            # Test patient upload page
            print("🔍 Testing patient upload page...")
            response = client.get('/patient/upload')
            if response.status_code == 200:
                print("✅ Patient upload page accessible")
            else:
                print(f"❌ Patient upload page failed: {response.status_code}")
            
            # Test clinical query functionality
            print("🔍 Testing clinical query processing...")
            test_patient = {
                "patient_id": "web-test-001",
                "demographics": {"age": 45, "gender": "male"},
                "past_medical_history": ["Asthma"],
                "medications": ["Albuterol"],
                "chief_complaint": "Shortness of breath"
            }
            
            # Simulate session data
            with client.session_transaction() as sess:
                sess['patient_data'] = test_patient
            
            # Test clinical query endpoint
            query_data = {
                "clinical_question": "How should I manage acute asthma exacerbation?",
                "chief_complaint": "Shortness of breath",
                "urgency": "urgent"
            }
            
            response = client.post('/clinical/query',
                                   data=json.dumps(query_data),
                                   content_type='application/json')
            
            if response.status_code == 200:
                result = response.get_json()
                if result and result.get('success'):
                    care_options = result.get('result', {}).get('care_options', [])
                    print(f"✅ Clinical query successful - {len(care_options)} care options")
                    return True
                else:
                    print(f"❌ Clinical query failed: {result}")
                    return False
            else:
                print(f"❌ Clinical query endpoint failed: {response.status_code}")
                print(f"Response: {response.get_data(as_text=True)}")
                return False
    
    except Exception as e:
        print(f"❌ Web app test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_web_app()
    if success:
        print("\n🎉 Web application integration test PASSED!")
        exit(0)
    else:
        print("\n❌ Web application integration test FAILED!")
        exit(1)