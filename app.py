"""
Clinical Decision Support Web Application
Flask-based web interface for doctors to interact with the AI-powered clinical decision system

Features:
- Patient data upload (FHIR JSON, manual entry)
- Clinical query interface
- Evidence-based care recommendations
- Citation tracking and explainable outputs
- Export functionality for clinical documentation
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from clinical_decision_engine import ClinicalDecisionEngine, create_clinical_decision_engine
from fhir_data_parser import FHIRDataParser, create_fhir_parser


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'clinical-decision-support-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

CORS(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global instances
clinical_engine = None
fhir_parser = None


def init_system():
    """Initialize the clinical decision system"""
    global clinical_engine, fhir_parser
    try:
        clinical_engine = create_clinical_decision_engine()
        fhir_parser = create_fhir_parser()
        print("✅ Clinical Decision Support System initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False


@app.route('/')
def index():
    """Main dashboard for clinical decision support"""
    return render_template('index.html')


@app.route('/patient/upload', methods=['GET', 'POST'])
def upload_patient():
    """Handle patient data upload - FHIR JSON or manual entry"""
    if request.method == 'GET':
        return render_template('patient_upload.html')
    
    try:
        upload_type = request.form.get('upload_type', 'manual')
        
        if upload_type == 'fhir':
            # Handle FHIR file upload
            if 'fhir_file' not in request.files:
                flash('No FHIR file provided', 'error')
                return redirect(request.url)
            
            file = request.files['fhir_file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if file and file.filename.endswith('.json'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Parse FHIR file
                with open(filepath, 'r') as f:
                    fhir_data = json.load(f)
                
                parsed_data = fhir_parser.parse_fhir_bundle(fhir_data)
                patient_data = fhir_parser.convert_to_clinical_format(parsed_data)
                
                # Store in session
                session['patient_data'] = patient_data
                session['patient_source'] = 'fhir'
                
                flash('FHIR data successfully uploaded and parsed', 'success')
                return redirect(url_for('patient_summary'))
        
        else:
            # Handle manual patient data entry
            patient_data = {
                "patient_id": request.form.get('patient_id', str(uuid.uuid4())),
                "demographics": {
                    "full_name": request.form.get('full_name', ''),
                    "age": int(request.form.get('age', 0)) if request.form.get('age') else 0,
                    "gender": request.form.get('gender', ''),
                    "phone": request.form.get('phone', ''),
                    "email": request.form.get('email', '')
                },
                "past_medical_history": [
                    item.strip() for item in request.form.get('medical_history', '').split('\n')
                    if item.strip()
                ],
                "medications": [
                    item.strip() for item in request.form.get('medications', '').split('\n')
                    if item.strip()
                ],
                "allergies": [
                    item.strip() for item in request.form.get('allergies', '').split('\n')
                    if item.strip()
                ],
                "chief_complaint": request.form.get('chief_complaint', ''),
                "vital_signs": {},
                "laboratory_results": {}
            }
            
            # Parse vital signs
            vitals_text = request.form.get('vital_signs', '')
            for line in vitals_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    patient_data["vital_signs"][key.strip()] = value.strip()
            
            # Parse lab results
            labs_text = request.form.get('lab_results', '')
            for line in labs_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    patient_data["laboratory_results"][key.strip()] = value.strip()
            
            # Store in session
            session['patient_data'] = patient_data
            session['patient_source'] = 'manual'
            
            flash('Patient data successfully entered', 'success')
            return redirect(url_for('patient_summary'))
    
    except Exception as e:
        flash(f'Error processing patient data: {str(e)}', 'error')
        return redirect(request.url)


@app.route('/patient/summary')
def patient_summary():
    """Display patient summary and allow for clinical queries"""
    patient_data = session.get('patient_data')
    if not patient_data:
        flash('No patient data available. Please upload patient information first.', 'error')
        return redirect(url_for('upload_patient'))
    
    return render_template('patient_summary.html', patient=patient_data)


@app.route('/clinical/query', methods=['POST'])
def clinical_query():
    """Process clinical query and return decision support"""
    try:
        # Get patient data from session
        patient_data = session.get('patient_data')
        if not patient_data:
            return jsonify({
                'error': 'No patient data available. Please upload patient information first.'
            }), 400
        
        # Get query parameters
        clinical_question = request.json.get('clinical_question', '').strip()
        chief_complaint = request.json.get('chief_complaint', patient_data.get('chief_complaint', '')).strip()
        urgency = request.json.get('urgency', 'routine')
        
        if not clinical_question:
            return jsonify({'error': 'Clinical question is required'}), 400
        
        # Process query through clinical decision engine
        print(f"🔍 Processing clinical query: {clinical_question}")
        
        result = clinical_engine.process_clinical_query(
            patient_data=patient_data,
            clinical_question=clinical_question,
            chief_complaint=chief_complaint,
            urgency=urgency
        )
        
        # Store result in session for potential export
        session['last_query_result'] = result
        session['last_query_timestamp'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        print(f"Error processing clinical query: {e}")
        return jsonify({
            'error': f'Error processing clinical query: {str(e)}'
        }), 500


@app.route('/export/report')
def export_report():
    """Export clinical decision report as JSON"""
    result = session.get('last_query_result')
    timestamp = session.get('last_query_timestamp')
    
    if not result:
        flash('No clinical query result available to export', 'error')
        return redirect(url_for('index'))
    
    # Prepare export data
    export_data = {
        'report_generated': datetime.now().isoformat(),
        'query_timestamp': timestamp,
        'clinical_decision_report': result,
        'export_version': '1.0'
    }
    
    # Return as JSON file download
    from flask import Response
    return Response(
        json.dumps(export_data, indent=2, default=str),
        mimetype='application/json',
        headers={'Content-Disposition': 'attachment; filename=clinical_decision_report.json'}
    )


@app.route('/api/system/status')
def system_status():
    """Check system health status"""
    try:
        status = {
            'clinical_engine': clinical_engine is not None,
            'fhir_parser': fhir_parser is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test hybrid RAG system if available
        if clinical_engine:
            try:
                # Test basic functionality
                hybrid_status = hasattr(clinical_engine, 'hybrid_rag') and clinical_engine.hybrid_rag is not None
                status['hybrid_rag'] = hybrid_status
            except:
                status['hybrid_rag'] = False
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/sample-data')
def get_sample_data():
    """Provide sample patient data for testing"""
    sample_patient = {
        "patient_id": "DEMO_001",
        "demographics": {
            "full_name": "John Doe",
            "age": 65,
            "gender": "male",
            "phone": "555-123-4567",
            "email": "john.doe@email.com"
        },
        "past_medical_history": [
            "Hypertension (diagnosed 2018)",
            "Type 2 diabetes mellitus (diagnosed 2020)", 
            "Hyperlipidemia (diagnosed 2019)"
        ],
        "medications": [
            "Lisinopril 10mg daily",
            "Metformin 1000mg twice daily",
            "Atorvastatin 40mg daily"
        ],
        "allergies": ["Penicillin"],
        "chief_complaint": "Elevated blood pressure readings at home",
        "vital_signs": {
            "Blood Pressure": "150/90 mmHg",
            "Heart Rate": "78 bpm",
            "Temperature": "98.6°F",
            "Respiratory Rate": "16/min"
        },
        "laboratory_results": {
            "HbA1c": "7.2%",
            "Total Cholesterol": "180 mg/dL",
            "Creatinine": "1.1 mg/dL"
        }
    }
    
    return jsonify(sample_patient)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="Internal server error"), 500


if __name__ == '__main__':
    print("🏥 Starting Clinical Decision Support Web Application...")
    
    # Initialize system
    if init_system():
        print("✅ System initialization complete")
        print("🌐 Starting server on http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ System initialization failed - cannot start web application")
        exit(1)