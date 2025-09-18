#!/usr/bin/env python3
"""
Comprehensive RAG Pipeline Test
Test the complete clinical decision support RAG system end-to-end

This script verifies:
1. Database connectivity (Neo4j + Qdrant)
2. Medical data retrieval from cached database
3. Patient data injection and contextualization
4. Evidence-based response generation with citations
5. Complete web application workflow simulation
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clinical_decision_engine import ClinicalDecisionEngine, create_clinical_decision_engine
from utils import HybridMedicalRAG
from fhir_data_parser import FHIRDataParser, create_fhir_parser


class RAGPipelineTest:
    """Comprehensive RAG pipeline testing suite"""
    
    def __init__(self):
        """Initialize test environment"""
        print("🧪 Initializing RAG Pipeline Test Suite")
        print("=" * 60)
        
        self.test_results = {
            'database_connectivity': False,
            'medical_data_retrieval': False,
            'patient_injection': False,
            'evidence_citations': False,
            'web_integration': False,
            'overall_score': 0
        }
        
        # Initialize components
        try:
            print("🔧 Initializing clinical decision engine...")
            self.clinical_engine = create_clinical_decision_engine()
            print("✅ Clinical decision engine initialized")
        except Exception as e:
            print(f"❌ Failed to initialize clinical engine: {e}")
            self.clinical_engine = None
        
        try:
            print("🔧 Initializing FHIR parser...")
            self.fhir_parser = create_fhir_parser()
            print("✅ FHIR parser initialized")
        except Exception as e:
            print(f"❌ Failed to initialize FHIR parser: {e}")
            self.fhir_parser = None
        
        try:
            print("🔧 Initializing hybrid RAG system...")
            self.hybrid_rag = HybridMedicalRAG()
            print("✅ Hybrid RAG system initialized")
        except Exception as e:
            print(f"❌ Failed to initialize hybrid RAG: {e}")
            self.hybrid_rag = None
    
    def test_database_connectivity(self) -> bool:
        """Test Neo4j and Qdrant database connectivity"""
        print("\n🔍 Testing Database Connectivity")
        print("-" * 40)
        
        if not self.hybrid_rag:
            print("❌ Hybrid RAG not initialized")
            return False
        
        try:
            # Test Neo4j connectivity
            print("📊 Testing Neo4j connectivity...")
            neo4j_stats = self.hybrid_rag.get_level_statistics()
            print(f"✅ Neo4j connected - found {len(neo4j_stats)} levels")
            
            # Test Qdrant connectivity
            print("🔍 Testing Qdrant connectivity...")
            test_query = self.hybrid_rag.semantic_search_across_levels(
                query_text="diabetes treatment",
                level_filter=["middle", "bottom"],  # Use lowercase levels
                top_k=3
            )
            vector_count = len(test_query.get('vector_results', []))
            graph_count = len(test_query.get('graph_results', []))
            
            print(f"✅ Qdrant connected - {vector_count} vector results, {graph_count} graph results")
            
            self.test_results['database_connectivity'] = True
            return True
            
        except Exception as e:
            print(f"❌ Database connectivity failed: {e}")
            return False
    
    def test_medical_data_retrieval(self) -> bool:
        """Test retrieval of medical books, papers, and dictionary from cached database"""
        print("\n📚 Testing Medical Data Retrieval")
        print("-" * 40)
        
        if not self.hybrid_rag:
            print("❌ Hybrid RAG not initialized")
            return False
        
        test_queries = [
            "hypertension management guidelines",
            "diabetes medication interactions", 
            "cardiac arrhythmia treatment protocols",
            "pneumonia antibiotic selection"
        ]
        
        total_retrieved = 0
        source_types_found = set()
        
        for query in test_queries:
            try:
                print(f"🔎 Searching: {query}")
                results = self.hybrid_rag.semantic_search_across_levels(
                    query_text=query,
                    level_filter=["middle", "bottom"],  # Medical literature and concepts (lowercase)
                    top_k=5
                )
                
                vector_results = results.get('vector_results', [])
                graph_results = results.get('graph_results', [])
                
                query_total = len(vector_results) + len(graph_results)
                total_retrieved += query_total
                
                # Check source types from vector results
                for result in vector_results:
                    if hasattr(result, 'payload'):
                        source_type = result.payload.get('node_type', 'unknown')
                        source_types_found.add(source_type)
                
                print(f"   📋 Found {query_total} relevant sources")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        print(f"\n📊 Total sources retrieved: {total_retrieved}")
        print(f"📋 Source types found: {list(source_types_found)}")
        
        success = total_retrieved > 0 and len(source_types_found) > 0
        self.test_results['medical_data_retrieval'] = success
        
        if success:
            print("✅ Medical data retrieval successful")
        else:
            print("❌ No medical data retrieved from database")
        
        return success
    
    def test_patient_injection(self) -> bool:
        """Test patient data injection into RAG context"""
        print("\n🏥 Testing Patient Data Injection")
        print("-" * 40)
        
        if not self.clinical_engine:
            print("❌ Clinical engine not initialized")
            return False
        
        # Load sample FHIR patient data
        sample_fhir_path = "sample_fhir_data/patient_diabetes_hypertension.json"
        if not os.path.exists(sample_fhir_path):
            print(f"❌ Sample FHIR data not found: {sample_fhir_path}")
            return False
        
        try:
            # Load and parse FHIR data
            with open(sample_fhir_path, 'r') as f:
                fhir_data = json.load(f)
            
            print("📋 Loading FHIR patient data...")
            if self.fhir_parser:
                parsed_data = self.fhir_parser.parse_fhir_bundle(fhir_data)
                patient_data = self.fhir_parser.convert_to_clinical_format(parsed_data)
            else:
                # Fallback to direct FHIR usage
                patient_data = fhir_data
            
            print(f"✅ Patient loaded: {patient_data.get('demographics', {}).get('age', 'unknown')} year old")
            
            # Test clinical query with patient context
            clinical_question = "What are the best treatment options for managing diabetes in this patient?"
            chief_complaint = "Poor blood sugar control"
            
            print(f"🔍 Processing clinical query with patient context...")
            result = self.clinical_engine.process_clinical_query(
                patient_data=patient_data,
                clinical_question=clinical_question,
                chief_complaint=chief_complaint,
                urgency="routine"
            )
            
            # Verify patient-specific context was used
            patient_summary = result.get('patient_summary', {})
            care_options = result.get('care_options', [])
            
            context_indicators = [
                'age' in str(patient_summary).lower(),
                'diabetes' in str(result).lower(),
                'hypertension' in str(result).lower(),
                len(care_options) > 0
            ]
            
            success = sum(context_indicators) >= 3
            
            print(f"📊 Patient context indicators: {sum(context_indicators)}/4")
            print(f"📋 Generated {len(care_options)} care options")
            
            self.test_results['patient_injection'] = success
            
            if success:
                print("✅ Patient data injection successful")
            else:
                print("❌ Patient context not properly injected")
            
            return success
            
        except Exception as e:
            print(f"❌ Patient injection test failed: {e}")
            return False
    
    def test_evidence_citations(self) -> bool:
        """Test evidence-based citations from database sources"""
        print("\n📚 Testing Evidence-Based Citations")
        print("-" * 40)
        
        if not self.clinical_engine:
            print("❌ Clinical engine not initialized")
            return False
        
        # Create test patient data
        test_patient = {
            "patient_id": "test-001",
            "demographics": {"age": 65, "gender": "female"},
            "past_medical_history": ["Type 2 Diabetes", "Hypertension"],
            "medications": ["Metformin", "Lisinopril"],
            "chief_complaint": "Chest pain"
        }
        
        try:
            print("🔍 Processing clinical query to test citations...")
            result = self.clinical_engine.process_clinical_query(
                patient_data=test_patient,
                clinical_question="What are the cardiac risk factors and management strategies?",
                chief_complaint="Chest pain",
                urgency="urgent"
            )
            
            # Check for evidence-based citations
            care_options = result.get('care_options', [])
            citation_indicators = []
            
            for i, option in enumerate(care_options):
                rationale = option.get('rationale', '')
                evidence_list = option.get('evidence', [])
                
                # Look for citation patterns [1], [2], etc.
                import re
                citations_found = len(re.findall(r'\[\d+\]', rationale))
                
                citation_indicators.extend([
                    citations_found > 0,
                    len(evidence_list) > 0,
                    any('research' in str(ev).lower() or 'guideline' in str(ev).lower() 
                        for ev in evidence_list)
                ])
                
                print(f"   📋 Option {i+1}: {citations_found} citations, {len(evidence_list)} evidence pieces")
            
            success = sum(citation_indicators) >= len(care_options)
            
            print(f"📊 Citation indicators: {sum(citation_indicators)}/{len(citation_indicators)}")
            
            self.test_results['evidence_citations'] = success
            
            if success:
                print("✅ Evidence-based citations working")
            else:
                print("❌ Citations not properly linked to evidence")
            
            return success
            
        except Exception as e:
            print(f"❌ Evidence citation test failed: {e}")
            return False
    
    def test_web_integration(self) -> bool:
        """Test web application integration workflow"""
        print("\n🌐 Testing Web Application Integration")
        print("-" * 40)
        
        try:
            # Import Flask app components directly
            import sys
            import os
            
            # Ensure we can import the app
            try:
                from app import app
                print("✅ Successfully imported Flask app")
            except Exception as e:
                print(f"❌ Failed to import Flask app: {e}")
                return False
            
            # Check if clinical engine is available in app context
            with app.app_context():
                # Import the clinical engine from the app module
                from app import clinical_engine as app_clinical_engine
                
                if not app_clinical_engine:
                    print("⚠️ App clinical engine not initialized, creating new one...")
                    try:
                        from clinical_decision_engine import create_clinical_decision_engine
                        app_clinical_engine = create_clinical_decision_engine()
                        print("✅ Created clinical engine for web integration test")
                    except Exception as e:
                        print(f"❌ Failed to create clinical engine: {e}")
                        return False
                
                # Test basic clinical query processing
                test_patient = {
                    "patient_id": "web-test-001",
                    "demographics": {"age": 45, "gender": "male"},
                    "past_medical_history": ["Asthma"],
                    "medications": ["Albuterol"],
                    "chief_complaint": "Shortness of breath"
                }
                
                print("🔍 Testing web application query processing...")
                result = app_clinical_engine.process_clinical_query(
                    patient_data=test_patient,
                    clinical_question="How should I manage acute asthma exacerbation?",
                    chief_complaint="Shortness of breath",
                    urgency="urgent"
                )
                
                # Verify web-compatible response format
                required_fields = [
                    'query_info', 'patient_summary', 'clinical_question', 
                    'care_options', 'risk_assessment'
                ]
                
                format_check = all(field in result for field in required_fields)
                care_options_valid = len(result.get('care_options', [])) >= 1
                
                print(f"📊 Response format check: {format_check}")
                print(f"📋 Care options generated: {len(result.get('care_options', []))}")
                
                # Additional web app functionality test
                if format_check and care_options_valid:
                    # Test JSON serialization (important for web responses)
                    try:
                        json_result = json.dumps(result, default=str, indent=2)
                        json_length = len(json_result)
                        print(f"� JSON response size: {json_length} characters")
                        serialization_ok = json_length > 100  # Should have substantial content
                    except Exception as e:
                        print(f"❌ JSON serialization failed: {e}")
                        serialization_ok = False
                    
                    success = format_check and care_options_valid and serialization_ok
                else:
                    success = False
                
                self.test_results['web_integration'] = success
                
                if success:
                    print("✅ Web application integration working")
                else:
                    print("❌ Web integration issues detected")
                
                return success
            
        except Exception as e:
            print(f"❌ Web integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        print("\n🚀 Running Comprehensive RAG Pipeline Test")
        print("=" * 60)
        
        test_methods = [
            ('Database Connectivity', self.test_database_connectivity),
            ('Medical Data Retrieval', self.test_medical_data_retrieval),
            ('Patient Data Injection', self.test_patient_injection),
            ('Evidence-Based Citations', self.test_evidence_citations),
            ('Web Application Integration', self.test_web_integration)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
        
        # Calculate overall score
        overall_score = (passed_tests / total_tests) * 100
        self.test_results['overall_score'] = overall_score
        
        # Generate final report
        print("\n" + "=" * 60)
        print("🏆 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            if test_name != 'overall_score':
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name.replace('_', ' ').title():<30} {status}")
        
        print("-" * 60)
        print(f"Overall Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("🎉 EXCELLENT: RAG pipeline is working properly!")
            print("✅ Your system is production-ready for clinical decision support")
        elif overall_score >= 60:
            print("⚠️ GOOD: RAG pipeline mostly working with some issues")
            print("🔧 Review failed tests and improve system configuration")
        else:
            print("❌ NEEDS WORK: RAG pipeline has significant issues")
            print("🔧 Major fixes required before deployment")
        
        return self.test_results


if __name__ == "__main__":
    """Run the comprehensive RAG pipeline test"""
    test_suite = RAGPipelineTest()
    results = test_suite.run_comprehensive_test()
    
    # Exit with appropriate code
    overall_score = results.get('overall_score', 0)
    exit_code = 0 if overall_score >= 80 else 1
    sys.exit(exit_code)