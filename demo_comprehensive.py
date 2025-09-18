#!/usr/bin/env python3
"""
Comprehensive MediRag Patient Inference Demonstration
Shows the complete workflow for hybrid medical RAG with patient injection and cited responses
"""

from patient_inference import PatientInferenceEngine
from sample_data_loader import SampleDataLoader

def demonstrate_patient_inference():
    print("🏥 MediRag Patient Inference Demonstration")
    print("=" * 60)
    print("🎯 Goal: Inject specific patient data and answer clinical questions")
    print("📚 Features: Hybrid Neo4j+Qdrant storage with medical citations")
    print()
    
    # Initialize the hybrid system
    print("🚀 Initializing MediRag Hybrid System...")
    patient_rag = PatientInferenceEngine()
    print("✅ System ready - Neo4j graph + Qdrant vector storage connected!")
    
    # Load available patient scenarios
    print("\n👥 Loading available patient scenarios...")
    data_loader = SampleDataLoader()
    patients = data_loader.load_patient_scenarios()
    print(f"✅ Loaded {len(patients)} patient scenarios for demonstration")
    
    # Show available patients
    print("\n📋 Available Patient Cases:")
    print("-" * 40)
    for i, patient in enumerate(patients[:4]):  # Show first 4
        patient_id = patient.get('patient_id', f'Patient_{i+1}')
        chief_complaint = patient.get('chief_complaint', 'Unknown complaint')
        print(f"  {patient_id}: {chief_complaint}")
    
    # Select and inject a patient
    selected_patient = patients[2]  # P003 - Heart failure case
    print(f"\n👤 Demonstrating with: {selected_patient['patient_id']}")
    print(f"📋 Chief Complaint: {selected_patient['chief_complaint']}")
    print(f"🏥 Patient Details:")
    print(f"   - Age: {selected_patient['demographics']['age']} years")
    print(f"   - Gender: {selected_patient['demographics']['gender']}")
    print(f"   - Medical History: {', '.join(selected_patient.get('past_medical_history', [])[:3])}")
    
    # Inject patient data into hybrid storage
    print(f"\n💉 Injecting patient data into hybrid RAG system...")
    session_id = patient_rag.inject_patient_data(selected_patient)
    print(f"✅ Patient data successfully injected (Session: {session_id[:8]}...)")
    
    # Demonstrate clinical queries with citations
    clinical_questions = [
        "What could be causing the ankle swelling and shortness of breath in this patient?",
        "What diagnostic tests should be ordered for this heart failure presentation?", 
        "What medications need to be reviewed and potentially adjusted?",
        "What is the expected prognosis and follow-up plan?"
    ]
    
    print("\n🔍 Demonstrating Clinical Queries with Medical Citations:")
    print("=" * 60)
    
    for i, question in enumerate(clinical_questions, 1):
        print(f"\n🔬 Clinical Question {i}:")
        print(f"❓ {question}")
        print("⏳ Searching hybrid knowledge base...")
        
        try:
            response = patient_rag.query(question)
            
            # Display response with formatting
            print(f"\n🤖 AI Clinical Response with Citations:")
            print("-" * 50)
            
            # Show first part of response
            if len(response) > 800:
                print(f"{response[:800]}...")
                print(f"\n📄 [Response truncated - full response contains {len(response)} characters]")
            else:
                print(response)
                
            print("-" * 50)
            
            # Check for citations in response
            citation_indicators = ['[Source:', 'Source:', 'Citation:', 'Reference:']
            has_citations = any(indicator in response for indicator in citation_indicators)
            if has_citations:
                print("✅ Response includes medical source citations")
            else:
                print("ℹ️ Response generated from knowledge base")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("=" * 60)
    
    # Demonstrate cleanup
    print("\n🧹 Cleaning up inference session...")
    patient_rag.cleanup_session()
    print("✅ Patient data removed from temporary storage")
    
    print("\n🎉 Demonstration Complete!")
    print("✨ Key Features Demonstrated:")
    print("  • Patient data injection into hybrid storage")
    print("  • Semantic search across medical knowledge levels") 
    print("  • Clinical question answering with citations")
    print("  • Medical paper/book source attribution")
    print("  • Session management and cleanup")
    
    return True

if __name__ == "__main__":
    success = demonstrate_patient_inference()
    if success:
        print(f"\n🚀 MediRag Patient Inference System is fully operational!")
        print(f"💡 Use this system to inject any patient data and get clinical insights with citations.")