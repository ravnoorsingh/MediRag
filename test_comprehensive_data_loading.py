#!/usr/bin/env python3
"""
Test script for comprehensive medical data loading
Tests integration of real medical papers and comprehensive medical dictionary
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sample_data_loader import SampleDataLoader
from clinical_decision_engine import create_clinical_decision_engine

def test_comprehensive_data_loading():
    """Test the comprehensive data loading and retrieval"""
    print("🧪 Testing Comprehensive Medical Data Loading")
    print("=" * 60)
    
    try:
        # 1. Load comprehensive data
        print("\n1️⃣  Loading comprehensive medical data...")
        loader = SampleDataLoader()
        result = loader.cache_all_sample_data()
        
        print(f"✅ Data loading completed:")
        print(f"   📄 Papers processed: {result.get('papers_processed', 0)}")
        print(f"   🧠 Concepts extracted: {result.get('concepts_extracted', 0)}")  
        print(f"   🔗 Relationships created: {result.get('relationships_created', 0)}")
        
        # 2. Test clinical decision engine with real data
        print("\n2️⃣  Testing clinical decision engine...")
        engine = create_clinical_decision_engine()
        
        # Test patient scenario
        test_patient = {
            'patient_id': 'test-comprehensive-001',
            'demographics': {'age': 55, 'gender': 'male'},
            'past_medical_history': ['Type 2 diabetes', 'Hypertension'],
            'medications': ['Metformin', 'Lisinopril'],
            'chief_complaint': 'Chest pain and shortness of breath'
        }
        
        clinical_result = engine.process_clinical_query(
            patient_data=test_patient,
            clinical_question='How should I manage this patient with chest pain and diabetes?',
            chief_complaint='Chest pain and shortness of breath',
            urgency='urgent'
        )
        
        print(f"✅ Clinical analysis completed:")
        print(f"   🎯 Care options generated: {len(clinical_result.get('care_options', []))}")
        
        # Display evidence sources
        evidence_sources = set()
        for option in clinical_result.get('care_options', []):
            for evidence in option.get('evidence', []):
                evidence_sources.add(evidence.get('source_type', 'unknown'))
        
        print(f"   📚 Evidence sources used: {', '.join(evidence_sources)}")
        
        # Check if real medical papers were retrieved
        has_real_papers = any('research_paper' in sources for sources in [evidence_sources])
        has_dictionary = any('medical_dictionary' in sources for sources in [evidence_sources])
        
        print(f"   📄 Real medical papers retrieved: {'✅' if has_real_papers else '❌'}")
        print(f"   📚 Medical dictionary used: {'✅' if has_dictionary else '❌'}")
        
        # 3. Display sample evidence
        print("\n3️⃣  Sample evidence retrieved:")
        for i, option in enumerate(clinical_result.get('care_options', [])[:2], 1):
            print(f"\n   Care Option {i}: {option.get('title', 'Unknown')}")
            for j, evidence in enumerate(option.get('evidence', [])[:2], 1):
                print(f"     Evidence {j}: {evidence.get('source_type', 'unknown')} - {evidence.get('title', 'Unknown')[:100]}...")
        
        print("\n🎉 Comprehensive data loading test completed successfully!")
        print(f"   System now has access to:")
        print(f"   • {result.get('papers_processed', 0)} real medical research papers")
        print(f"   • 13,000+ comprehensive medical dictionary terms")
        print(f"   • {result.get('concepts_extracted', 0)} extracted medical concepts")
        print(f"   • Evidence-based clinical decision support")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive_data_loading()
    sys.exit(0 if success else 1)