"""
Main entry point for MediRag Hierarchical Medical Graph RAG System
"""
import os
import argparse
import requests
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from camel.storages import Neo4jGraph
from utils import *

def check_ollama_status():
    """Check if Ollama is running and has required models"""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    models_to_check = [
        os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ]
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        models_data = response.json()
        
        available_models = [model['name'] for model in models_data.get('models', [])]
        
        print("🤖 Ollama Status Check:")
        print(f"✅ Ollama server running at {ollama_url}")
        
        missing_models = []
        for model in models_to_check:
            if any(model in available for available in available_models):
                print(f"✅ Model {model} available")
            else:
                print(f"❌ Model {model} NOT FOUND")
                missing_models.append(model)
        
        if missing_models:
            print(f"\n🚨 Missing models: {', '.join(missing_models)}")
            print("Please run:")
            for model in missing_models:
                print(f"  ollama pull {model}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server not running")
        print("💡 Start Ollama with: ollama serve")
        print("📖 See OLLAMA_SETUP.md for detailed setup instructions")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='MediRag Hierarchical Medical Graph RAG System')
    parser.add_argument('-setup_hierarchy', action='store_true', 
                       help='Setup 3-level hierarchical database structure')
    parser.add_argument('-test_hybrid', action='store_true', 
                       help='Test hybrid Neo4j + Qdrant system')
    parser.add_argument('-cache_sample_data', action='store_true',
                       help='Cache sample medical data (dictionary + literature)')
    parser.add_argument('-run_inference', action='store_true',
                       help='Run clinical RAG inference with patient injection')
    parser.add_argument('-list_patients', action='store_true',
                       help='List available patient scenarios')
    parser.add_argument('-test_ollama', action='store_true',
                       help='Test Ollama setup and connectivity')
    args = parser.parse_args()

    # Get Neo4j credentials from environment
    url = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")  
    password = os.getenv("NEO4J_PASSWORD")

    if args.setup_hierarchy:
        print("🏗️ Setting up 3-level hierarchical database structure...")
        
        # Check Ollama first
        if not check_ollama_status():
            print("\n⚠️ Ollama setup required. See OLLAMA_SETUP.md for instructions.")
            exit(1)
        try:
            from utils import setup_hierarchical_structure
            hybrid_rag = setup_hierarchical_structure()
            
            # Display statistics
            stats = hybrid_rag.get_level_statistics()
            print("\n📊 Database Structure:")
            for stat in stats:
                print(f"  {stat['level']} Level: {stat['count']} nodes ({stat['node_types']})")
            
            print("\n✅ Hierarchical setup complete!")
            print("💡 Next steps:")
            print("  - Use -test_hybrid to test the system")
            print("  - Access Neo4j browser at: http://localhost:7474")
            print("  - Access Qdrant API at: http://localhost:6333/dashboard")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            print("🔧 Troubleshooting:")
            print("  - Ensure Docker containers are running: docker compose up -d")
            print("  - Check .env file has correct credentials")
            print("  - Verify GOOGLE_API_KEY is set")
            exit(1)
            
    elif args.test_hybrid:
        print("🧪 Testing hybrid Neo4j + Qdrant system...")
        try:
            from utils import HybridMedicalRAG
            hybrid_rag = HybridMedicalRAG()
            
            # Test database connections
            print("✓ Successfully connected to both Neo4j and Qdrant")
            
            # Get system statistics
            stats = hybrid_rag.get_level_statistics()
            print("\n📊 System Statistics:")
            for stat in stats:
                print(f"  {stat['level']} Level: {stat['count']} nodes")
            
            # Test semantic search if available
            try:
                print("\n🔍 Testing semantic search...")
                results = hybrid_rag.semantic_search_across_levels("hypertension treatment")
                print(f"  Found {len(results)} relevant results")
            except Exception as e:
                print(f"  ⚠️  Semantic search test failed: {e}")
            
            print("\n✅ Hybrid system test completed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            exit(1)
    
    elif args.cache_sample_data:
        print("💾 Caching sample medical data...")
        
        # Check Ollama first
        if not check_ollama_status():
            print("\n⚠️ Ollama setup required. See OLLAMA_SETUP.md for instructions.")
            exit(1)
        try:
            from sample_data_loader import setup_sample_data_cache
            result = setup_sample_data_cache()
            print(f"\n✅ Sample data caching completed: {result}")
            
        except Exception as e:
            print(f"❌ Caching failed: {e}")
            exit(1)
    
    elif args.run_inference:
        print("🧠 Running clinical RAG inference...")
        
        # Check Ollama first
        if not check_ollama_status():
            print("\n⚠️ Ollama setup required. See OLLAMA_SETUP.md for instructions.")
            exit(1)
        try:
            from patient_inference import setup_clinical_rag_pipeline
            
            pipeline = setup_clinical_rag_pipeline()
            patients = pipeline.list_available_patients()
            
            print(f"\n👥 Available patients ({len(patients)}):")
            for i, patient in enumerate(patients):
                if 'patient_id' in patient:
                    print(f"  {i}: {patient['patient_id']} - {patient['chief_complaint']}")
                else:
                    print(f"  {i}: {patient['title']} ({patient['scenario_type']})")
            
            # Interactive query
            print("\n🔍 Enter clinical query (or 'exit' to quit):")
            while True:
                query = input("Query: ").strip()
                if query.lower() == 'exit':
                    break
                
                print("Select patient index: ", end="")
                try:
                    patient_idx = int(input().strip())
                    result = pipeline.run_clinical_inference(patient_idx, query)
                    print(f"\n📋 Response:\n{result['response']}\n")
                    print(f"🔍 Sources used: {len(result['sources'])}")
                    print("-" * 50)
                except (ValueError, IndexError) as e:
                    print(f"❌ Invalid input: {e}")
                except Exception as e:
                    print(f"❌ Inference error: {e}")
            
        except Exception as e:
            print(f"❌ Inference setup failed: {e}")
            exit(1)
    
    elif args.list_patients:
        print("👥 Listing available patient scenarios...")
        try:
            import json
            from sample_data_loader import SampleDataLoader
            
            loader = SampleDataLoader()
            patients = loader.load_patient_scenarios()
            
            print(f"\n📊 Found {len(patients)} patient scenarios:\n")
            
            for i, patient in enumerate(patients):
                if 'patient_id' in patient:
                    demo = patient.get('demographics', {})
                    print(f"[{i}] Patient Record: {patient['patient_id']}")
                    print(f"    Demographics: {demo.get('age', 'unknown')} yo {demo.get('gender', 'unknown')}")
                    print(f"    Chief Complaint: {patient.get('chief_complaint', 'N/A')}")
                    print(f"    Diagnosis: {patient.get('diagnosis', 'N/A')}")
                else:
                    print(f"[{i}] Clinical Scenario: {patient.get('scenario_id', f'scenario_{i}')}")
                    print(f"    Title: {patient.get('title', 'Unknown')}")
                    print(f"    Type: {patient.get('scenario_type', 'general')}")
                print()
            
        except Exception as e:
            print(f"❌ Failed to list patients: {e}")
            exit(1)
    
    elif args.test_ollama:
        print("🤖 Testing Ollama setup...")
        
        if check_ollama_status():
            print("\n🧪 Testing LLM generation...")
            try:
                response = call_llm("You are a helpful medical AI assistant.", "What is hypertension?")
                print(f"✅ LLM Response: {response[:100]}...")
                
                print("\n🔍 Testing embedding generation...")
                embedding = get_embedding("hypertension is high blood pressure")
                print(f"✅ Embedding generated: {len(embedding)} dimensions")
                
                print("\n✅ Ollama setup is working correctly!")
                
            except Exception as e:
                print(f"❌ Ollama test failed: {e}")
                print("💡 Check that models are properly loaded")
        else:
            print("❌ Ollama setup incomplete")
            print("📖 See OLLAMA_SETUP.md for setup instructions")
    
    else:
        print("🚀 MediRag Hierarchical Medical Graph RAG System")
        print("=" * 50)
        print()
        print("Available commands:")
        print("  python run.py -test_ollama       : Test Ollama setup and connectivity")
        print("  python run.py -setup_hierarchy   : Setup 3-level medical database")
        print("  python run.py -test_hybrid       : Test hybrid Neo4j + Qdrant system")
        print("  python run.py -cache_sample_data : Cache comprehensive medical data (270+ papers + 13K dictionary)")
        print("  python run.py -run_inference     : Run interactive clinical RAG inference")
        print("  python run.py -list_patients     : List available patient scenarios")
        print()
        print("Workflow:")
        print("  1. python run.py -test_ollama        # Test Ollama setup first")
        print("  2. python run.py -setup_hierarchy    # Setup basic structure")
        print("  3. python run.py -cache_sample_data  # Cache comprehensive medical knowledge")
        print("  4. python run.py -run_inference      # Run evidence-based clinical inference")
        print()
        print("Prerequisites:")
        print("  - Ollama running with llama3.1:8b and nomic-embed-text models")
        print("  - Docker containers running: docker compose up -d")
        print("  - Environment variables set in .env file")
        print()
        print("For Ollama setup, see: OLLAMA_SETUP.md")

if __name__ == "__main__":
    main()
