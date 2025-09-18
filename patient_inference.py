"""
Patient Injection System for MediRag
Handles dynamic patient data injection during RAG pipeline inference
"""

import json
import os
from typing import Dict, List, Any, Optional
from utils import HybridMedicalRAG, get_embedding, call_llm, sys_prompt_one, sys_prompt_two
import uuid

class PatientInferenceEngine:
    def __init__(self, hybrid_rag: HybridMedicalRAG = None):
        self.hybrid_rag = hybrid_rag or HybridMedicalRAG()
        self.current_patient = None
        self.inference_session_id = None
    
    def inject_patient_data(self, patient_data: Dict[str, Any]) -> str:
        """Dynamically inject patient data for current inference session"""
        self.inference_session_id = str(uuid.uuid4())
        self.current_patient = patient_data
        
        print(f"👤 Injecting patient data for session: {self.inference_session_id}")
        
        # Create temporary patient node in Neo4j
        patient_id = patient_data.get('patient_id', f"temp_{self.inference_session_id}")
        
        # Extract key patient information for embedding
        patient_text = self._create_patient_embedding_text(patient_data)
        
        # Add patient to Neo4j temporarily
        cypher_query = """
        CREATE (p:Patient:Temporary {
            id: $id,
            session_id: $session_id,
            patient_id: $patient_id,
            patient_summary: $patient_summary,
            level: 'top',
            data_type: 'patient_runtime'
        })
        RETURN p
        """
        
        self.hybrid_rag.n4j.query(cypher_query, {
            'id': self.inference_session_id,
            'session_id': self.inference_session_id,
            'patient_id': patient_id,
            'patient_summary': patient_text
        })
        
        # Add patient to Qdrant temporarily
        embedding = get_embedding(patient_text)
        self.hybrid_rag.add_to_qdrant(
            embedding=embedding,
            payload={
                'id': self.inference_session_id,
                'type': 'patient_runtime',
                'patient_id': patient_id,
                'text': patient_text,
                'level': 'top',
                'data_source': 'patient_injection',
                'session_id': self.inference_session_id
            },
            point_id=self.inference_session_id
        )
        
        print(f"✅ Patient {patient_id} injected successfully")
        return self.inference_session_id
    
    def _create_patient_embedding_text(self, patient_data: Dict[str, Any]) -> str:
        """Create comprehensive text representation of patient for embedding"""
        if 'patient_records' in str(type(patient_data)) or 'demographics' in patient_data:
            return self._format_patient_record(patient_data)
        elif 'scenario_type' in patient_data:
            return self._format_clinical_scenario(patient_data)
        else:
            return str(patient_data)
    
    def _format_patient_record(self, patient_data: Dict[str, Any]) -> str:
        """Format patient record for embedding"""
        components = []
        
        # Demographics
        if 'demographics' in patient_data:
            demo = patient_data['demographics']
            components.append(f"Patient: {demo.get('age', 'unknown')} year old {demo.get('gender', 'unknown')}")
        
        # Chief complaint
        if 'chief_complaint' in patient_data:
            components.append(f"Chief complaint: {patient_data['chief_complaint']}")
        
        # History
        if 'history_present_illness' in patient_data:
            components.append(f"History: {patient_data['history_present_illness']}")
        
        # Past medical history
        if 'past_medical_history' in patient_data:
            pmh = ', '.join(patient_data['past_medical_history'])
            components.append(f"Past medical history: {pmh}")
        
        # Current medications
        if 'medications' in patient_data:
            meds = ', '.join(patient_data['medications'])
            components.append(f"Current medications: {meds}")
        
        # Vital signs
        if 'vital_signs' in patient_data:
            vitals = []
            for key, value in patient_data['vital_signs'].items():
                vitals.append(f"{key}: {value}")
            components.append(f"Vital signs: {', '.join(vitals)}")
        
        # Laboratory results
        if 'laboratory_results' in patient_data:
            labs = []
            for key, value in patient_data['laboratory_results'].items():
                labs.append(f"{key}: {value}")
            components.append(f"Laboratory results: {', '.join(labs)}")
        
        # Diagnosis
        if 'diagnosis' in patient_data:
            components.append(f"Diagnosis: {patient_data['diagnosis']}")
        
        return '. '.join(components)
    
    def _format_clinical_scenario(self, scenario_data: Dict[str, Any]) -> str:
        """Format clinical scenario for embedding"""
        components = []
        
        components.append(f"Clinical scenario: {scenario_data.get('title', 'Unknown scenario')}")
        components.append(f"Type: {scenario_data.get('scenario_type', 'general')}")
        components.append(f"Question: {scenario_data.get('clinical_question', '')}")
        
        if 'patient_summary' in scenario_data:
            summary = scenario_data['patient_summary']
            if isinstance(summary, dict):
                components.append(f"Patient: {summary.get('age', 'unknown')} year old {summary.get('gender', 'unknown')}")
                if 'presentation' in summary:
                    components.append(f"Presentation: {summary['presentation']}")
        
        if 'key_findings' in scenario_data:
            findings = ', '.join(scenario_data['key_findings'])
            components.append(f"Key findings: {findings}")
        
        return '. '.join(components)
    
    def run_inference(self, clinical_query: str, top_k: int = 10) -> Dict[str, Any]:
        """Run RAG inference with injected patient data"""
        if not self.current_patient or not self.inference_session_id:
            raise ValueError("No patient data injected. Call inject_patient_data() first.")
        
        print(f"🧠 Running RAG inference for query: '{clinical_query}'")
        
        # Step 1: Semantic search across all levels including injected patient
        search_results = self.hybrid_rag.semantic_search_across_levels(clinical_query, top_k=top_k)
        
        # Step 2: Filter and prioritize results
        prioritized_results = self._prioritize_search_results(search_results, clinical_query)
        
        # Step 3: Retrieve relevant graph connections
        graph_context = self._get_graph_context(clinical_query)
        
        # Step 4: Generate response using LLM
        response = self._generate_clinical_response(clinical_query, prioritized_results, graph_context)
        
        return {
            'query': clinical_query,
            'patient_id': self.current_patient.get('patient_id', 'unknown'),
            'session_id': self.inference_session_id,
            'response': response,
            'sources': prioritized_results,
            'graph_context': graph_context
        }
    
    def _prioritize_search_results(self, search_results: List[Dict], query: str) -> List[Dict]:
        """Prioritize search results based on relevance and data source"""
        # Separate results by source
        patient_results = []
        literature_results = []
        dictionary_results = []
        
        for result in search_results:
            payload = result.get('payload', {})
            data_source = payload.get('data_source', 'unknown')
            
            if 'patient' in data_source:
                patient_results.append(result)
            elif 'literature' in data_source:
                literature_results.append(result)
            elif 'dictionary' in data_source:
                dictionary_results.append(result)
        
        # Prioritize: Patient context first, then literature, then dictionary
        prioritized = patient_results + literature_results[:5] + dictionary_results[:3]
        return prioritized[:10]  # Limit to top 10 results
    
    def _get_graph_context(self, query: str) -> List[Dict]:
        """Get relevant graph relationships and connections"""
        # Query Neo4j for relevant relationships
        graph_query = """
        MATCH (n)-[r]->(m)
        WHERE n.level IN ['bottom', 'middle', 'top']
        AND (toLower(n.content) CONTAINS toLower($query) 
             OR toLower(m.content) CONTAINS toLower($query)
             OR toLower(type(r)) CONTAINS toLower($query))
        RETURN n, r, m
        LIMIT 20
        """
        
        try:
            results = self.hybrid_rag.n4j.query(graph_query, {'query': query})
            return results
        except Exception as e:
            print(f"Warning: Graph context query failed: {e}")
            return []
    
    def _generate_clinical_response(self, query: str, search_results: List[Dict], graph_context: List[Dict]) -> str:
        """Generate clinical response using LLM with retrieved context"""
        # Prepare context from search results
        context_parts = []
        
        # Add patient context
        patient_contexts = [r for r in search_results if 'patient' in r.get('payload', {}).get('data_source', '')]
        if patient_contexts:
            context_parts.append("CURRENT PATIENT CONTEXT:")
            for ctx in patient_contexts[:2]:  # Limit to avoid token overflow
                context_parts.append(f"- {ctx.get('payload', {}).get('text', '')}")
        
        # Add literature evidence
        literature_contexts = [r for r in search_results if 'literature' in r.get('payload', {}).get('data_source', '')]
        if literature_contexts:
            context_parts.append("\nRELEVANT MEDICAL LITERATURE:")
            for i, ctx in enumerate(literature_contexts[:3]):
                context_parts.append(f"[{i+1}] {ctx.get('payload', {}).get('text', '')}")
        
        # Add medical knowledge
        dictionary_contexts = [r for r in search_results if 'dictionary' in r.get('payload', {}).get('data_source', '')]
        if dictionary_contexts:
            context_parts.append("\nMEDICAL KNOWLEDGE BASE:")
            for ctx in dictionary_contexts[:2]:
                context_parts.append(f"- {ctx.get('payload', {}).get('text', '')}")
        
        # Prepare full context
        full_context = '\n'.join(context_parts)
        
        # Generate response with two-step prompting
        try:
            # Step 1: Initial response with graph data
            initial_response = call_llm(sys_prompt_one, f"Context: {full_context}\n\nQuestion: {query}")
            
            # Step 2: Enhanced response with citations
            final_response = call_llm(
                sys_prompt_two, 
                f"Original response: {initial_response}\n\nReferences: {full_context}\n\nQuestion: {query}"
            )
            
            return final_response
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return f"Unable to generate response due to: {e}"
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Generate LLM response using the call_llm function"""
        try:
            system_prompt = """You are a medical AI assistant with access to comprehensive medical literature and patient data. 
Provide accurate, evidence-based responses with proper citations from medical sources."""
            
            response = call_llm(system_prompt, prompt)
            return response
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return f"Unable to generate response due to: {e}"
    
    def query(self, query_text: str) -> str:
        """
        Query the hybrid RAG system with patient context
        Returns response with medical source citations
        """
        if not self.current_patient:
            return "❌ No patient data injected. Please inject patient data first."
        
        # Perform semantic search across all levels
        search_results = self.hybrid_rag.semantic_search_across_levels(
            query_text=query_text,
            level_filter=None,  # Search all levels 
            top_k=5
        )
        
        # Extract relevant information
        context_pieces = []
        
        # Process vector results
        for result in search_results['vector_results']:
            payload = result.payload
            score = result.score
            context_pieces.append({
                'content': payload.get('content', ''),
                'source': payload.get('node_id', 'unknown'),
                'level': payload.get('level', 'unknown'),
                'type': payload.get('node_type', 'unknown'),
                'score': score
            })
        
        # Process graph results  
        for result in search_results['graph_results']:
            node_data = result.get('n', {})
            if hasattr(node_data, 'get'):
                context_pieces.append({
                    'content': node_data.get('content', ''),
                    'source': node_data.get('id', node_data.get('cui', 'unknown')),
                    'level': node_data.get('level', 'unknown'),
                    'type': 'graph_node',
                    'score': 0.8  # Default score for graph results
                })
        
        # Create context for LLM
        context = self._build_context_with_citations(context_pieces, query_text)
        
        # Generate response using LLM with patient context
        patient_summary = f"Patient context: {self._create_patient_embedding_text(self.current_patient)}"
        
        prompt = f"""
{patient_summary}

Medical Knowledge Context:
{context}

Clinical Question: {query_text}

Please provide a comprehensive answer based on the patient's condition and the available medical literature. 
Include specific citations from medical papers and books when making recommendations.
Format citations as [Source: paper/book title or ID].
"""
        
        response = self._generate_llm_response(prompt)
        return response
    
    def _build_context_with_citations(self, context_pieces, query_text):
        """Build context string with proper citations"""
        if not context_pieces:
            return "No relevant medical literature found."
        
        # Sort by relevance score
        context_pieces.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        context_parts = []
        seen_sources = set()
        
        for i, piece in enumerate(context_pieces[:5]):  # Top 5 results
            source = piece['source']
            if source not in seen_sources:
                seen_sources.add(source)
                level_desc = {
                    'TOP': 'Patient Record',
                    'MIDDLE': 'Medical Literature',
                    'BOTTOM': 'Medical Concept'
                }.get(piece['level'], 'Medical Source')
                
                context_parts.append(f"[{level_desc} - {source}]: {piece['content'][:300]}...")
        
        return '\n\n'.join(context_parts)
    
    def cleanup_session(self):
        """Clean up temporary patient data after inference"""
        if self.inference_session_id:
            print(f"🧹 Cleaning up inference session: {self.inference_session_id}")
            
            # Remove temporary patient data from Neo4j
            cleanup_query = """
            MATCH (n:Temporary {session_id: $session_id})
            DETACH DELETE n
            """
            self.hybrid_rag.n4j.query(cleanup_query, {'session_id': self.inference_session_id})
            
            # Remove from Qdrant (Qdrant will handle point deletion)
            try:
                self.hybrid_rag.qdrant_client.delete(
                    collection_name=self.hybrid_rag.collection_name,
                    points_selector=[self.inference_session_id]
                )
            except Exception as e:
                print(f"Warning: Could not remove point from Qdrant: {e}")
            
            print("✅ Session cleanup completed")
            
        # Reset state
        self.current_patient = None
        self.inference_session_id = None

class ClinicalRAGPipeline:
    """Main pipeline combining caching and inference"""
    
    def __init__(self):
        self.hybrid_rag = HybridMedicalRAG()
        self.inference_engine = PatientInferenceEngine(self.hybrid_rag)
        self.patient_scenarios = []
    
    def initialize_with_sample_data(self):
        """Initialize the pipeline with cached sample data"""
        from sample_data_loader import SampleDataLoader
        
        print("🚀 Initializing Clinical RAG Pipeline...")
        
        # Load and cache medical knowledge
        loader = SampleDataLoader()
        loader.initialize_hybrid_rag()
        cache_result = loader.cache_all_sample_data()
        
        # Load patient scenarios for runtime injection
        self.patient_scenarios = loader.load_patient_scenarios()
        
        print("✅ Pipeline initialization complete!")
        return cache_result
    
    def run_clinical_inference(self, patient_index: int, clinical_query: str) -> Dict[str, Any]:
        """Run complete clinical inference with patient injection"""
        if patient_index >= len(self.patient_scenarios):
            raise ValueError(f"Patient index {patient_index} out of range. Available: 0-{len(self.patient_scenarios)-1}")
        
        patient_data = self.patient_scenarios[patient_index]
        
        try:
            # Inject patient data
            session_id = self.inference_engine.inject_patient_data(patient_data)
            
            # Run inference
            result = self.inference_engine.run_inference(clinical_query)
            
            return result
            
        finally:
            # Always cleanup
            self.inference_engine.cleanup_session()
    
    def list_available_patients(self) -> List[Dict[str, Any]]:
        """List available patient scenarios"""
        patient_summaries = []
        for i, patient in enumerate(self.patient_scenarios):
            if 'patient_id' in patient:
                summary = {
                    'index': i,
                    'patient_id': patient['patient_id'],
                    'type': 'patient_record',
                    'demographics': patient.get('demographics', {}),
                    'chief_complaint': patient.get('chief_complaint', 'N/A')
                }
            else:
                summary = {
                    'index': i,
                    'scenario_id': patient.get('scenario_id', f'scenario_{i}'),
                    'type': 'clinical_scenario',
                    'title': patient.get('title', 'Unknown scenario'),
                    'scenario_type': patient.get('scenario_type', 'general')
                }
            patient_summaries.append(summary)
        
        return patient_summaries

# Utility functions for easy import
def setup_clinical_rag_pipeline():
    """Setup complete clinical RAG pipeline"""
    pipeline = ClinicalRAGPipeline()
    pipeline.initialize_with_sample_data()
    return pipeline

if __name__ == "__main__":
    # Example usage
    pipeline = ClinicalRAGPipeline()
    pipeline.initialize_with_sample_data()
    
    # List available patients
    patients = pipeline.list_available_patients()
    print(f"\n👥 Available patients: {len(patients)}")
    
    for patient in patients[:3]:  # Show first 3
        print(f"  - Index {patient['index']}: {patient}")
    
    # Run sample inference
    if patients:
        print("\n🧠 Running sample inference...")
        result = pipeline.run_clinical_inference(0, "What is the most likely diagnosis and recommended treatment?")
        print(f"Result: {result['response'][:200]}...")