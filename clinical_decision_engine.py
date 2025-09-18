"""
Clinical Decision Support System (CDSS)
AI-powered clinical agent for evidence-based medical decision making

This module implements the core clinical decision engine that:
1. Processes FHIR patient records
2. Extracts relevant clinical data based on doctor's query
3. Provides 3 evidence-based care options with citations
4. Presents explainable, actionable clinical insights
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils import HybridMedicalRAG, get_embedding, call_llm
from sample_data_loader import SampleDataLoader


class ConfidenceLevel(Enum):
    """Confidence levels for clinical recommendations"""
    HIGH = "HIGH"      # >0.8 - Strong evidence
    MEDIUM = "MEDIUM"  # 0.6-0.8 - Moderate evidence  
    LOW = "LOW"        # <0.6 - Limited evidence


@dataclass
class ClinicalEvidence:
    """Structure for medical evidence with citations"""
    source_id: str
    source_type: str  # 'research_paper', 'guideline', 'textbook'
    title: str
    excerpt: str
    confidence_score: float
    publication_year: Optional[int] = None
    authors: Optional[str] = None


@dataclass
class CareOption:
    """Structure for evidence-based care recommendations"""
    option_id: str
    title: str
    description: str
    rationale: str
    confidence: ConfidenceLevel
    evidence: List[ClinicalEvidence]
    contraindications: List[str]
    monitoring_requirements: List[str]
    expected_outcomes: str


@dataclass
class ClinicalQuery:
    """Structure for doctor's clinical question"""
    query_id: str
    patient_id: str
    chief_complaint: str
    specific_question: str
    urgency_level: str  # 'urgent', 'routine', 'followup'
    timestamp: datetime


@dataclass
class PatientContext:
    """Extracted relevant patient data"""
    patient_id: str
    demographics: Dict[str, Any]
    relevant_history: List[str]
    current_medications: List[str]
    vital_signs: Dict[str, Any]
    lab_results: Dict[str, Any]
    allergies: List[str]
    risk_factors: List[str]


class ClinicalDecisionEngine:
    """
    Core AI-powered clinical decision support engine
    
    Implements the 'clinical copilot' that transforms complex health records
    into trustworthy, evidence-backed decisions at the point of care.
    """
    
    def __init__(self):
        """Initialize the clinical decision engine"""
        self.hybrid_rag = HybridMedicalRAG()
        self.session_id = str(uuid.uuid4())
        print("🏥 Clinical Decision Support System initialized")
        print(f"📋 Session ID: {self.session_id}")
    
    def process_clinical_query(
        self, 
        patient_data: Dict[str, Any],
        clinical_question: str,
        chief_complaint: str,
        urgency: str = "routine"
    ) -> Dict[str, Any]:
        """
        Main entry point for clinical decision support
        
        Args:
            patient_data: FHIR-formatted patient record
            clinical_question: Doctor's specific query
            chief_complaint: Patient's primary concern
            urgency: Priority level ('urgent', 'routine', 'followup')
            
        Returns:
            Comprehensive clinical decision support response
        """
        print(f"🔍 Processing clinical query: {clinical_question}")
        
        # 1. Create clinical query object
        query = ClinicalQuery(
            query_id=str(uuid.uuid4()),
            patient_id=patient_data.get('id', 'unknown'),
            chief_complaint=chief_complaint,
            specific_question=clinical_question,
            urgency_level=urgency,
            timestamp=datetime.now()
        )
        
        # 2. Extract relevant patient context
        patient_context = self._extract_patient_context(patient_data, clinical_question)
        
        # 3. Perform intelligent medical retrieval
        relevant_evidence = self._retrieve_relevant_evidence(
            patient_context, clinical_question, chief_complaint
        )
        
        # 4. Generate evidence-based care options
        care_options = self._generate_care_options(
            patient_context, clinical_question, relevant_evidence
        )
        
        # 5. Compile comprehensive response
        response = {
            "query_info": {
                "query_id": query.query_id,
                "patient_id": query.patient_id,
                "timestamp": query.timestamp.isoformat(),
                "urgency": query.urgency_level
            },
            "patient_summary": self._create_patient_summary(patient_context),
            "clinical_question": clinical_question,
            "care_options": [self._serialize_care_option(option) for option in care_options],
            "risk_assessment": self._assess_clinical_risks(patient_context, care_options),
            "evidence_quality": self._assess_evidence_quality(relevant_evidence),
            "follow_up_recommendations": self._generate_followup_plan(patient_context, care_options)
        }
        
        print(f"✅ Generated {len(care_options)} evidence-based care options")
        return response
    
    def _extract_patient_context(
        self, 
        patient_data: Dict[str, Any], 
        clinical_question: str
    ) -> PatientContext:
        """Extract and filter relevant patient data based on clinical question"""
        print("📊 Extracting relevant patient context...")
        
        # Parse FHIR or standard patient data
        demographics = patient_data.get('demographics', {})
        
        # Use AI to identify relevant history based on the question
        full_history = patient_data.get('past_medical_history', [])
        relevant_history = self._filter_relevant_history(full_history, clinical_question)
        
        # Extract current clinical state
        medications = patient_data.get('medications', [])
        vitals = patient_data.get('vital_signs', {})
        labs = patient_data.get('laboratory_results', {})
        allergies = patient_data.get('allergies', [])
        
        # Identify risk factors relevant to the query
        risk_factors = self._identify_risk_factors(patient_data, clinical_question)
        
        context = PatientContext(
            patient_id=patient_data.get('patient_id', patient_data.get('id', 'unknown')),
            demographics=demographics,
            relevant_history=relevant_history,
            current_medications=medications,
            vital_signs=vitals,
            lab_results=labs,
            allergies=allergies,
            risk_factors=risk_factors
        )
        
        print(f"✅ Extracted context: {len(relevant_history)} relevant conditions, {len(medications)} medications")
        return context
    
    def _filter_relevant_history(
        self, 
        full_history: List[str], 
        clinical_question: str
    ) -> List[str]:
        """Use AI to filter medical history relevant to the clinical question"""
        if not full_history:
            return []
        
        # Create a prompt to identify relevant history
        history_text = "; ".join(full_history)
        prompt = f"""
        Given this patient's medical history: {history_text}
        
        Clinical question: {clinical_question}
        
        Identify only the medical conditions, procedures, or treatments that are directly relevant 
        to answering the clinical question. Return as a list separated by semicolons.
        If no conditions are relevant, return "None relevant".
        """
        
        try:
            response = call_llm(
                "You are a clinical assistant that identifies relevant medical history.",
                prompt
            )
            
            if "None relevant" in response:
                return []
            
            # Parse the response into individual items
            relevant_items = [item.strip() for item in response.split(';') if item.strip()]
            return relevant_items[:10]  # Limit to top 10 most relevant
            
        except Exception as e:
            print(f"Warning: Could not filter history: {e}")
            return full_history[:5]  # Fallback to first 5 items
    
    def _identify_risk_factors(
        self, 
        patient_data: Dict[str, Any], 
        clinical_question: str
    ) -> List[str]:
        """Identify patient-specific risk factors relevant to the clinical question"""
        risk_factors = []
        
        # Age-based risks
        age = patient_data.get('demographics', {}).get('age', 0)
        if age >= 65:
            risk_factors.append("Advanced age (≥65 years)")
        elif age <= 18:
            risk_factors.append("Pediatric patient")
        
        # Comorbidity-based risks
        history = patient_data.get('past_medical_history', [])
        high_risk_conditions = [
            'diabetes', 'hypertension', 'heart failure', 'kidney disease',
            'liver disease', 'cancer', 'immunocompromised'
        ]
        
        for condition in history:
            condition_lower = condition.lower()
            for risk_condition in high_risk_conditions:
                if risk_condition in condition_lower:
                    risk_factors.append(f"History of {condition}")
        
        # Medication-based risks
        medications = patient_data.get('medications', [])
        high_risk_meds = ['warfarin', 'chemotherapy', 'immunosuppressant']
        for med in medications:
            med_lower = med.lower() if isinstance(med, str) else str(med).lower()
            for risk_med in high_risk_meds:
                if risk_med in med_lower:
                    risk_factors.append(f"High-risk medication: {med}")
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _retrieve_relevant_evidence(
        self,
        patient_context: PatientContext,
        clinical_question: str,
        chief_complaint: str
    ) -> List[ClinicalEvidence]:
        """Use hybrid retrieval to find relevant medical evidence"""
        print("🔍 Retrieving relevant medical evidence...")
        
        # Create comprehensive search context including patient-specific information
        search_queries = []
        
        # Primary query focused on chief complaint and question
        primary_query = f"""
        Chief complaint: {chief_complaint}
        Clinical question: {clinical_question}
        Patient age: {patient_context.demographics.get('age', 'unknown')}
        Patient gender: {patient_context.demographics.get('gender', 'unknown')}
        """
        search_queries.append(primary_query)
        
        # Secondary queries for specific patient context
        if patient_context.relevant_history:
            history_query = f"Medical history: {'; '.join(patient_context.relevant_history[:5])} related to {clinical_question}"
            search_queries.append(history_query)
        
        if patient_context.current_medications:
            medication_query = f"Medications: {'; '.join(patient_context.current_medications[:5])} drug interactions {clinical_question}"
            search_queries.append(medication_query)
        
        if patient_context.risk_factors:
            risk_query = f"Risk factors: {'; '.join(patient_context.risk_factors[:3])} clinical management {clinical_question}"
            search_queries.append(risk_query)
        
        evidence_list = []
        
        # Search across different levels with multiple queries
        for query_text in search_queries:
            print(f"🔎 Searching with: {query_text[:80]}...")
            
            # Search medical literature and concepts (MIDDLE and BOTTOM levels)
            search_results = self.hybrid_rag.semantic_search_across_levels(
                query_text=query_text,
                level_filter=["MIDDLE", "BOTTOM"],  # Focus on literature and medical concepts
                top_k=8
            )
            
            # Process vector search results from Qdrant
            for result in search_results['vector_results']:
                if hasattr(result, 'payload'):
                    payload = result.payload
                    confidence = float(result.score) if hasattr(result, 'score') else 0.8
                    
                    evidence = ClinicalEvidence(
                        source_id=payload.get('node_id', str(uuid.uuid4())),
                        source_type=self._determine_source_type(payload.get('node_type', 'unknown')),
                        title=payload.get('title', payload.get('content', 'Unknown source')[:100]),
                        excerpt=self._create_evidence_excerpt(payload.get('content', ''), query_text),
                        confidence_score=confidence,
                        publication_year=payload.get('year'),
                        authors=payload.get('authors')
                    )
                    evidence_list.append(evidence)
            
            # Process graph search results from Neo4j
            for result in search_results['graph_results']:
                node_data = result.get('n', {})
                similarity = result.get('similarity', 0.7)
                
                if hasattr(node_data, 'get') or isinstance(node_data, dict):
                    node_dict = node_data if isinstance(node_data, dict) else dict(node_data)
                    
                    evidence = ClinicalEvidence(
                        source_id=node_dict.get('id', node_dict.get('cui', str(uuid.uuid4()))),
                        source_type=self._determine_graph_source_type(node_dict),
                        title=self._extract_node_title(node_dict),
                        excerpt=self._create_evidence_excerpt(node_dict.get('content', ''), query_text),
                        confidence_score=float(similarity),
                        publication_year=node_dict.get('year'),
                        authors=node_dict.get('authors')
                    )
                    evidence_list.append(evidence)
        
        # Remove duplicates and sort by confidence
        seen_sources = set()
        unique_evidence = []
        for evidence in evidence_list:
            source_key = f"{evidence.source_id}_{evidence.title[:50]}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_evidence.append(evidence)
        
        unique_evidence.sort(key=lambda x: x.confidence_score, reverse=True)
        final_evidence = unique_evidence[:15]  # Return top 15 pieces of evidence
        
        print(f"✅ Retrieved {len(final_evidence)} unique pieces of medical evidence")
        return final_evidence
    
    def _create_evidence_excerpt(self, content: str, query_context: str) -> str:
        """Create a relevant excerpt from content based on query context"""
        if not content:
            return "No content available"
        
        # If content is short, return as is
        if len(content) <= 300:
            return content
        
        # Try to find relevant sections based on query keywords
        query_keywords = query_context.lower().split()
        content_lower = content.lower()
        
        # Find best starting position based on keyword matches
        best_pos = 0
        max_matches = 0
        
        for i in range(0, len(content) - 300, 50):
            excerpt = content[i:i+300].lower()
            matches = sum(1 for keyword in query_keywords if keyword in excerpt)
            if matches > max_matches:
                max_matches = matches
                best_pos = i
        
        excerpt = content[best_pos:best_pos+300]
        return excerpt + "..." if best_pos + 300 < len(content) else excerpt
    
    def _extract_node_title(self, node_dict: dict) -> str:
        """Extract a meaningful title from node data"""
        # Try different title fields
        for field in ['title', 'name', 'preferred_term', 'content']:
            if field in node_dict and node_dict[field]:
                title = str(node_dict[field])[:100]
                return title if title else "Medical concept"
        
        return "Medical concept"
    
    def _determine_graph_source_type(self, node_dict: dict) -> str:
        """Determine source type from Neo4j node data"""
        # Check node labels if available
        labels = node_dict.get('labels', [])
        if isinstance(labels, list):
            for label in labels:
                if 'paper' in label.lower() or 'research' in label.lower():
                    return 'research_paper'
                elif 'book' in label.lower() or 'textbook' in label.lower():
                    return 'textbook'
                elif 'guideline' in label.lower():
                    return 'clinical_guideline'
                elif 'concept' in label.lower():
                    return 'medical_concept'
        
        # Fallback to level-based determination
        level = node_dict.get('level', '')
        if level == 'MIDDLE':
            return 'medical_literature'
        elif level == 'BOTTOM':
            return 'medical_concept'
        else:
            return 'unknown'
    
    def _determine_source_type(self, node_type: str) -> str:
        """Map node types to standardized source types"""
        type_mapping = {
            'paper': 'research_paper',
            'research_paper': 'research_paper',
            'book': 'textbook',
            'textbook': 'textbook',
            'guideline': 'clinical_guideline',
            'clinical_guideline': 'clinical_guideline',
            'concept': 'medical_concept'
        }
        return type_mapping.get(node_type.lower(), 'unknown')
    
    def _generate_care_options(
        self,
        patient_context: PatientContext,
        clinical_question: str,
        evidence: List[ClinicalEvidence]
    ) -> List[CareOption]:
        """Generate 3 evidence-based care options using AI"""
        print("💡 Generating evidence-based care options...")
        
        # Prepare context for AI
        patient_summary = f"""
        Patient: {patient_context.demographics.get('age', 'unknown')} year old {patient_context.demographics.get('gender', 'unknown')}
        Relevant conditions: {'; '.join(patient_context.relevant_history)}
        Current medications: {'; '.join(patient_context.current_medications)}
        Risk factors: {'; '.join(patient_context.risk_factors)}
        Allergies: {'; '.join(patient_context.allergies)}
        """
        
        # Prepare evidence context with numbered references
        evidence_context = ""
        evidence_index_map = {}
        
        for i, ev in enumerate(evidence[:15]):  # Top 15 pieces of evidence
            evidence_index = i + 1
            evidence_index_map[evidence_index] = ev
            
            # Create detailed evidence entry
            source_info = f"{ev.source_type.upper()}"
            if ev.publication_year:
                source_info += f" ({ev.publication_year})"
            if ev.authors:
                source_info += f" by {ev.authors}"
            
            evidence_context += f"[{evidence_index}] {source_info}: {ev.title}\n"
            evidence_context += f"    {ev.excerpt}\n"
            evidence_context += f"    Confidence: {ev.confidence_score:.2f}\n\n"
        
        # AI prompt for generating care options
        prompt = f"""
        As a clinical decision support system, provide exactly 3 evidence-based care options for this patient.

        PATIENT CONTEXT:
        {patient_summary}

        CLINICAL QUESTION: {clinical_question}

        AVAILABLE EVIDENCE:
        {evidence_context}

        For each care option, provide:
        1. Option title (concise treatment approach)
        2. Detailed description of the intervention
        3. Clinical rationale with specific evidence citations [1], [2], etc.
        4. Confidence level (HIGH/MEDIUM/LOW based on evidence strength)
        5. Contraindications to consider
        6. Monitoring requirements
        7. Expected outcomes

        Format as JSON array with these exact fields:
        [{{
            "title": "Treatment Option Title",
            "description": "Detailed intervention description",
            "rationale": "Clinical reasoning with evidence citations [1][2]",
            "confidence": "HIGH|MEDIUM|LOW",
            "contraindications": ["contraindication 1", "contraindication 2"],
            "monitoring": ["monitoring requirement 1", "monitoring requirement 2"],
            "outcomes": "Expected clinical outcomes"
        }}]

        Ensure recommendations are clinically appropriate and evidence-based.
        """
        
        try:
            response = call_llm(
                "You are an expert clinical decision support system providing evidence-based care recommendations.",
                prompt
            )
            
            # Parse AI response
            care_options_data = self._parse_care_options_response(response)
            
            # Convert to CareOption objects
            care_options = []
            for i, option_data in enumerate(care_options_data):
                # Assign relevant evidence to each option based on citations
                option_evidence = self._assign_evidence_to_option(
                    evidence_index_map, 
                    option_data.get('rationale', '')
                )
                
                # Ensure contraindications and monitoring are arrays
                contraindications = option_data.get('contraindications', [])
                if isinstance(contraindications, str):
                    contraindications = [contraindications] if contraindications else []
                elif not isinstance(contraindications, list):
                    contraindications = []
                
                monitoring = option_data.get('monitoring', [])
                if isinstance(monitoring, str):
                    monitoring = [monitoring] if monitoring else []
                elif not isinstance(monitoring, list):
                    monitoring = []
                
                care_option = CareOption(
                    option_id=str(uuid.uuid4()),
                    title=option_data.get('title', f'Care Option {i+1}'),
                    description=option_data.get('description', ''),
                    rationale=option_data.get('rationale', ''),
                    confidence=ConfidenceLevel(option_data.get('confidence', 'MEDIUM')),
                    evidence=option_evidence,
                    contraindications=contraindications,
                    monitoring_requirements=monitoring,
                    expected_outcomes=option_data.get('outcomes', '')
                )
                care_options.append(care_option)
            
            print(f"✅ Generated {len(care_options)} care options with evidence citations")
            return care_options
            
        except Exception as e:
            print(f"Error generating care options: {e}")
            return self._generate_fallback_care_options(patient_context, clinical_question, evidence)
    
    def _parse_care_options_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured care options"""
        try:
            # Try to find JSON array in the response
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON array found")
                
        except Exception as e:
            print(f"Warning: Could not parse AI response as JSON: {e}")
            # Fallback parsing
            return self._fallback_parse_care_options(response)
    
    def _fallback_parse_care_options(self, response: str) -> List[Dict[str, Any]]:
        """Fallback parser for care options when JSON parsing fails"""
        # Simple fallback - create 3 basic options
        return [
            {
                "title": "Primary Treatment Approach",
                "description": "Standard evidence-based treatment protocol",
                "rationale": "Based on current clinical guidelines and patient presentation",
                "confidence": "MEDIUM",
                "contraindications": ["Patient allergies", "Drug interactions"],
                "monitoring": ["Regular follow-up", "Monitor for adverse effects"],
                "outcomes": "Expected improvement with standard care"
            },
            {
                "title": "Alternative Treatment Option",
                "description": "Alternative therapeutic approach",
                "rationale": "Suitable for patients who cannot tolerate first-line therapy",
                "confidence": "MEDIUM",
                "contraindications": ["Specific patient factors"],
                "monitoring": ["Close monitoring required", "Regular assessments"],
                "outcomes": "Good clinical response expected"
            },
            {
                "title": "Conservative Management",
                "description": "Non-pharmacological supportive care",
                "rationale": "Appropriate for stable patients with minimal symptoms",
                "confidence": "LOW",
                "contraindications": ["Severe symptoms", "High-risk patients"],
                "monitoring": ["Symptom monitoring", "Regular check-ups"],
                "outcomes": "Gradual improvement expected"
            }
        ]
    
    def _assign_evidence_to_option(
        self, 
        evidence_index_map: dict, 
        rationale: str
    ) -> List[ClinicalEvidence]:
        """Assign relevant evidence pieces to specific care options based on citations"""
        assigned_evidence = []
        
        # Find evidence references in rationale (e.g., [1], [2], [3])
        import re
        citations = re.findall(r'\[(\d+)\]', rationale)
        
        for citation in citations:
            try:
                index = int(citation)
                if index in evidence_index_map:
                    assigned_evidence.append(evidence_index_map[index])
            except (ValueError, KeyError):
                continue
        
        # If no specific citations found, assign top 3 pieces of evidence from the map
        if not assigned_evidence:
            # Get first 3 evidence pieces from the index map
            sorted_indices = sorted(evidence_index_map.keys())
            for i in sorted_indices[:3]:
                assigned_evidence.append(evidence_index_map[i])
        
        return assigned_evidence
    
    def _generate_fallback_care_options(
        self,
        patient_context: PatientContext,
        clinical_question: str,
        evidence: List[ClinicalEvidence]
    ) -> List[CareOption]:
        """Generate basic care options when AI generation fails"""
        print("⚠️ Generating fallback care options")
        
        fallback_options = []
        
        for i in range(3):
            option = CareOption(
                option_id=str(uuid.uuid4()),
                title=f"Care Option {i+1}",
                description=f"Evidence-based treatment approach {i+1} for the clinical question",
                rationale="Based on available clinical evidence and patient presentation",
                confidence=ConfidenceLevel.MEDIUM,
                evidence=evidence[:2] if evidence else [],
                contraindications=["Patient-specific contraindications"],
                monitoring_requirements=["Regular clinical monitoring"],
                expected_outcomes="Monitor patient response and adjust treatment as needed"
            )
            fallback_options.append(option)
        
        return fallback_options
    
    def _create_patient_summary(self, patient_context: PatientContext) -> Dict[str, Any]:
        """Create a concise patient summary for clinical decision making"""
        return {
            "patient_id": patient_context.patient_id,
            "demographics": patient_context.demographics,
            "key_conditions": patient_context.relevant_history[:5],
            "current_medications": patient_context.current_medications[:10],
            "risk_factors": patient_context.risk_factors,
            "allergies": patient_context.allergies,
            "vital_signs": patient_context.vital_signs
        }
    
    def _serialize_care_option(self, care_option: CareOption) -> Dict[str, Any]:
        """Convert CareOption object to dictionary for JSON serialization"""
        return {
            "option_id": care_option.option_id,
            "title": care_option.title,
            "description": care_option.description,
            "rationale": care_option.rationale,
            "confidence": care_option.confidence.value,
            "evidence": [
                {
                    "source_id": ev.source_id,
                    "source_type": ev.source_type,
                    "title": ev.title,
                    "excerpt": ev.excerpt,
                    "confidence_score": ev.confidence_score,
                    "publication_year": ev.publication_year,
                    "authors": ev.authors
                }
                for ev in care_option.evidence
            ],
            "contraindications": care_option.contraindications,
            "monitoring_requirements": care_option.monitoring_requirements,
            "expected_outcomes": care_option.expected_outcomes
        }
    
    def _assess_clinical_risks(
        self,
        patient_context: PatientContext,
        care_options: List[CareOption]
    ) -> Dict[str, Any]:
        """Assess clinical risks for the patient and proposed treatments"""
        return {
            "patient_risk_factors": patient_context.risk_factors,
            "high_risk_indicators": [
                factor for factor in patient_context.risk_factors 
                if any(high_risk in factor.lower() 
                      for high_risk in ['age', 'kidney', 'heart', 'liver', 'cancer'])
            ],
            "treatment_risks": [
                {
                    "option_id": option.option_id,
                    "risk_level": option.confidence.value,
                    "contraindications": option.contraindications
                }
                for option in care_options
            ]
        }
    
    def _assess_evidence_quality(self, evidence: List[ClinicalEvidence]) -> Dict[str, Any]:
        """Assess the quality and strength of available evidence"""
        if not evidence:
            return {"overall_quality": "INSUFFICIENT", "evidence_count": 0}
        
        avg_confidence = sum(ev.confidence_score for ev in evidence) / len(evidence)
        
        source_types = {}
        for ev in evidence:
            source_types[ev.source_type] = source_types.get(ev.source_type, 0) + 1
        
        quality_level = "HIGH" if avg_confidence > 0.8 else "MEDIUM" if avg_confidence > 0.6 else "LOW"
        
        return {
            "overall_quality": quality_level,
            "evidence_count": len(evidence),
            "average_confidence": round(avg_confidence, 2),
            "source_distribution": source_types,
            "high_confidence_sources": len([ev for ev in evidence if ev.confidence_score > 0.8])
        }
    
    def _generate_followup_plan(
        self,
        patient_context: PatientContext,
        care_options: List[CareOption]
    ) -> Dict[str, Any]:
        """Generate follow-up recommendations based on care options"""
        all_monitoring = []
        for option in care_options:
            all_monitoring.extend(option.monitoring_requirements)
        
        # Remove duplicates while preserving order
        unique_monitoring = list(dict.fromkeys(all_monitoring))
        
        return {
            "immediate_actions": ["Review care options with patient", "Discuss risks and benefits"],
            "monitoring_schedule": unique_monitoring[:5],  # Top 5 monitoring requirements
            "follow_up_timeline": "Follow up within 1-2 weeks or as clinically indicated",
            "red_flag_symptoms": [
                "Worsening of presenting symptoms",
                "New concerning symptoms", 
                "Medication side effects"
            ]
        }


# Utility function for easy integration
def create_clinical_decision_engine() -> ClinicalDecisionEngine:
    """Factory function to create a new clinical decision engine instance"""
    return ClinicalDecisionEngine()


if __name__ == "__main__":
    # Example usage
    engine = ClinicalDecisionEngine()
    
    # Sample patient data
    sample_patient = {
        "patient_id": "DEMO_001",
        "demographics": {"age": 65, "gender": "male"},
        "past_medical_history": ["Hypertension", "Type 2 diabetes", "Hyperlipidemia"],
        "medications": ["Lisinopril 10mg daily", "Metformin 1000mg BID"],
        "vital_signs": {"BP": "150/90", "HR": "78", "RR": "16"},
        "allergies": ["Penicillin"]
    }
    
    # Process clinical query
    result = engine.process_clinical_query(
        patient_data=sample_patient,
        clinical_question="What are the best treatment options for this patient's hypertension?",
        chief_complaint="Elevated blood pressure readings",
        urgency="routine"
    )
    
    print("\n" + "="*50)
    print("CLINICAL DECISION SUPPORT RESULT")
    print("="*50)
    print(json.dumps(result, indent=2, default=str))