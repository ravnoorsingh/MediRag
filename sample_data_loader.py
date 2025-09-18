"""
Sample Data Loader for MediRag System
Implements hybrid caching by pre-embedding medical dictionary and literature data
"""

import json
import os
from typing import List, Dict, Any
from utils import HybridMedicalRAG, get_embedding
import uuid

class SampleDataLoader:
    def __init__(self, sample_data_path: str = "sample_data"):
        self.sample_data_path = sample_data_path
        self.hybrid_rag = None
    
    def initialize_hybrid_rag(self):
        """Initialize the hybrid RAG system"""
        self.hybrid_rag = HybridMedicalRAG()
        return self.hybrid_rag
    
    def load_json_file(self, filepath: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath}: {e}")
            return {}
    
    def embed_medical_dictionary(self):
        """Pre-embed medical dictionary data into the database"""
        print("📚 Loading and embedding medical dictionary data...")
        
        # Load medical terms
        terms_path = os.path.join(self.sample_data_path, "medical_dictionary", "medical_terms.json")
        terms_data = self.load_json_file(terms_path)
        
        if "medical_terms" in terms_data:
            for term in terms_data["medical_terms"]:
                self._embed_medical_term(term)
        
        # Load medications
        meds_path = os.path.join(self.sample_data_path, "medical_dictionary", "medications.json")
        meds_data = self.load_json_file(meds_path)
        
        if "medications" in meds_data:
            for med in meds_data["medications"]:
                self._embed_medication(med)
        
        # Load procedures
        proc_path = os.path.join(self.sample_data_path, "medical_dictionary", "procedures.json")
        proc_data = self.load_json_file(proc_path)
        
        if "procedures" in proc_data:
            for proc in proc_data["procedures"]:
                self._embed_procedure(proc)
        
        print("✅ Medical dictionary data embedded successfully")
    
    def embed_medical_literature(self):
        """Pre-embed medical literature data into the database"""
        print("📖 Loading and embedding medical literature data...")
        
        # Load research papers
        papers_path = os.path.join(self.sample_data_path, "medical_books_papers", "research_papers.json")
        papers_data = self.load_json_file(papers_path)
        
        if "research_papers" in papers_data:
            for paper in papers_data["research_papers"]:
                self._embed_research_paper(paper)
        
        # Load clinical guidelines
        guidelines_path = os.path.join(self.sample_data_path, "medical_books_papers", "clinical_guidelines.json")
        guidelines_data = self.load_json_file(guidelines_path)
        
        if "clinical_guidelines" in guidelines_data:
            for guideline in guidelines_data["clinical_guidelines"]:
                self._embed_clinical_guideline(guideline)
        
        # Load medical textbooks
        textbooks_path = os.path.join(self.sample_data_path, "medical_books_papers", "medical_textbooks.json")
        textbooks_data = self.load_json_file(textbooks_path)
        
        if "medical_textbooks" in textbooks_data:
            for textbook in textbooks_data["medical_textbooks"]:
                self._embed_medical_textbook(textbook)
        
        print("✅ Medical literature data embedded successfully")
    
    def _embed_medical_term(self, term_data: Dict[str, Any]):
        """Embed a single medical term"""
        try:
            # Create text for embedding
            embedding_text = f"{term_data['term']}: {term_data['definition']}. Category: {term_data['category']}. ICD-10: {term_data.get('icd_10', 'N/A')}"
            
            # Add to Neo4j
            node_id = str(uuid.uuid4())
            cypher_query = """
            CREATE (t:MedicalTerm:Dictionary {
                id: $id,
                term: $term,
                definition: $definition,
                category: $category,
                icd_10: $icd_10,
                synonyms: $synonyms,
                related_terms: $related_terms,
                level: 'bottom',
                data_type: 'medical_dictionary'
            })
            RETURN t
            """
            
            self.hybrid_rag.n4j.query(cypher_query, {
                'id': node_id,
                'term': term_data['term'],
                'definition': term_data['definition'],
                'category': term_data['category'],
                'icd_10': term_data.get('icd_10', ''),
                'synonyms': term_data.get('synonyms', []),
                'related_terms': term_data.get('related_terms', [])
            })
            
            # Add to Qdrant with embedding
            embedding = get_embedding(embedding_text)
            self.hybrid_rag.add_to_qdrant(
                embedding=embedding,
                payload={
                    'id': node_id,
                    'type': 'medical_term',
                    'term': term_data['term'],
                    'category': term_data['category'],
                    'text': embedding_text,
                    'level': 'bottom',
                    'data_source': 'medical_dictionary'
                },
                point_id=node_id
            )
            
        except Exception as e:
            print(f"Error embedding medical term {term_data.get('term', 'unknown')}: {e}")
    
    def _embed_medication(self, med_data: Dict[str, Any]):
        """Embed a single medication"""
        try:
            # Create text for embedding
            embedding_text = f"{med_data['name']} ({med_data['generic_name']}): {med_data['drug_class']} used for {med_data['indication']}. Mechanism: {med_data['mechanism']}"
            
            # Add to Neo4j
            node_id = str(uuid.uuid4())
            cypher_query = """
            CREATE (m:Medication:Dictionary {
                id: $id,
                name: $name,
                generic_name: $generic_name,
                brand_names: $brand_names,
                drug_class: $drug_class,
                indication: $indication,
                mechanism: $mechanism,
                level: 'bottom',
                data_type: 'medical_dictionary'
            })
            RETURN m
            """
            
            self.hybrid_rag.n4j.query(cypher_query, {
                'id': node_id,
                'name': med_data['name'],
                'generic_name': med_data['generic_name'],
                'brand_names': med_data.get('brand_names', []),
                'drug_class': med_data['drug_class'],
                'indication': med_data['indication'],
                'mechanism': med_data['mechanism']
            })
            
            # Add to Qdrant with embedding
            embedding = get_embedding(embedding_text)
            self.hybrid_rag.add_to_qdrant(
                embedding=embedding,
                payload={
                    'id': node_id,
                    'type': 'medication',
                    'name': med_data['name'],
                    'drug_class': med_data['drug_class'],
                    'text': embedding_text,
                    'level': 'bottom',
                    'data_source': 'medical_dictionary'
                },
                point_id=node_id
            )
            
        except Exception as e:
            print(f"Error embedding medication {med_data.get('name', 'unknown')}: {e}")
    
    def _embed_procedure(self, proc_data: Dict[str, Any]):
        """Embed a single procedure"""
        try:
            # Create text for embedding
            embedding_text = f"{proc_data['procedure_name']} (CPT: {proc_data['cpt_code']}): {proc_data['description']}. Category: {proc_data['category']}"
            
            # Add to Neo4j
            node_id = str(uuid.uuid4())
            cypher_query = """
            CREATE (p:Procedure:Dictionary {
                id: $id,
                procedure_name: $procedure_name,
                cpt_code: $cpt_code,
                category: $category,
                description: $description,
                indications: $indications,
                level: 'bottom',
                data_type: 'medical_dictionary'
            })
            RETURN p
            """
            
            self.hybrid_rag.n4j.query(cypher_query, {
                'id': node_id,
                'procedure_name': proc_data['procedure_name'],
                'cpt_code': proc_data['cpt_code'],
                'category': proc_data['category'],
                'description': proc_data['description'],
                'indications': proc_data.get('indications', [])
            })
            
            # Add to Qdrant with embedding
            embedding = get_embedding(embedding_text)
            self.hybrid_rag.add_to_qdrant(
                embedding=embedding,
                payload={
                    'id': node_id,
                    'type': 'procedure',
                    'procedure_name': proc_data['procedure_name'],
                    'category': proc_data['category'],
                    'text': embedding_text,
                    'level': 'bottom',
                    'data_source': 'medical_dictionary'
                },
                point_id=node_id
            )
            
        except Exception as e:
            print(f"Error embedding procedure {proc_data.get('procedure_name', 'unknown')}: {e}")
    
    def _embed_research_paper(self, paper_data: Dict[str, Any]):
        """Embed a research paper"""
        try:
            # Create text for embedding
            embedding_text = f"{paper_data['title']} by {', '.join(paper_data['authors'])} ({paper_data['year']}). Abstract: {paper_data['abstract']}"
            
            # Add to Neo4j
            node_id = str(uuid.uuid4())
            cypher_query = """
            CREATE (r:ResearchPaper:Literature {
                id: $id,
                title: $title,
                authors: $authors,
                journal: $journal,
                year: $year,
                doi: $doi,
                abstract: $abstract,
                level: 'middle',
                data_type: 'medical_literature'
            })
            RETURN r
            """
            
            self.hybrid_rag.n4j.query(cypher_query, {
                'id': node_id,
                'title': paper_data['title'],
                'authors': paper_data['authors'],
                'journal': paper_data['journal'],
                'year': paper_data['year'],
                'doi': paper_data.get('doi', ''),
                'abstract': paper_data['abstract']
            })
            
            # Add to Qdrant with embedding
            embedding = get_embedding(embedding_text)
            self.hybrid_rag.add_to_qdrant(
                embedding=embedding,
                payload={
                    'id': node_id,
                    'type': 'research_paper',
                    'title': paper_data['title'],
                    'journal': paper_data['journal'],
                    'year': paper_data['year'],
                    'text': embedding_text,
                    'level': 'middle',
                    'data_source': 'medical_literature'
                },
                point_id=node_id
            )
            
        except Exception as e:
            print(f"Error embedding research paper {paper_data.get('title', 'unknown')}: {e}")
    
    def _embed_clinical_guideline(self, guideline_data: Dict[str, Any]):
        """Embed a clinical guideline"""
        try:
            # Create text for embedding
            recommendations_text = ". ".join([rec['recommendation'] for rec in guideline_data.get('key_recommendations', [])])
            embedding_text = f"{guideline_data['guideline_title']} by {guideline_data['organization']} ({guideline_data['publication_year']}). Key recommendations: {recommendations_text}"
            
            # Add to Neo4j
            node_id = str(uuid.uuid4())
            cypher_query = """
            CREATE (g:ClinicalGuideline:Literature {
                id: $id,
                guideline_title: $guideline_title,
                organization: $organization,
                publication_year: $publication_year,
                scope: $scope,
                level: 'middle',
                data_type: 'medical_literature'
            })
            RETURN g
            """
            
            self.hybrid_rag.n4j.query(cypher_query, {
                'id': node_id,
                'guideline_title': guideline_data['guideline_title'],
                'organization': guideline_data['organization'],
                'publication_year': guideline_data['publication_year'],
                'scope': guideline_data['scope']
            })
            
            # Add to Qdrant with embedding
            embedding = get_embedding(embedding_text)
            self.hybrid_rag.add_to_qdrant(
                embedding=embedding,
                payload={
                    'id': node_id,
                    'type': 'clinical_guideline',
                    'title': guideline_data['guideline_title'],
                    'organization': guideline_data['organization'],
                    'year': guideline_data['publication_year'],
                    'text': embedding_text,
                    'level': 'middle',
                    'data_source': 'medical_literature'
                },
                point_id=node_id
            )
            
        except Exception as e:
            print(f"Error embedding clinical guideline {guideline_data.get('guideline_title', 'unknown')}: {e}")
    
    def _embed_medical_textbook(self, textbook_data: Dict[str, Any]):
        """Embed medical textbook chapters"""
        try:
            for chapter in textbook_data.get('chapters', []):
                # Create text for embedding
                embedding_text = f"{textbook_data['title']} - Chapter {chapter['chapter_number']}: {chapter['chapter_title']}. {chapter['key_content']}"
                
                # Add to Neo4j
                node_id = str(uuid.uuid4())
                cypher_query = """
                CREATE (t:TextbookChapter:Literature {
                    id: $id,
                    textbook_title: $textbook_title,
                    chapter_number: $chapter_number,
                    chapter_title: $chapter_title,
                    key_content: $key_content,
                    authors: $authors,
                    edition: $edition,
                    year: $year,
                    level: 'middle',
                    data_type: 'medical_literature'
                })
                RETURN t
                """
                
                self.hybrid_rag.n4j.query(cypher_query, {
                    'id': node_id,
                    'textbook_title': textbook_data['title'],
                    'chapter_number': chapter['chapter_number'],
                    'chapter_title': chapter['chapter_title'],
                    'key_content': chapter['key_content'],
                    'authors': textbook_data['authors'],
                    'edition': textbook_data['edition'],
                    'year': textbook_data['year']
                })
                
                # Add to Qdrant with embedding
                embedding = get_embedding(embedding_text)
                self.hybrid_rag.add_to_qdrant(
                    embedding=embedding,
                    payload={
                        'id': node_id,
                        'type': 'textbook_chapter',
                        'textbook_title': textbook_data['title'],
                        'chapter_title': chapter['chapter_title'],
                        'text': embedding_text,
                        'level': 'middle',
                        'data_source': 'medical_literature'
                    },
                    point_id=node_id
                )
                
        except Exception as e:
            print(f"Error embedding textbook {textbook_data.get('title', 'unknown')}: {e}")
    
    def load_patient_scenarios(self) -> List[Dict[str, Any]]:
        """Load patient data for runtime injection (not pre-embedded)"""
        print("👥 Loading patient scenarios for runtime injection...")
        
        # Load patient records
        records_path = os.path.join(self.sample_data_path, "patients_data", "patient_records.json")
        records_data = self.load_json_file(records_path)
        
        # Load clinical scenarios
        scenarios_path = os.path.join(self.sample_data_path, "patients_data", "clinical_scenarios.json")
        scenarios_data = self.load_json_file(scenarios_path)
        
        patient_data = []
        
        if "patient_records" in records_data:
            patient_data.extend(records_data["patient_records"])
        
        if "clinical_scenarios" in scenarios_data:
            patient_data.extend(scenarios_data["clinical_scenarios"])
        
        print(f"✅ Loaded {len(patient_data)} patient scenarios for runtime injection")
        return patient_data
    
    def cache_all_sample_data(self):
        """Main method to cache all dictionary and literature data"""
        print("🚀 Starting hybrid caching of sample data...")
        
        if not self.hybrid_rag:
            self.initialize_hybrid_rag()
        
        # Pre-embed dictionary and literature data for fast retrieval
        self.embed_medical_dictionary()
        self.embed_medical_literature()
        
        # Load patient scenarios (not pre-embedded, for runtime injection)
        patient_scenarios = self.load_patient_scenarios()
        
        print("\n📊 Sample Data Caching Complete!")
        print("✅ Medical dictionary and literature data cached in Neo4j + Qdrant")
        print("✅ Patient scenarios loaded for runtime injection")
        print("💡 System ready for efficient RAG inference!")
        
        return {
            'cached_data': 'medical_dictionary + medical_literature',
            'runtime_data': 'patient_scenarios',
            'total_patient_scenarios': len(patient_scenarios)
        }

# Utility function for easy import
def setup_sample_data_cache():
    """Setup sample data caching system"""
    loader = SampleDataLoader()
    return loader.cache_all_sample_data()

if __name__ == "__main__":
    loader = SampleDataLoader()
    result = loader.cache_all_sample_data()
    print(f"\n🎯 Sample Data Setup Result: {result}")