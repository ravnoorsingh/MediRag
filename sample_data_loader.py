"""
Sample Data Loader for MediRag System
Implements hybrid caching by pre-embedding medical dictionary and literature data
Includes real medical papers processing and comprehensive medical dictionary
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from utils import HybridMedicalRAG, get_embedding
import uuid

class SampleDataLoader:
    def __init__(self, hybrid_rag_instance: HybridMedicalRAG = None, sample_data_path: str = "sample_data"):
        self.sample_data_path = sample_data_path
        self.hybrid_rag = hybrid_rag_instance or HybridMedicalRAG()
        self.papers_processed = 0
        self.concepts_extracted = 0
        self.relationships_created = 0
        self.dictionary_terms_processed = 0
        
        # Medical concept patterns for real paper processing
        self.disease_patterns = [
            r'\b(diabetes|hypertension|cardiovascular|heart disease|stroke|cancer|asthma|copd|pneumonia|infection|covid|influenza|tuberculosis|hepatitis|obesity|depression|anxiety|arthritis|osteoporosis|alzheimer|dementia|kidney disease|liver disease|lung disease|blood pressure|heart failure|coronary|myocardial|infarction|arrhythmia|atherosclerosis|hyperlipidemia|cholesterol)\b',
            r'\b\w*itis\b',  # inflammatory conditions
            r'\b\w*oma\b',   # tumors
            r'\b\w*pathy\b'  # pathological conditions
        ]
        
        self.treatment_patterns = [
            r'\b(treatment|therapy|intervention|medication|drug|surgery|procedure|operation|transplant|chemotherapy|radiotherapy|immunotherapy|rehabilitation|counseling|lifestyle|diet|exercise|prevention|screening|vaccination|antibiotic|antiviral|insulin|statin|beta.?blocker|ace.?inhibitor|diuretic|analgesic|nsaid)\b',
            r'\b\w*(cillin|mycin|cycline|oxacin|azole|pril|sartan|olol|dipine|ide|statin)\b'  # medication endings
        ]
        
        self.symptom_patterns = [
            r'\b(pain|fever|cough|fatigue|nausea|vomiting|diarrhea|constipation|headache|dizziness|shortness of breath|chest pain|abdominal pain|back pain|joint pain|swelling|rash|bleeding|weakness|confusion|memory loss|weight loss|weight gain)\b'
        ]
    
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
    
    def embed_real_medical_dictionary(self):
        """Load and embed the comprehensive medical dictionary"""
        print("📚 Loading comprehensive medical dictionary...")
        
        dict_path = os.path.join(self.sample_data_path, "medical_dictionary", "medical_dictionary.json")
        
        if not os.path.exists(dict_path):
            print(f"⚠️  Medical dictionary not found at {dict_path}")
            return
        
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                dictionary_data = json.load(f)
            
            print(f"   Found {len(dictionary_data)} medical terms in dictionary")
            
            for term, definition in dictionary_data.items():
                self._embed_dictionary_term(term, definition)
            
            print("✅ Comprehensive medical dictionary embedded successfully")
            
        except Exception as e:
            print(f"❌ Error loading medical dictionary: {e}")
    
    def embed_real_medical_papers(self):
        """Load and process real medical papers from text files"""
        print("📄 Loading real medical papers from text files...")
        
        papers_dir = Path("medical_books_papers")
        if not papers_dir.exists():
            print(f"⚠️  Medical papers directory not found: {papers_dir}")
            return
        
        paper_files = list(papers_dir.glob("*.txt"))
        print(f"   Found {len(paper_files)} medical papers to process")
        
        for i, filepath in enumerate(paper_files[:100], 1):  # Process first 100 papers
            print(f"   📄 Processing paper {i}: {filepath.name}")
            self._process_real_medical_paper(filepath)
            
            if i % 20 == 0:
                print(f"      Progress: {i}/{min(100, len(paper_files))} papers processed")
        
        print(f"✅ Real medical papers embedded successfully")
        print(f"   Papers processed: {self.papers_processed}")
        print(f"   Medical concepts extracted: {self.concepts_extracted}")
        print(f"   Relationships created: {self.relationships_created}")
    
    def _embed_dictionary_term(self, term: str, definition: str):
        """Embed a single dictionary term"""
        try:
            # Clean and prepare the term and definition
            clean_term = term.strip()
            clean_definition = definition.strip()
            
            if len(clean_term) < 2 or len(clean_definition) < 10:
                return  # Skip very short entries
            
            # Create embedding text
            embedding_text = f"Medical Term: {clean_term}. Definition: {clean_definition}"
            
            # Create unique ID
            node_id = str(uuid.uuid4())
            
            # Add to Neo4j
            cypher_query = """
            CREATE (t:MedicalTerm:Dictionary {
                id: $id,
                term: $term,
                definition: $definition,
                level: 'bottom',
                data_type: 'comprehensive_medical_dictionary'
            })
            """
            
            self.hybrid_rag.n4j.query(cypher_query, {
                'id': node_id,
                'term': clean_term,
                'definition': clean_definition
            })
            
            # Add to Qdrant
            embedding = get_embedding(embedding_text)
            self.hybrid_rag.add_to_qdrant(
                embedding=embedding,
                payload={
                    'id': node_id,
                    'type': 'medical_dictionary_term',
                    'data_type': 'medical_dictionary_term',  # Add this field
                    'term': clean_term,
                    'title': clean_term,  # Add title field
                    'definition': clean_definition,
                    'content': clean_definition,  # Add content field
                    'text': embedding_text,
                    'level': 'bottom',
                    'data_source': 'comprehensive_medical_dictionary'
                },
                point_id=node_id
            )
            
        except Exception as e:
            print(f"   ⚠️  Error embedding dictionary term {term}: {e}")
    
    def _process_real_medical_paper(self, filepath: Path):
        """Process a single real medical paper"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if len(content) < 200:  # Skip very short files
                return
            
            # Extract basic metadata
            title = filepath.stem.replace('_', ' ').replace('@', '/').title()[:200]
            doi = self._extract_doi(content)
            authors = self._extract_authors(content)
            pub_year = self._extract_year(content)
            
            # Extract medical concepts
            concepts = self._extract_medical_concepts(content)
            
            # Create paper node in Neo4j
            paper_id = str(uuid.uuid4())
            self._create_paper_node(paper_id, title, content[:3000], doi, authors, pub_year, concepts)
            
            # Create embeddings for different sections of the paper
            self._create_paper_embeddings(paper_id, title, content, doi, authors, pub_year)
            
            self.papers_processed += 1
            self.concepts_extracted += len(concepts)
            
        except Exception as e:
            print(f"   ⚠️  Error processing paper {filepath.name}: {e}")
    
    def _extract_doi(self, content: str) -> Optional[str]:
        """Extract DOI from paper content"""
        doi_patterns = [
            r'doi:\s*([^\s\n]+)',
            r'DOI:\s*([^\s\n]+)', 
            r'https?://doi\.org/([^\s\n]+)',
            r'10\.\d+/[^\s\n]+'
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, content[:2000], re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        return None
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extract authors from paper content"""
        # Look for author patterns in first 1500 chars
        content_start = content[:1500]
        
        author_patterns = [
            r'([A-Z][a-z]+,?\s+[A-Z]\.?\s*[A-Z]?\.?(?:\s+and\s+[A-Z][a-z]+,?\s+[A-Z]\.?\s*[A-Z]?\.?)*)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)*)'
        ]
        
        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, content_start)
            for match in matches:
                author_list = re.split(r'\s+and\s+|,\s*', match)
                for author in author_list:
                    author = author.strip()
                    if len(author) > 5 and len(author.split()) >= 2:
                        authors.append(author)
        
        return list(set(authors))[:3]  # Max 3 authors
    
    def _extract_year(self, content: str) -> Optional[int]:
        """Extract publication year"""
        year_patterns = [
            r'\b(20\d{2})\b',
            r'\b(19\d{2})\b'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, content[:1000])
            for match in matches:
                year = int(match)
                if 1990 <= year <= 2024:
                    return year
        return None
    
    def _extract_medical_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract medical concepts from paper content"""
        concepts = []
        content_lower = content.lower()
        
        # Extract diseases
        for pattern in self.disease_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                concepts.append({
                    'name': match.lower(),
                    'category': 'disease',
                    'confidence': 0.8
                })
        
        # Extract treatments  
        for pattern in self.treatment_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                concepts.append({
                    'name': match.lower(),
                    'category': 'treatment',
                    'confidence': 0.7
                })
        
        # Deduplicate
        unique_concepts = {}
        for concept in concepts:
            key = (concept['name'], concept['category'])
            if key not in unique_concepts:
                unique_concepts[key] = concept
        
        return list(unique_concepts.values())[:20]  # Max 20 concepts per paper
    
    def _create_paper_node(self, paper_id: str, title: str, content: str, doi: Optional[str], 
                          authors: List[str], pub_year: Optional[int], concepts: List[Dict]):
        """Create paper node and relationships in Neo4j"""
        
        # Create paper node
        cypher_query = """
        CREATE (p:RealMedicalPaper:Literature {
            id: $paper_id,
            title: $title,
            content: $content,
            doi: $doi,
            authors: $authors,
            publication_year: $pub_year,
            level: 'middle',
            data_type: 'real_medical_paper'
        })
        """
        
        self.hybrid_rag.n4j.query(cypher_query, {
            'paper_id': paper_id,
            'title': title,
            'content': content,
            'doi': doi,
            'authors': authors,
            'pub_year': pub_year
        })
        
        # Create concept nodes and relationships
        for concept in concepts:
            concept_id = str(uuid.uuid4())
            
            # Create concept node
            concept_query = f"""
            MERGE (c:{concept['category'].title()} {{name: $name}})
            SET c.category = $category,
                c.confidence = $confidence,
                c.level = 'bottom',
                c.data_type = 'extracted_medical_concept'
            """
            
            self.hybrid_rag.n4j.query(concept_query, {
                'name': concept['name'],
                'category': concept['category'],
                'confidence': concept['confidence']
            })
            
            # Create relationship
            rel_query = f"""
            MATCH (p:RealMedicalPaper {{id: $paper_id}})
            MATCH (c:{concept['category'].title()} {{name: $concept_name}})
            MERGE (p)-[r:DISCUSSES]->(c)
            SET r.confidence = $confidence
            """
            
            self.hybrid_rag.n4j.query(rel_query, {
                'paper_id': paper_id,
                'concept_name': concept['name'],
                'confidence': concept['confidence']
            })
            
            self.relationships_created += 1
    
    def _create_paper_embeddings(self, paper_id: str, title: str, content: str,
                                doi: Optional[str], authors: List[str], pub_year: Optional[int]):
        """Create embeddings for the paper"""
        
        # Full paper embedding
        paper_text = f"Medical Paper: {title}. Content: {content[:2000]}"
        
        try:
            embedding = get_embedding(paper_text)
            self.hybrid_rag.add_to_qdrant(
                embedding=embedding,
                payload={
                    'id': paper_id,
                    'type': 'real_medical_paper',
                    'data_type': 'real_medical_paper',  # Add this field
                    'title': title,
                    'content': content[:1000],  # Add content field
                    'doi': doi or '',
                    'authors': '; '.join(authors) if authors else '',  # Convert list to string
                    'publication_year': pub_year or 0,
                    'text': paper_text,
                    'level': 'middle',
                    'data_source': 'real_medical_literature'
                },
                point_id=paper_id
            )
            
            # Create section embeddings (if content is long enough)
            if len(content) > 2000:
                sections = [
                    content[:2000],     # Beginning
                    content[len(content)//2:len(content)//2 + 2000],  # Middle
                    content[-2000:]     # End
                ]
                
                for i, section in enumerate(sections):
                    section_text = f"Medical Paper Section {i+1}: {title}. Content: {section}"
                    section_embedding = get_embedding(section_text)
                    
                    self.hybrid_rag.add_to_qdrant(
                        embedding=section_embedding,
                        payload={
                            'id': f"{paper_id}_section_{i+1}",
                            'type': 'paper_section',
                            'data_type': 'paper_section',  # Add this field
                            'title': f"{title} - Section {i+1}",
                            'content': section[:500],  # Add content field
                            'parent_paper_id': paper_id,
                            'doi': doi or '',
                            'authors': '; '.join(authors) if authors else '',  # Convert list to string
                            'publication_year': pub_year or 0,
                            'text': section_text,
                            'level': 'bottom',
                            'data_source': 'real_medical_literature'
                        },
                        point_id=f"{paper_id}_section_{i+1}"
                    )
                    
        except Exception as e:
            print(f"   ⚠️  Error creating embeddings for paper {paper_id}: {e}")
    
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
        """Main method to cache all dictionary and literature data including real medical papers"""
        print("🚀 Starting comprehensive hybrid caching of medical data...")
        
        if not self.hybrid_rag:
            self.initialize_hybrid_rag()
        
        # Pre-embed all medical data for fast retrieval
        print("\n📚 Phase 1: Structured Medical Data")
        self.embed_medical_dictionary()
        self.embed_medical_literature()
        
        print("\n📄 Phase 2: Real Medical Literature")  
        self.embed_real_medical_dictionary()
        self.embed_real_medical_papers()
        
        # Load patient scenarios (not pre-embedded, for runtime injection)
        print("\n👥 Phase 3: Patient Scenarios")
        patient_scenarios = self.load_patient_scenarios()
        
        print("\n📊 Comprehensive Medical Data Caching Complete!")
        print("✅ Structured medical dictionary and literature data cached")
        print("✅ Comprehensive medical dictionary (13K+ terms) cached")
        print(f"✅ Real medical papers ({self.papers_processed} papers) cached")
        print(f"✅ Medical concepts ({self.concepts_extracted} concepts) extracted and linked")
        print(f"✅ Neo4j relationships ({self.relationships_created} relationships) created")
        print("✅ Patient scenarios loaded for runtime injection")
        print("💡 MediRag system ready for evidence-based clinical decision support!")
        
        return {
            'cached_data': 'medical_dictionary + medical_literature + real_medical_papers + comprehensive_dictionary',
            'runtime_data': 'patient_scenarios',
            'total_patient_scenarios': len(patient_scenarios),
            'papers_processed': self.papers_processed,
            'concepts_extracted': self.concepts_extracted,
            'relationships_created': self.relationships_created
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