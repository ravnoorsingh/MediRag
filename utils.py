import requests
import json
import os
from neo4j import GraphDatabase
import numpy as np
from camel.storages import Neo4jGraph
from camel.memories.blocks.vectordb_block import VectorDBBlock
from camel.storages.vectordb_storages import QdrantStorage
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys_prompt_one = """
Please answer the question using insights supported by provided graph-based data relevant to medical information.
"""

sys_prompt_two = """
Modify the response to the question using the provided references. Include precise citations relevant to your answer. You may use multiple citations simultaneously, denoting each with the reference index number. For example, cite the first and third documents as [1][3]. If the references do not pertain to the response, simply provide a concise answer to the original question.
"""

# API configuration
ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "mediragpassword123")
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
qdrant_collection = os.getenv("QDRANT_COLLECTION_NAME", "medirag_vectors")

def get_embedding(text, model=None):
    """Get embeddings using Ollama"""
    if model is None:
        model = ollama_embedding_model
    
    try:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result['embedding']
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding from Ollama: {e}")
        # Fallback to random embedding for testing
        return np.random.rand(384).tolist()  # nomic-embed-text dimension
    except Exception as e:
        print(f"Unexpected error in get_embedding: {e}")
        return np.random.rand(384).tolist()
    
    return result['embedding']

def get_qdrant_client():
    """Initialize Qdrant client with proper configuration including API key"""
    if qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        return QdrantClient(url=qdrant_url)

def setup_qdrant_collection(client=None):
    """Setup Qdrant collection for medical embeddings with proper dimension validation"""
    if client is None:
        client = get_qdrant_client()
    
    # First, determine the actual embedding dimensions from Ollama
    try:
        test_embedding = get_embedding("test")
        actual_dim = len(test_embedding)
        print(f"🔍 Detected embedding dimensions: {actual_dim}")
    except Exception:
        actual_dim = 768  # Default fallback
        print(f"⚠️ Could not detect dimensions, using default: {actual_dim}")
    
    # Create collection if it doesn't exist, or recreate if dimensions are wrong
    try:
        collection_info = client.get_collection(qdrant_collection)
        current_size = collection_info.config.params.vectors.size
        print(f"✓ Collection '{qdrant_collection}' exists with {current_size} dimensions")
        
        # Validate vector size matches actual embedding dimensions
        if current_size != actual_dim:
            print(f"⚠️ Dimension mismatch! Expected {actual_dim}, got {current_size}. Recreating...")
            client.delete_collection(qdrant_collection)
            client.create_collection(
                collection_name=qdrant_collection,
                vectors_config=VectorParams(size=actual_dim, distance=Distance.COSINE)
            )
            print(f"✓ Recreated collection '{qdrant_collection}' with correct {actual_dim} dimensions")
    except Exception:
        client.create_collection(
            collection_name=qdrant_collection,
            vectors_config=VectorParams(size=actual_dim, distance=Distance.COSINE)
        )
        print(f"✓ Created collection '{qdrant_collection}' with {actual_dim} dimensions")
    
    return client

def fetch_texts(n4j):
    # Fetch the text for each node
    query = "MATCH (n) RETURN n.id AS id"
    return n4j.query(query)

def add_embeddings(n4j, node_id, embedding):
    # Upload embeddings to Neo4j
    query = "MATCH (n) WHERE n.id = $node_id SET n.embedding = $embedding"
    n4j.query(query, params = {"node_id":node_id, "embedding":embedding})

def add_nodes_emb(n4j):
    nodes = fetch_texts(n4j)

    for node in nodes:
        # Calculate embedding for each node's text
        if node['id']:  # Ensure there is text to process
            embedding = get_embedding(node['id'])
            # Store embedding back in the node
            add_embeddings(n4j, node['id'], embedding)

def add_ge_emb(graph_element):
    for node in graph_element.nodes:
        emb = get_embedding(node.id)
        node.properties['embedding'] = emb
    return graph_element

def add_gid(graph_element, gid):
    for node in graph_element.nodes:
        node.properties['gid'] = gid
    for rel in graph_element.relationships:
        rel.properties['gid'] = gid
    return graph_element

def call_llm(sys, user):
    """Call Ollama LLM for text generation"""
    try:
        prompt = f"System: {sys}\n\nUser: {user}"
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "max_tokens": 500
                }
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result['response']
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama LLM: {e}")
        return f"Error: Unable to generate response. Please ensure Ollama is running with {ollama_model} model."
    except Exception as e:
        print(f"Unexpected error in call_llm: {e}")
        return f"Error: {str(e)}"

def find_index_of_largest(nums):
    # Sorting the list while keeping track of the original indexes
    sorted_with_index = sorted((num, index) for index, num in enumerate(nums))
    
    # Extracting the original index of the largest element
    largest_original_index = sorted_with_index[-1][1]
    
    return largest_original_index

def get_response(n4j, gid, query):
    selfcont = ret_context(n4j, gid)
    linkcont = link_context(n4j, gid)
    user_one = "the question is: " + query + "the provided information is:" +  "".join(selfcont)
    res = call_llm(sys_prompt_one,user_one)
    user_two = "the question is: " + query + "the last response of it is:" +  res + "the references are: " +  "".join(linkcont)
    res = call_llm(sys_prompt_two,user_two)
    return res

def link_context(n4j, gid):
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'REFERENCES' relationship
        MATCH (n)-[r:REFERENCE]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'REFERENCE' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS ReferenceType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def ret_context(n4j, gid):
    cont = []
    ret_query = """
    // Match all nodes with a specific gid but not of type "Summary" and collect them
    MATCH (n)
    WHERE n.gid = $gid AND NOT n:Summary
    WITH collect(n) AS nodes

    // Unwind the nodes to a pairs and match relationships between them
    UNWIND nodes AS n
    UNWIND nodes AS m
    MATCH (n)-[r]-(m)
    WHERE n.gid = m.gid AND id(n) < id(m) AND NOT n:Summary AND NOT m:Summary // Ensure each pair is processed once and exclude "Summary" nodes in relationships
    WITH n, m, TYPE(r) AS relType

    // Return node IDs and relationship types in structured format
    RETURN n.id AS NodeId1, relType, m.id AS NodeId2
    """
    res = n4j.query(ret_query, {'gid': gid})
    for r in res:
        cont.append(r['NodeId1'] + r['relType'] + r['NodeId2'])
    return cont

def merge_similar_nodes(n4j, gid):
    # Define your merge query here. Adjust labels and properties according to your graph schema
    if gid:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n.gid = m.gid AND n.gid = $gid AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        result = n4j.query(merge_query, {'gid': gid})
    else:
        merge_query = """
            // Define a threshold for cosine similarity
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        result = n4j.query(merge_query)
    return result

def ref_link(n4j, gid1, gid2):
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.6 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:REFERENCE]->(n)

        // Return results
        RETURN n, m
"""
    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result


def str_uuid():
    # Generate a random UUID
    generated_uuid = uuid.uuid4()

    # Convert UUID to a string
    return str(generated_uuid)


def setup_hierarchical_structure():
    """
    Set up 3-level hierarchical structure using HybridMedicalRAG:
    - Bottom Level: Medical Dictionary (UMLS concepts)
    - Middle Level: Medical Books and Papers 
    - Top Level: Patient Records
    """
    print("🚀 Setting up hierarchical medical database structure...")
    
    # Initialize hybrid RAG system
    hybrid_rag = HybridMedicalRAG()
    
    # Create constraints and indexes for better performance
    constraints_queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MedicalConcept) REQUIRE n.cui IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MedicalBook) REQUIRE n.id IS UNIQUE", 
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MedicalPaper) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:PatientRecord) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Guideline) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Summary) REQUIRE (n.gid, n.level) IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (n:MedicalConcept) ON n.semantic_type"
        # Note: General indexes on gid and level removed due to Neo4j syntax changes
    ]
    
    for query in constraints_queries:
        try:
            hybrid_rag.n4j.query(query)
        except Exception as e:
            print(f"Warning: Could not create constraint/index: {e}")
    
    print("✓ Created database constraints and indexes")
    
    # Populate with comprehensive sample data
    from sample_data_loader import SampleDataLoader
    data_loader = SampleDataLoader()
    data_loader.hybrid_rag = hybrid_rag  # Use the existing hybrid_rag instance
    
    print("� Loading medical dictionary data...")
    data_loader.embed_medical_dictionary()
    
    print("📚 Loading medical literature data...")
    data_loader.embed_medical_literature()
    
    print("🎉 Hierarchical structure setup completed!")
    return hybrid_rag

def setup_hierarchical_structure_legacy(n4j):
    """
    Set up 3-level hierarchical structure in Neo4j:
    - Bottom Level: Medical Dictionary (UMLS concepts)
    - Middle Level: Medical Books and Papers 
    - Top Level: Patient Records
    """
    # Create constraints and indexes for better performance
    constraints_queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MedicalConcept) REQUIRE n.cui IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MedicalBook) REQUIRE n.id IS UNIQUE", 
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MedicalPaper) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:PatientRecord) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Summary) REQUIRE (n.gid, n.level) IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (n:MedicalConcept) ON n.semantic_type",
        "CREATE INDEX IF NOT EXISTS FOR (n) ON n.gid",
        "CREATE INDEX IF NOT EXISTS FOR (n) ON n.level"
    ]
    
    for query in constraints_queries:
        try:
            n4j.query(query)
        except Exception as e:
            print(f"Constraint/Index creation note: {e}")
    
    print("✅ Hierarchical structure constraints and indexes created")


def create_medical_concept(n4j, cui, preferred_name, semantic_type, definition="", level="BOTTOM"):
    """Create a medical concept node (Bottom level - Medical Dictionary)"""
    query = """
        MERGE (mc:MedicalConcept {cui: $cui})
        SET mc.preferred_name = $preferred_name,
            mc.semantic_type = $semantic_type,
            mc.definition = $definition,
            mc.level = $level,
            mc.created_at = datetime()
        RETURN mc
    """
    return n4j.query(query, {
        'cui': cui,
        'preferred_name': preferred_name, 
        'semantic_type': semantic_type,
        'definition': definition,
        'level': level
    })


def create_medical_literature(n4j, doc_id, title, doc_type, content, level="MIDDLE"):
    """Create medical literature node (Middle level - Books/Papers)"""
    label = "MedicalBook" if doc_type == "book" else "MedicalPaper"
    query = f"""
        MERGE (ml:{label} {{id: $doc_id}})
        SET ml.title = $title,
            ml.content = $content,
            ml.doc_type = $doc_type,
            ml.level = $level,
            ml.created_at = datetime()
        RETURN ml
    """
    return n4j.query(query, {
        'doc_id': doc_id,
        'title': title,
        'content': content, 
        'doc_type': doc_type,
        'level': level
    })


def create_patient_record(n4j, record_id, patient_id, content, record_type="clinical_note", level="TOP"):
    """Create patient record node (Top level - Patient Data)"""
    query = """
        MERGE (pr:PatientRecord {id: $record_id})
        SET pr.patient_id = $patient_id,
            pr.content = $content,
            pr.record_type = $record_type,
            pr.level = $level,
            pr.created_at = datetime()
        RETURN pr
    """
    return n4j.query(query, {
        'record_id': record_id,
        'patient_id': patient_id,
        'content': content,
        'record_type': record_type,
        'level': level
    })


def create_hierarchical_relationship(n4j, from_id, to_id, relationship_type, from_level, to_level, similarity_score=None):
    """Create relationships between different hierarchy levels"""
    # Determine node labels based on levels
    from_label = get_node_label_by_level(from_level)
    to_label = get_node_label_by_level(to_level)
    
    query = f"""
        MATCH (from:{from_label}), (to:{to_label})
        WHERE from.id = $from_id OR from.cui = $from_id
        AND (to.id = $to_id OR to.cui = $to_id)
        MERGE (from)-[r:{relationship_type}]->(to)
        SET r.created_at = datetime()
        """ + (", r.similarity_score = $similarity_score" if similarity_score else "") + """
        RETURN from, r, to
    """
    
    params = {'from_id': from_id, 'to_id': to_id}
    if similarity_score:
        params['similarity_score'] = similarity_score
        
    return n4j.query(query, params)


def get_node_label_by_level(level):
    """Get appropriate node label based on hierarchy level"""
    level_map = {
        "BOTTOM": "MedicalConcept",
        "MIDDLE": "MedicalBook", # or MedicalPaper 
        "TOP": "PatientRecord"
    }
    return level_map.get(level, "Entity")


class HybridMedicalRAG:
    """
    Hybrid system combining Neo4j graph relationships with Qdrant vector storage
    for semantic embeddings across the 3-level hierarchy
    """
    
    def __init__(self, vector_dim=None):  # Auto-detect dimensions
        # Auto-detect embedding dimensions
        if vector_dim is None:
            try:
                test_embedding = get_embedding("test")
                vector_dim = len(test_embedding)
                print(f"🔍 Auto-detected embedding dimensions: {vector_dim}")
            except Exception:
                vector_dim = 768  # Default fallback
                print(f"⚠️ Could not detect dimensions, using default: {vector_dim}")
        
        # Initialize Neo4j connection
        try:
            self.n4j = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_username, 
                password=neo4j_password
            )
            print("✓ Connected to Neo4j")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            raise
            
        # Initialize Qdrant client and collection
        try:
            self.qdrant_client = get_qdrant_client()
            setup_qdrant_collection(self.qdrant_client)
            self.vector_dim = vector_dim
            self.collection_name = qdrant_collection
            print("✓ Connected to Qdrant")
        except Exception as e:
            print(f"✗ Failed to connect to Qdrant: {e}")
            raise
        
        # Initialize vector storage using direct Qdrant with proper error handling
        try:
            self.vector_storage = None  # Using direct Qdrant instead of CAMEL
            self.vectordb_block = None
            print("✓ Initialized direct Qdrant storage with Ollama embeddings")
        except Exception as e:
            print(f"✗ Failed to initialize vector storage: {e}")
            # Fallback to direct Qdrant usage
            self.vector_storage = None
            self.vectordb_block = None
        
    def add_node_with_embedding(self, node_id, content, level, node_type, metadata=None):
        """Add a node to both Neo4j and Qdrant with embeddings"""
        # Get embedding for the content
        embedding = get_embedding(content)
        
        # Add to Neo4j with embedding property
        if level == "BOTTOM":
            # Medical concept
            query = """
                MERGE (n:MedicalConcept {cui: $node_id})
                SET n.embedding = $embedding,
                    n.level = $level,
                    n.content = $content
            """
        elif level == "MIDDLE":
            # Medical literature
            label = "MedicalBook" if node_type == "book" else "MedicalPaper"
            query = f"""
                MERGE (n:{label} {{id: $node_id}})
                SET n.embedding = $embedding,
                    n.level = $level,
                    n.content = $content
            """
        else:  # TOP
            # Patient record
            query = """
                MERGE (n:PatientRecord {id: $node_id})
                SET n.embedding = $embedding,
                    n.level = $level,
                    n.content = $content
            """
            
        if metadata:
            for key, value in metadata.items():
                query += f", n.{key} = ${key}"
        
        query += " RETURN n"
        
        params = {
            'node_id': node_id,
            'embedding': embedding,
            'level': level,
            'content': content
        }
        if metadata:
            params.update(metadata)
            
        self.n4j.query(query, params)
        
        # Add to Qdrant for semantic search using direct client
        try:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "node_id": node_id,
                    "level": level,
                    "node_type": node_type,
                    "content": content[:1000]  # Limit content size for payload
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=qdrant_collection,
                points=[point]
            )
            print(f"✓ Added {node_id} to Qdrant")
        except Exception as e:
            print(f"✗ Failed to add {node_id} to Qdrant: {e}")
        
    def semantic_search_across_levels(self, query_text, level_filter=None, top_k=15):
        """
        Perform enhanced semantic search across all levels
        Returns both graph relationships and vector similarity results
        """
        print(f"🔍 Performing semantic search for: {query_text[:100]}...")
        
        # Debug: Check what's in the database (only on first search to avoid spam)
        if not hasattr(self, '_debug_done'):
            self.debug_database_contents()
            self._debug_done = True
        
        # Vector-based semantic search using direct Qdrant
        vector_results = []
        try:
            # Use only the clinical question part for embedding if possible
            if '\nClinical question:' in query_text:
                query_for_embedding = query_text.split('\nClinical question:')[1].strip()
            else:
                query_for_embedding = query_text.strip()
            query_embedding = get_embedding(query_for_embedding)

            # Build filter for level if specified
            search_filter = None
            if level_filter:
                if isinstance(level_filter, str):
                    level_filter = [level_filter]
                search_filter = {
                    "should": [
                        {
                            "key": "level",
                            "match": {"value": level}
                        } for level in level_filter
                    ]
                }

            # Raise score threshold for more relevant results
            vector_results = self.qdrant_client.search(
                collection_name=qdrant_collection,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k * 2,  # Get more results for better selection
                with_payload=True,
                with_vectors=False,
                score_threshold=0.55  # Higher threshold for more relevant results
            )
            print(f"✓ Found {len(vector_results)} vector matches")
            for i, result in enumerate(vector_results, 1):
                title = result.payload.get('title', 'No title') if hasattr(result, 'payload') else 'No title'
                score = getattr(result, 'score', None)
                print(f"   {i}. Score: {score:.3f} | Title: {title[:60]}")
        except Exception as e:
            print(f"✗ Vector search failed: {e}")
            vector_results = []
        
        # Enhanced graph-based search using multiple strategies
        graph_results = []
        try:
            # Extract key search terms
            search_terms = [term.strip() for term in query_text.lower().split() if len(term.strip()) > 2][:8]
            
            # Strategy 1: Direct content search
            content_search = """
                MATCH (n)
                WHERE n.content IS NOT NULL 
                AND (
                    any(term in $search_terms WHERE toLower(n.content) CONTAINS term)
                    OR toLower(n.content) CONTAINS toLower($query_text)
                )
                RETURN n, 0.9 as similarity, labels(n) as node_labels, n.data_type as data_type
                LIMIT $limit
            """
            
            # Strategy 2: Medical dictionary search
            dict_search = """
                MATCH (n)
                WHERE (n.definition IS NOT NULL AND (
                    any(term in $search_terms WHERE toLower(n.definition) CONTAINS term)
                    OR toLower(n.definition) CONTAINS toLower($query_text)
                ))
                OR (n.term IS NOT NULL AND (
                    any(term in $search_terms WHERE toLower(n.term) CONTAINS term)
                    OR toLower(n.term) CONTAINS toLower($query_text)
                ))
                RETURN n, 0.8 as similarity, labels(n) as node_labels, n.data_type as data_type
                LIMIT $limit
            """
            
            # Strategy 3: Title/name search
            title_search = """
                MATCH (n)
                WHERE (n.title IS NOT NULL AND (
                    any(term in $search_terms WHERE toLower(n.title) CONTAINS term)
                    OR toLower(n.title) CONTAINS toLower($query_text)
                ))
                OR (n.name IS NOT NULL AND (
                    any(term in $search_terms WHERE toLower(n.name) CONTAINS term)
                    OR toLower(n.name) CONTAINS toLower($query_text)
                ))
                RETURN n, 0.7 as similarity, labels(n) as node_labels, n.data_type as data_type
                LIMIT $limit
            """
            
            # Execute all search strategies
            search_params = {
                'query_text': query_text,
                'search_terms': search_terms,
                'limit': top_k
            }
            
            all_results = []
            for search_query in [content_search, dict_search, title_search]:
                try:
                    results = self.n4j.query(search_query, search_params)
                    all_results.extend(results)
                except Exception as e:
                    print(f"   ⚠️ Search strategy failed: {e}")
            
            # Remove duplicates based on node ID
            seen_ids = set()
            unique_results = []
            for result in all_results:
                node = result.get('n', {})
                node_id = node.get('id', str(node))
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    unique_results.append(result)
            
            # Sort by similarity and take top results
            unique_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            graph_results = unique_results[:top_k]
            
            print(f"✓ Found {len(graph_results)} graph matches with relationships")
            
            # If still no results, try basic fallback
            if not graph_results:
                print("   🔄 Trying basic fallback search...")
                fallback_query = """
                    MATCH (n)
                    WHERE n.content IS NOT NULL OR n.definition IS NOT NULL OR n.title IS NOT NULL
                    RETURN n, 0.5 as similarity, labels(n) as node_labels, n.data_type as data_type
                    LIMIT $limit
                """
                graph_results = self.n4j.query(fallback_query, {'limit': top_k})
                print(f"   ✓ Fallback found {len(graph_results)} nodes")
                
        except Exception as e:
            print(f"⚠️ Graph search failed completely: {e}")
            graph_results = []
        
        return {
            'vector_results': vector_results,
            'graph_results': graph_results
        }
        
    def find_cross_level_connections(self, node_id, max_depth=2):
        """Find connections across different hierarchy levels"""
        query = """
            MATCH (start)
            WHERE start.id = $node_id OR start.cui = $node_id
            MATCH path = (start)-[*1..$max_depth]-(connected)
            WHERE start.level <> connected.level
            RETURN path, 
                   start.level as start_level,
                   connected.level as connected_level,
                   labels(connected) as connected_labels,
                   connected.content as connected_content
            LIMIT 20
        """
        
        return self.n4j.query(query, {'node_id': node_id, 'max_depth': max_depth})
        
    def get_level_statistics(self):
        """Get statistics about nodes at each level"""
        query = """
            MATCH (n)
            WHERE n.level IS NOT NULL
            RETURN n.level as level, 
                   labels(n) as node_types,
                   count(*) as count
            ORDER BY n.level
        """
        
        return self.n4j.query(query)
    
    def debug_database_contents(self):
        """Enhanced debug method to check database contents"""
        try:
            print("\n🔍 DATABASE CONTENTS DEBUG:")
            print("=" * 60)
            
            # Check Neo4j content with better queries
            print("\n📊 NEO4J DATABASE ANALYSIS:")
            
            # Count all nodes
            count_query = "MATCH (n) RETURN count(n) as total_nodes"
            total_result = self.n4j.query(count_query)
            total_nodes = total_result[0]['total_nodes'] if total_result else 0
            print(f"   Total nodes: {total_nodes}")
            
            # Count by labels
            label_query = """
                MATCH (n)
                WITH labels(n) as node_labels
                UNWIND node_labels as label
                RETURN label, count(*) as count
                ORDER BY count DESC
            """
            label_results = self.n4j.query(label_query)
            print("   Nodes by label:")
            for result in label_results[:10]:  # Top 10 labels
                print(f"     - {result['label']}: {result['count']}")
            
            # Count by levels/data_type
            level_query = """
                MATCH (n)
                WHERE n.level IS NOT NULL OR n.data_type IS NOT NULL
                RETURN coalesce(n.level, n.data_type, 'unknown') as category, count(*) as count
                ORDER BY count DESC
            """
            level_results = self.n4j.query(level_query)
            print("   Nodes by level/data_type:")
            for result in level_results[:10]:
                print(f"     - {result['category']}: {result['count']}")
            
            # Enhanced content analysis for different data types
            print("\n   Content type analysis:")
            content_types = [
                ("Medical Dictionary Terms", "n.definition IS NOT NULL AND n.term IS NOT NULL"),
                ("Real Medical Papers", "n.content IS NOT NULL AND n.data_type = 'real_medical_paper'"),
                ("Research Papers (Legacy)", "n.content IS NOT NULL AND n.data_type = 'research_paper'"),
                ("Medical Concepts", "n.concept IS NOT NULL"),
                ("Patients", "n.name IS NOT NULL AND n.data_type = 'patient'"),
                ("Any Content", "n.content IS NOT NULL OR n.definition IS NOT NULL OR n.title IS NOT NULL")
            ]
            
            for category, condition in content_types:
                try:
                    query = f"MATCH (n) WHERE {condition} RETURN count(n) as count"
                    result = self.n4j.query(query)
                    count = result[0]['count'] if result else 0
                    print(f"     - {category}: {count}")
                except Exception as e:
                    print(f"     - {category}: Error ({e})")
            
            # Sample actual content with better categorization
            sample_query = """
                MATCH (n)
                WHERE n.content IS NOT NULL OR n.title IS NOT NULL OR n.definition IS NOT NULL OR n.term IS NOT NULL
                RETURN 
                    coalesce(n.title, n.term, n.name, 'Untitled') as title,
                    coalesce(n.content, n.definition, 'No content') as content,
                    labels(n) as labels, 
                    n.data_type as data_type,
                    n.level as level
                LIMIT 8
            """
            sample_results = self.n4j.query(sample_query)
            print("\n   Sample content from database:")
            for i, result in enumerate(sample_results, 1):
                title = (result.get('title') or 'No title')[:60]
                content = (result.get('content') or 'No content')[:120]
                labels = ', '.join(result.get('labels', []))
                data_type = result.get('data_type', 'Unknown')
                level = result.get('level', 'No level')
                print(f"     {i}. Type: {data_type} | Level: {level}")
                print(f"        Title: {title}")
                print(f"        Labels: {labels}")
                print(f"        Content: {content}...")
                print()
            
            # Check for searchable content
            searchable_query = """
                MATCH (n)
                WHERE (n.content IS NOT NULL AND length(n.content) > 10)
                   OR (n.definition IS NOT NULL AND length(n.definition) > 10)
                   OR (n.title IS NOT NULL AND length(n.title) > 3)
                RETURN count(n) as searchable_count
            """
            searchable_result = self.n4j.query(searchable_query)
            searchable_count = searchable_result[0]['searchable_count'] if searchable_result else 0
            print(f"   Nodes with searchable content: {searchable_count}")
            
            # Check Qdrant content
            print("\n🔍 QDRANT DATABASE ANALYSIS:")
            try:
                collection_info = self.qdrant_client.get_collection(qdrant_collection)
                vectors_count = collection_info.vectors_count
                print(f"   Total vectors: {vectors_count}")
                
                if vectors_count > 0:
                    # Sample some vector payloads
                    sample_vectors = self.qdrant_client.scroll(
                        collection_name=qdrant_collection,
                        limit=8,
                        with_payload=True,
                        with_vectors=False
                    )
                    print("   Sample vector payloads:")
                    for i, point in enumerate(sample_vectors[0], 1):
                        payload = point.payload
                        title = (payload.get('title') or payload.get('content') or 'No title')[:60]
                        level = payload.get('level', payload.get('data_type', 'Unknown'))
                        content_preview = (payload.get('content') or payload.get('definition') or '')[:80]
                        print(f"     {i}. [{level}] {title}")
                        if content_preview:
                            print(f"        Content: {content_preview}...")
            except Exception as e:
                print(f"   ⚠️ Could not access Qdrant: {e}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"⚠️ Database debug failed: {e}")
    
    def add_to_qdrant(self, embedding, payload, point_id=None):
        """Add a point to Qdrant vector database"""
        if point_id is None:
            point_id = str(uuid.uuid4())
        
        try:
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print(f"✓ Added point {point_id} to Qdrant collection {self.collection_name}")
            return point_id
        except Exception as e:
            print(f"✗ Failed to add point to Qdrant: {e}")
            return None


def add_literature_item(hybrid_rag, item_id, title, item_type, content, metadata=None):
    """Add literature item to middle level of hierarchy"""
    label = "MedicalBook" if item_type == "book" else "MedicalPaper"
    
    # Create node in Neo4j
    query = f"""
        MERGE (n:{label} {{id: $item_id}})
        SET n.title = $title,
            n.type = $item_type,
            n.content = $content,
            n.level = "MIDDLE"
    """
    
    params = {
        'item_id': item_id,
        'title': title,
        'item_type': item_type,
        'content': content
    }
    
    if metadata:
        for key, value in metadata.items():
            query += f", n.{key} = ${key}"
            params[key] = value
    
    query += " RETURN n"
    hybrid_rag.n4j.query(query, params)
    
    # Add to vector storage
    full_content = f"{title}: {content}"
    hybrid_rag.add_node_with_embedding(item_id, full_content, "MIDDLE", item_type, metadata)

def add_patient_record(hybrid_rag, patient_id, record_type, content, metadata=None):
    """Add patient record to top level of hierarchy"""
    
    # Create node in Neo4j
    query = """
        MERGE (n:PatientRecord {id: $patient_id})
        SET n.record_type = $record_type,
            n.content = $content,
            n.level = "TOP"
    """
    
    params = {
        'patient_id': patient_id,
        'record_type': record_type,
        'content': content
    }
    
    if metadata:
        for key, value in metadata.items():
            query += f", n.{key} = ${key}"
            params[key] = value
    
    query += " RETURN n"
    hybrid_rag.n4j.query(query, params)
    
    # Add to vector storage
    hybrid_rag.add_node_with_embedding(patient_id, content, "TOP", record_type, metadata)

def create_cross_level_relationships(hybrid_rag):
    """Create semantic relationships across hierarchy levels"""
    
    # Connect medical concepts to literature (BOTTOM -> MIDDLE)
    concept_literature_relationships = [
        ("Myocardial Infarction", "book_001", "MENTIONED_IN"),
        ("Myocardial Infarction", "paper_001", "DETAILED_IN"),
        ("Diabetes", "book_001", "MENTIONED_IN"),
        ("Diabetes", "guideline_001", "DETAILED_IN"),
        ("Hypertension", "book_001", "MENTIONED_IN"),
        ("Hypertension", "review_001", "DISCUSSED_IN"),
        ("Chest Pain", "protocol_001", "ADDRESSED_IN"),
        ("Cardiovascular Disease", "review_001", "DETAILED_IN")
    ]
    
    for concept, literature_id, rel_type in concept_literature_relationships:
        query = f"""
            MATCH (c:MedicalConcept), (l)
            WHERE c.content CONTAINS $concept 
            AND (l.id = $literature_id OR l.title CONTAINS $literature_id)
            MERGE (c)-[r:{rel_type}]->(l)
            RETURN c, r, l
        """
        try:
            hybrid_rag.n4j.query(query, {'concept': concept, 'literature_id': literature_id})
        except Exception as e:
            print(f"Warning: Could not create relationship {concept} -> {literature_id}: {e}")
    
    # Connect literature to patient records (MIDDLE -> TOP)
    literature_patient_relationships = [
        ("paper_001", "P001", "APPLIED_TO"),
        ("guideline_001", "P002", "APPLIED_TO"),
        ("protocol_001", "P001", "USED_FOR"),
        ("book_001", "P003", "REFERENCED_FOR"),
        ("review_001", "P003", "RELEVANT_TO")
    ]
    
    for literature_id, patient_id, rel_type in literature_patient_relationships:
        query = f"""
            MATCH (l), (p:PatientRecord)
            WHERE (l.id = $literature_id OR l.title CONTAINS $literature_id)
            AND p.id = $patient_id
            MERGE (l)-[r:{rel_type}]->(p)
            RETURN l, r, p
        """
        try:
            hybrid_rag.n4j.query(query, {'literature_id': literature_id, 'patient_id': patient_id})
        except Exception as e:
            print(f"Warning: Could not create relationship {literature_id} -> {patient_id}: {e}")
    
    # Connect concepts directly to patient records (BOTTOM -> TOP)
    concept_patient_relationships = [
        ("Myocardial Infarction", "P001", "DIAGNOSED_WITH"),
        ("Diabetes", "P001", "COMORBID_WITH"),
        ("Hypertension", "P001", "COMORBID_WITH"),
        ("Diabetes", "P002", "DIAGNOSED_WITH"),
        ("Hypertension", "P002", "DIAGNOSED_WITH"),
        ("Dyspnea", "P003", "SYMPTOM_OF"),
        ("Cardiovascular Disease", "P003", "DIAGNOSED_WITH")
    ]
    
    for concept, patient_id, rel_type in concept_patient_relationships:
        query = f"""
            MATCH (c:MedicalConcept), (p:PatientRecord)
            WHERE c.content CONTAINS $concept 
            AND p.id = $patient_id
            MERGE (c)-[r:{rel_type}]->(p)
            RETURN c, r, p
        """
        try:
            hybrid_rag.n4j.query(query, {'concept': concept, 'patient_id': patient_id})
        except Exception as e:
            print(f"Warning: Could not create relationship {concept} -> {patient_id}: {e}")


def setup_hybrid_rag_system(n4j):
    """Initialize the hybrid RAG system with sample data"""
    
    # Initialize hybrid system
    hybrid_rag = HybridMedicalRAG(n4j)
    
    print("🔄 Setting up hybrid Neo4j + Qdrant system...")
    
    # Populate sample data with embeddings
    populate_sample_data(n4j)
    
    # Add embeddings to existing nodes
    print("🧠 Adding embeddings to all nodes...")
    
    # Get all nodes and add embeddings
    all_nodes_query = """
        MATCH (n)
        WHERE n.level IS NOT NULL AND n.content IS NOT NULL
        RETURN n.id as node_id, n.cui as cui, n.content as content, 
               n.level as level, labels(n) as node_types
    """
    
    nodes = n4j.query(all_nodes_query)
    
    for node in nodes:
        node_id = node['node_id'] or node['cui']
        content = node['content'] 
        level = node['level']
        node_type = 'concept' if 'MedicalConcept' in node['node_types'] else \
                   'book' if 'MedicalBook' in node['node_types'] else \
                   'paper' if 'MedicalPaper' in node['node_types'] else \
                   'patient_record'
        
        hybrid_rag.add_node_with_embedding(node_id, content, level, node_type)
    
    print("✅ Hybrid RAG system setup complete!")
    return hybrid_rag


def populate_sample_data(hybrid_rag):
    """
    Populate the database with sample medical data across all 3 levels
    using the HybridMedicalRAG system
    """
    print("📊 Populating sample medical data...")
    
    # BOTTOM LEVEL: Medical Concepts
    medical_concepts = [
        ("C0027051", "Myocardial Infarction", "disorder", "Heart attack caused by blocked coronary artery"),
        ("C0011847", "Diabetes", "disorder", "Metabolic disorder characterized by high blood glucose"),
        ("C0020538", "Hypertension", "disorder", "High blood pressure condition"),
        ("C0020649", "Hypotension", "disorder", "Low blood pressure condition"),
        ("C0008031", "Chest Pain", "symptom", "Pain or discomfort in the chest area"),
        ("C0013404", "Dyspnea", "symptom", "Difficulty breathing or shortness of breath"),
        ("C0015967", "Fever", "symptom", "Elevated body temperature above normal"),
        ("C0007222", "Cardiovascular Disease", "disorder", "Disease affecting the heart and blood vessels")
    ]
    
    for cui, concept_name, semantic_type, description in medical_concepts:
        try:
            create_medical_concept(
                hybrid_rag, 
                concept_name, 
                semantic_type, 
                cui, 
                description
            )
        except Exception as e:
            print(f"Warning: Could not create concept {concept_name}: {e}")
    
    print("✓ Created medical concepts (Bottom Level)")
    
    # MIDDLE LEVEL: Medical Literature
    literature_items = [
        ("book_001", "Harrison's Principles of Internal Medicine", "book", 
         "Comprehensive textbook covering all aspects of internal medicine including cardiovascular diseases, diabetes, and hypertension."),
        ("paper_001", "ACC/AHA Guidelines for MI Management", "paper", 
         "Clinical guidelines for the diagnosis and treatment of myocardial infarction with evidence-based recommendations."),
        ("guideline_001", "Diabetes Management Protocol 2024", "guideline", 
         "Updated protocols for diabetes diagnosis, monitoring, and treatment including lifestyle interventions."),
        ("protocol_001", "Emergency Chest Pain Assessment", "protocol", 
         "Systematic approach to evaluating patients presenting with chest pain in emergency settings."),
        ("review_001", "Hypertension and Cardiovascular Risk", "review", 
         "Comprehensive review of the relationship between hypertension and cardiovascular disease outcomes.")
    ]
    
    for item_id, title, item_type, content in literature_items:
        try:
            add_literature_item(hybrid_rag, item_id, title, item_type, content)
        except Exception as e:
            print(f"Warning: Could not create literature item {title}: {e}")
    
    print("✓ Created literature items (Middle Level)")
    
    # TOP LEVEL: Sample Patient Records
    patient_records = [
        ("P001", "clinical_note", 
         "65-year-old male presents with acute chest pain, elevated troponins, and ECG changes consistent with STEMI. History of diabetes and hypertension."),
        ("P002", "test_result", 
         "Laboratory results show HbA1c 9.2%, fasting glucose 280 mg/dL, indicating poor diabetes control. Blood pressure readings consistently >160/90."),
        ("P003", "clinical_note", 
         "45-year-old female with shortness of breath and fatigue. Echocardiogram shows reduced ejection fraction. History of hypertension."),
        ("P004", "treatment_record", 
         "Post-MI treatment plan: dual antiplatelet therapy, ACE inhibitor, beta-blocker, statin. Diabetes management with metformin adjustment.")
    ]
    
    for patient_id, record_type, content in patient_records:
        try:
            add_patient_record(hybrid_rag, patient_id, record_type, content)
        except Exception as e:
            print(f"Warning: Could not create patient record {patient_id}: {e}")
    
    print("✓ Created patient records (Top Level)")
    
    # Create cross-level relationships
    create_cross_level_relationships(hybrid_rag)
    
    print("✓ Established cross-level relationships")
    print("🎉 Sample data population completed!")

def populate_sample_data_legacy(n4j):
    """Populate the database with sample data for all three levels"""
    
    print("🏗️ Setting up hierarchical structure...")
    setup_hierarchical_structure(n4j)
    
    print("📚 Creating Bottom Level - Medical Dictionary (UMLS concepts)...")
    
    # Bottom Level: Medical Dictionary Concepts
    medical_concepts = [
        ("C0015967", "Fever", "Sign or Symptom", "Body temperature above normal range"),
        ("C0018681", "Headache", "Sign or Symptom", "Pain in the head or neck area"),
        ("C0003862", "Arthritis", "Disease or Syndrome", "Inflammation of joints"),
        ("C0020538", "Hypertension", "Disease or Syndrome", "High blood pressure"),
        ("C0011847", "Diabetes", "Disease or Syndrome", "Group of disorders characterized by high blood glucose"),
        ("C0034063", "Pulmonary Edema", "Disease or Syndrome", "Fluid accumulation in lungs"),
        ("C0004096", "Asthma", "Disease or Syndrome", "Chronic inflammatory airway disease"),
        ("C0027051", "Myocardial Infarction", "Disease or Syndrome", "Heart attack"),
    ]
    
    for cui, name, sem_type, definition in medical_concepts:
        create_medical_concept(n4j, cui, name, sem_type, definition)
    
    print("📖 Creating Middle Level - Medical Literature...")
    
    # Middle Level: Medical Literature  
    literature_data = [
        ("book_001", "Harrison's Principles of Internal Medicine", "book", 
         "Comprehensive textbook covering fever, hypertension, diabetes, and cardiovascular diseases."),
        ("book_002", "Gray's Anatomy", "book",
         "Detailed anatomical reference covering body systems and structures."),
        ("paper_001", "Fever Management in Clinical Practice", "paper",
         "Research paper discussing fever symptoms, causes and treatment protocols."),
        ("paper_002", "Hypertension Guidelines 2024", "paper", 
         "Updated clinical guidelines for hypertension diagnosis and management."),
        ("paper_003", "Diabetes Complications Prevention", "paper",
         "Study on preventing complications in diabetic patients."),
    ]
    
    for doc_id, title, doc_type, content in literature_data:
        create_medical_literature(n4j, doc_id, title, doc_type, content)
    
    print("🏥 Creating Top Level - Patient Records...")
    
    # Top Level: Patient Records
    patient_records = [
        ("record_001", "patient_123", 
         "Patient presents with fever (101.2°F), headache, and fatigue. Vital signs stable.", "clinical_note"),
        ("record_002", "patient_124",
         "Hypertension diagnosis confirmed. BP 160/95. Started on ACE inhibitor.", "clinical_note"),
        ("record_003", "patient_125", 
         "Type 2 diabetes mellitus. HbA1c 8.2%. Adjusting medication regimen.", "clinical_note"),
        ("record_004", "patient_126",
         "Joint pain and swelling consistent with rheumatoid arthritis.", "clinical_note"),
    ]
    
    for record_id, patient_id, content, record_type in patient_records:
        create_patient_record(n4j, record_id, patient_id, content, record_type)
    
    print("🔗 Creating cross-level semantic relationships...")
    
    # Create semantic relationships between levels
    relationships = [
        # Patient records to Medical concepts
        ("record_001", "C0015967", "MENTIONS", "TOP", "BOTTOM"),  # fever record -> fever concept
        ("record_001", "C0018681", "MENTIONS", "TOP", "BOTTOM"),  # fever record -> headache concept  
        ("record_002", "C0020538", "MENTIONS", "TOP", "BOTTOM"),  # hypertension record -> hypertension concept
        ("record_003", "C0011847", "MENTIONS", "TOP", "BOTTOM"),  # diabetes record -> diabetes concept
        ("record_004", "C0003862", "MENTIONS", "TOP", "BOTTOM"),  # arthritis record -> arthritis concept
        
        # Medical literature to Medical concepts  
        ("book_001", "C0015967", "DESCRIBES", "MIDDLE", "BOTTOM"),  # Harrison's -> fever
        ("book_001", "C0020538", "DESCRIBES", "MIDDLE", "BOTTOM"),  # Harrison's -> hypertension
        ("book_001", "C0011847", "DESCRIBES", "MIDDLE", "BOTTOM"),  # Harrison's -> diabetes
        ("paper_001", "C0015967", "FOCUSES_ON", "MIDDLE", "BOTTOM"),  # fever paper -> fever concept
        ("paper_002", "C0020538", "FOCUSES_ON", "MIDDLE", "BOTTOM"),  # hypertension paper -> hypertension concept
        ("paper_003", "C0011847", "FOCUSES_ON", "MIDDLE", "BOTTOM"),  # diabetes paper -> diabetes concept
        
        # Patient records to Medical literature (evidence-based connections)
        ("record_001", "paper_001", "SUPPORTED_BY", "TOP", "MIDDLE"),  # fever record -> fever paper
        ("record_002", "paper_002", "SUPPORTED_BY", "TOP", "MIDDLE"),  # hypertension record -> hypertension paper  
        ("record_003", "paper_003", "SUPPORTED_BY", "TOP", "MIDDLE"),  # diabetes record -> diabetes paper
    ]
    
    for from_id, to_id, rel_type, from_level, to_level in relationships:
        create_hierarchical_relationship(n4j, from_id, to_id, rel_type, from_level, to_level)
    
    print("✅ Sample hierarchical database populated successfully!")
    print("📊 Structure: Bottom (8 concepts) -> Middle (5 literature) -> Top (4 patient records)")

