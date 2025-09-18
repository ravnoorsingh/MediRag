#!/usr/bin/env python3
"""
Database Structure Verification Script
Tests both hierarchical and hybrid database implementations in the Clinical Decision Support System
"""

import sys
import json
from datetime import datetime
import traceback

def test_database_structures():
    """Test both hierarchical and hybrid database structures"""
    
    print("🔍 Clinical Decision Support System - Database Structure Test")
    print("=" * 80)
    
    test_results = {
        "neo4j_connection": False,
        "qdrant_connection": False,
        "hybrid_rag_init": False,
        "hierarchical_structure": False,
        "vector_search": False,
        "graph_queries": False,
        "semantic_search": False,
        "clinical_integration": False
    }
    
    try:
        # Test 1: Database Connections
        print("1. Testing Database Connections...")
        print("-" * 40)
        
        # Test Neo4j connection
        try:
            from utils import Neo4jGraph, neo4j_url, neo4j_username, neo4j_password
            n4j = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password
            )
            
            # Test basic query
            result = n4j.query("RETURN 1 as test")
            if result:
                print("   ✅ Neo4j connection: SUCCESSFUL")
                test_results["neo4j_connection"] = True
            else:
                print("   ❌ Neo4j connection: Failed - No response")
                
        except Exception as e:
            print(f"   ❌ Neo4j connection: Failed - {str(e)}")
        
        # Test Qdrant connection
        try:
            from utils import get_qdrant_client, qdrant_collection
            qdrant_client = get_qdrant_client()
            
            # Test collection exists
            collections = qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if qdrant_collection in collection_names:
                print("   ✅ Qdrant connection: SUCCESSFUL")
                print(f"   📋 Collection '{qdrant_collection}' found")
                test_results["qdrant_connection"] = True
            else:
                print(f"   ⚠️  Qdrant connected but collection '{qdrant_collection}' not found")
                print(f"   📋 Available collections: {collection_names}")
                
        except Exception as e:
            print(f"   ❌ Qdrant connection: Failed - {str(e)}")
        
        # Test 2: Hybrid RAG System
        print("\n2. Testing Hybrid RAG System...")
        print("-" * 40)
        
        try:
            from utils import HybridMedicalRAG
            hybrid_rag = HybridMedicalRAG()
            
            print("   ✅ HybridMedicalRAG initialization: SUCCESSFUL")
            print(f"   📊 Vector dimensions: {hybrid_rag.vector_dim}")
            print(f"   🗃️  Collection name: {hybrid_rag.collection_name}")
            test_results["hybrid_rag_init"] = True
            
        except Exception as e:
            print(f"   ❌ HybridMedicalRAG initialization: Failed - {str(e)}")
            hybrid_rag = None
        
        # Test 3: Hierarchical Structure
        print("\n3. Testing Hierarchical Structure...")
        print("-" * 40)
        
        if test_results["neo4j_connection"]:
            try:
                # Check for hierarchical levels
                hierarchy_query = """
                MATCH (n)
                WHERE n.level IS NOT NULL
                RETURN DISTINCT n.level as level, count(n) as count
                ORDER BY level
                """
                
                levels = n4j.query(hierarchy_query)
                if levels:
                    print("   ✅ Hierarchical structure found:")
                    for level in levels:
                        print(f"      📊 Level '{level['level']}': {level['count']} nodes")
                    test_results["hierarchical_structure"] = True
                else:
                    print("   ⚠️  No hierarchical levels found in database")
                    
            except Exception as e:
                print(f"   ❌ Hierarchical structure test: Failed - {str(e)}")
        else:
            print("   ⏭️  Skipped (Neo4j not connected)")
        
        # Test 4: Vector Search Functionality
        print("\n4. Testing Vector Search...")
        print("-" * 40)
        
        if hybrid_rag and test_results["qdrant_connection"]:
            try:
                # Test vector search
                test_query = "hypertension treatment"
                print(f"   🔍 Testing search query: '{test_query}'")
                
                # Get embedding for test query
                from utils import get_embedding
                query_embedding = get_embedding(test_query)
                print(f"   📊 Query embedding generated: {len(query_embedding)} dimensions")
                
                # Test Qdrant search
                search_results = hybrid_rag.qdrant_client.search(
                    collection_name=hybrid_rag.collection_name,
                    query_vector=query_embedding,
                    limit=3
                )
                
                if search_results:
                    print(f"   ✅ Vector search: SUCCESSFUL ({len(search_results)} results)")
                    for i, result in enumerate(search_results[:2]):
                        print(f"      📄 Result {i+1}: Score {result.score:.3f}")
                    test_results["vector_search"] = True
                else:
                    print("   ⚠️  Vector search returned no results")
                    
            except Exception as e:
                print(f"   ❌ Vector search test: Failed - {str(e)}")
        else:
            print("   ⏭️  Skipped (Hybrid RAG not initialized)")
        
        # Test 5: Graph Queries
        print("\n5. Testing Graph Queries...")
        print("-" * 40)
        
        if test_results["neo4j_connection"]:
            try:
                # Test medical concept relationships
                relationship_query = """
                MATCH (a)-[r]->(b)
                WHERE a.level IS NOT NULL AND b.level IS NOT NULL
                RETURN type(r) as relationship_type, count(r) as count
                LIMIT 5
                """
                
                relationships = n4j.query(relationship_query)
                if relationships:
                    print("   ✅ Graph relationships found:")
                    for rel in relationships:
                        print(f"      🔗 {rel['relationship_type']}: {rel['count']} connections")
                    test_results["graph_queries"] = True
                else:
                    # Check if we have any relationships at all
                    all_rels_query = "MATCH (a)-[r]->(b) RETURN count(r) as total_relationships"
                    all_rels = n4j.query(all_rels_query)
                    if all_rels and all_rels[0]['total_relationships'] > 0:
                        print(f"   ⚠️  Found {all_rels[0]['total_relationships']} relationships but none with hierarchy levels")
                    else:
                        print("   ⚠️  No graph relationships found - database may need relationship population")
                    
            except Exception as e:
                print(f"   ❌ Graph query test: Failed - {str(e)}")
        else:
            print("   ⏭️  Skipped (Neo4j not connected)")
        
        # Test 6: Semantic Search Across Levels
        print("\n6. Testing Semantic Search Across Levels...")
        print("-" * 40)
        
        if hybrid_rag and test_results["hybrid_rag_init"]:
            try:
                # Test the hybrid search method
                test_query = "diabetes management guidelines"
                print(f"   🔍 Testing semantic search: '{test_query}'")
                
                # Check if method exists
                if hasattr(hybrid_rag, 'semantic_search_across_levels'):
                    search_results = hybrid_rag.semantic_search_across_levels(
                        query_text=test_query,  # Fixed parameter name
                        top_k=5
                    )
                    
                    if search_results:
                        print(f"   ✅ Semantic search: SUCCESSFUL")
                        
                        # Handle different result formats safely
                        if isinstance(search_results, dict):
                            vector_results = search_results.get('vector_results', [])
                            graph_results = search_results.get('graph_results', [])
                            result_count = len(vector_results) + len(graph_results)
                            print(f"   📊 Found {len(vector_results)} vector + {len(graph_results)} graph results")
                        else:
                            result_count = len(search_results) if hasattr(search_results, '__len__') else 0
                            print(f"   📊 Found {result_count} total results")
                        
                        if result_count > 0:
                            test_results["semantic_search"] = True
                        else:
                            print("   ⚠️  Semantic search returned empty results")
                    else:
                        print("   ⚠️  Semantic search returned no results")
                else:
                    print("   ⚠️  semantic_search_across_levels method not found")
                    
            except Exception as e:
                print(f"   ❌ Semantic search test: Failed - {str(e)}")
        else:
            print("   ⏭️  Skipped (Hybrid RAG not initialized)")
        
        # Test 7: Clinical Decision Engine Integration
        print("\n7. Testing Clinical Decision Engine Integration...")
        print("-" * 40)
        
        try:
            from clinical_decision_engine import ClinicalDecisionEngine
            
            # Initialize clinical engine
            engine = ClinicalDecisionEngine()
            print("   ✅ Clinical Decision Engine initialized")
            
            # Check if it has hybrid_rag
            if hasattr(engine, 'hybrid_rag') and engine.hybrid_rag:
                print("   ✅ Clinical engine has hybrid RAG system")
                test_results["clinical_integration"] = True
            else:
                print("   ⚠️  Clinical engine missing hybrid RAG integration")
                
        except Exception as e:
            print(f"   ❌ Clinical engine test: Failed - {str(e)}")
        
        # Test Summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(test_results.values())
        total = len(test_results)
        
        print(f"Overall Status: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        print()
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {test_name.replace('_', ' ').title()}")
        
        print("\n🔧 RECOMMENDATIONS:")
        
        if not test_results["neo4j_connection"]:
            print("   • Start Neo4j database: Check if Neo4j is running on bolt://localhost:7687")
        
        if not test_results["qdrant_connection"]:
            print("   • Start Qdrant database: Check if Qdrant is running on http://localhost:6333")
        
        if not test_results["hierarchical_structure"]:
            print("   • Load hierarchical data: Run sample_data_loader.py to populate hierarchy")
        
        if not test_results["vector_search"]:
            print("   • Populate vector data: Ensure Qdrant collection has embedded vectors")
        
        if passed == total:
            print("   🎉 All systems operational! The Clinical Decision Support System is ready.")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Critical error in database testing: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return test_results

if __name__ == "__main__":
    results = test_database_structures()
    
    # Exit code based on results
    passed = sum(results.values())
    total = len(results)
    
    if passed >= total * 0.75:  # 75% pass rate
        sys.exit(0)
    else:
        sys.exit(1)