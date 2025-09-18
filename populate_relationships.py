#!/usr/bin/env python3
"""
Database Relationship Populator
Creates relationships between nodes in the hierarchical structure
"""

import sys
from utils import HybridMedicalRAG, Neo4jGraph, neo4j_url, neo4j_username, neo4j_password

def populate_graph_relationships():
    """Add relationships between nodes in the hierarchical structure"""
    
    print("🔗 Populating Graph Relationships in Hierarchical Structure")
    print("=" * 60)
    
    try:
        # Initialize systems
        n4j = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Step 1: Create relationships between TOP -> MIDDLE levels
        print("1. Creating TOP -> MIDDLE level relationships...")
        
        top_to_middle_query = """
        MATCH (top) WHERE top.level = 'top'
        MATCH (middle) WHERE middle.level = 'middle'
        WITH top, middle, 
             CASE 
                WHEN middle.content CONTAINS 'cardiovascular' OR middle.content CONTAINS 'cardiac' OR middle.content CONTAINS 'heart'
                     THEN CASE WHEN top.content CONTAINS 'Cardiovascular' THEN 1.0 ELSE 0.3 END
                WHEN middle.content CONTAINS 'diabetes' OR middle.content CONTAINS 'endocrine' OR middle.content CONTAINS 'metabolic'
                     THEN CASE WHEN top.content CONTAINS 'Endocrine' THEN 1.0 ELSE 0.3 END
                WHEN middle.content CONTAINS 'respiratory' OR middle.content CONTAINS 'pulmonary' OR middle.content CONTAINS 'lung'
                     THEN CASE WHEN top.content CONTAINS 'Respiratory' THEN 1.0 ELSE 0.3 END
                WHEN middle.content CONTAINS 'kidney' OR middle.content CONTAINS 'renal' OR middle.content CONTAINS 'nephro'
                     THEN CASE WHEN top.content CONTAINS 'Renal' THEN 1.0 ELSE 0.3 END
                ELSE 0.5
             END AS similarity
        WHERE similarity >= 0.5
        MERGE (top)-[r:CONTAINS]->(middle)
        SET r.similarity = similarity, r.relationship_type = 'hierarchical'
        RETURN count(r) as relationships_created
        """
        
        result1 = n4j.query(top_to_middle_query)
        if result1:
            print(f"   ✅ Created {result1[0]['relationships_created']} TOP->MIDDLE relationships")
        
        # Step 2: Create relationships between MIDDLE -> BOTTOM levels  
        print("2. Creating MIDDLE -> BOTTOM level relationships...")
        
        middle_to_bottom_query = """
        MATCH (middle) WHERE middle.level = 'middle'
        MATCH (bottom) WHERE bottom.level = 'bottom'
        WITH middle, bottom,
             CASE
                WHEN bottom.content CONTAINS 'hypertension' OR bottom.content CONTAINS 'blood pressure'
                     THEN CASE WHEN middle.content CONTAINS 'cardiovascular' OR middle.content CONTAINS 'hypertension' THEN 0.9 ELSE 0.3 END
                WHEN bottom.content CONTAINS 'diabetes' OR bottom.content CONTAINS 'glucose' OR bottom.content CONTAINS 'insulin'
                     THEN CASE WHEN middle.content CONTAINS 'diabetes' OR middle.content CONTAINS 'endocrine' THEN 0.9 ELSE 0.3 END
                WHEN bottom.content CONTAINS 'asthma' OR bottom.content CONTAINS 'respiratory' OR bottom.content CONTAINS 'lung'
                     THEN CASE WHEN middle.content CONTAINS 'respiratory' OR middle.content CONTAINS 'pulmonary' THEN 0.9 ELSE 0.3 END
                WHEN bottom.content CONTAINS 'kidney' OR bottom.content CONTAINS 'renal'
                     THEN CASE WHEN middle.content CONTAINS 'kidney' OR middle.content CONTAINS 'renal' THEN 0.9 ELSE 0.3 END
                ELSE 0.4
             END AS similarity
        WHERE similarity >= 0.4
        MERGE (middle)-[r:DEFINES]->(bottom)
        SET r.similarity = similarity, r.relationship_type = 'hierarchical'
        RETURN count(r) as relationships_created
        """
        
        result2 = n4j.query(middle_to_bottom_query)
        if result2:
            print(f"   ✅ Created {result2[0]['relationships_created']} MIDDLE->BOTTOM relationships")
        
        # Step 3: Create cross-level relationships (TOP -> BOTTOM for highly related concepts)
        print("3. Creating TOP -> BOTTOM cross-level relationships...")
        
        cross_level_query = """
        MATCH (top) WHERE top.level = 'top'
        MATCH (bottom) WHERE bottom.level = 'bottom'
        WITH top, bottom,
             CASE
                WHEN (top.content CONTAINS 'Cardiovascular' AND 
                     (bottom.content CONTAINS 'hypertension' OR bottom.content CONTAINS 'cardiac' OR bottom.content CONTAINS 'heart'))
                     THEN 0.8
                WHEN (top.content CONTAINS 'Endocrine' AND 
                     (bottom.content CONTAINS 'diabetes' OR bottom.content CONTAINS 'insulin' OR bottom.content CONTAINS 'glucose'))
                     THEN 0.8
                WHEN (top.content CONTAINS 'Respiratory' AND 
                     (bottom.content CONTAINS 'asthma' OR bottom.content CONTAINS 'copd' OR bottom.content CONTAINS 'lung'))
                     THEN 0.8
                ELSE 0.0
             END AS similarity
        WHERE similarity >= 0.7
        MERGE (top)-[r:RELATES_TO]->(bottom)
        SET r.similarity = similarity, r.relationship_type = 'cross_level'
        RETURN count(r) as relationships_created
        """
        
        result3 = n4j.query(cross_level_query)
        if result3:
            print(f"   ✅ Created {result3[0]['relationships_created']} TOP->BOTTOM cross-level relationships")
        
        # Step 4: Create peer relationships within the same level
        print("4. Creating peer relationships within levels...")
        
        peer_relationships_query = """
        MATCH (a), (b) 
        WHERE a.level = b.level AND a <> b AND a.level = 'bottom'
        WITH a, b,
             CASE
                WHEN (a.content CONTAINS 'hypertension' AND b.content CONTAINS 'cardiovascular') OR
                     (a.content CONTAINS 'diabetes' AND b.content CONTAINS 'glucose') OR
                     (a.content CONTAINS 'asthma' AND b.content CONTAINS 'respiratory')
                     THEN 0.6
                ELSE 0.0
             END AS similarity
        WHERE similarity >= 0.5
        WITH a, b, similarity
        LIMIT 20
        MERGE (a)-[r:RELATED_TO]->(b)
        SET r.similarity = similarity, r.relationship_type = 'peer'
        RETURN count(r) as relationships_created
        """
        
        result4 = n4j.query(peer_relationships_query)
        if result4:
            print(f"   ✅ Created {result4[0]['relationships_created']} peer relationships")
        
        # Step 5: Verify relationships created
        print("\n5. Verifying relationship creation...")
        
        verification_query = """
        MATCH (a)-[r]->(b)
        WHERE a.level IS NOT NULL AND b.level IS NOT NULL
        RETURN type(r) as relationship_type, 
               a.level + '->' + b.level as level_connection,
               count(r) as count
        ORDER BY count DESC
        """
        
        verification_results = n4j.query(verification_query)
        if verification_results:
            print("   ✅ Relationship verification:")
            for result in verification_results:
                print(f"      🔗 {result['relationship_type']}: {result['level_connection']} ({result['count']} connections)")
        
        # Summary
        total_relationships = sum([
            result1[0]['relationships_created'] if result1 else 0,
            result2[0]['relationships_created'] if result2 else 0,
            result3[0]['relationships_created'] if result3 else 0,
            result4[0]['relationships_created'] if result4 else 0
        ])
        
        print(f"\n🎉 Relationship population complete!")
        print(f"📊 Total relationships created: {total_relationships}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error populating relationships: {str(e)}")
        return False

if __name__ == "__main__":
    success = populate_graph_relationships()
    sys.exit(0 if success else 1)