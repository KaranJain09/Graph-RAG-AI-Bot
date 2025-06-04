# from neo4j import GraphDatabase
# from typing import List, Dict, Optional
# import json
# from sentence_transformers import SentenceTransformer
# import numpy as np

# class Neo4jManager:
#     def __init__(self, uri: str, username: str, password: str, embedding_model: str = "all-MiniLM-L6-v2"):
#         self.driver = GraphDatabase.driver(uri, auth=(username, password))
#         self.embedding_model = SentenceTransformer(embedding_model)
#         self._create_constraints()
    
#     def __del__(self):
#         if hasattr(self, 'driver'):
#             self.driver.close()
    
#     def _create_constraints(self):
#         """Create constraints and indexes for better performance"""
#         with self.driver.session() as session:
#             # Create constraints
#             constraints = [
#                 "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.url IS UNIQUE",
#                 "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
#                 "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.embedding_vector)",
#                 "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.header)",
#                 "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)"
#             ]
            
#             for constraint in constraints:
#                 try:
#                     session.run(constraint)
#                 except Exception as e:
#                     print(f"Constraint/Index creation warning: {e}")
    
#     def store_document_chunks(self, chunks: List[Dict], url: str, title: str) -> bool:
#         """
#         Store document chunks in Neo4j with proper relationships
        
#         Args:
#             chunks: List of chunk dictionaries
#             url: Source URL
#             title: Document title
            
#         Returns:
#             bool: Success status
#         """
#         try:
#             with self.driver.session() as session:
#                 # First, create or merge document node
#                 session.run("""
#                     MERGE (d:Document {url: $url})
#                     SET d.title = $title,
#                         d.total_chunks = $total_chunks,
#                         d.created_at = datetime(),
#                         d.updated_at = datetime()
#                 """, url=url, title=title, total_chunks=len(chunks))
                
#                 # Process chunks in batches
#                 batch_size = 10
#                 for i in range(0, len(chunks), batch_size):
#                     batch = chunks[i:i + batch_size]
#                     self._store_chunk_batch(session, batch, url)
                
#                 # Create relationships between sequential chunks
#                 self._create_chunk_relationships(session, chunks, url)
                
#                 return True
                
#         except Exception as e:
#             print(f"Error storing document chunks: {e}")
#             return False
    
#     def _store_chunk_batch(self, session, chunks: List[Dict], url: str):
#         """Store a batch of chunks"""
#         for chunk in chunks:
#             # Generate embedding
#             embedding = self.embedding_model.encode(chunk['content']).tolist()
            
#             # Store chunk node
#             session.run("""
#                 MERGE (c:Chunk {chunk_id: $chunk_id})
#                 SET c.content = $content,
#                     c.header = $header,
#                     c.level = $level,
#                     c.token_count = $token_count,
#                     c.chunk_index = $chunk_index,
#                     c.total_chunks = $total_chunks,
#                     c.hash = $hash,
#                     c.embedding_vector = $embedding,
#                     c.created_at = datetime(),
#                     c.updated_at = datetime()
#             """, **chunk, embedding=embedding)
            
#             # Create relationship to document
#             session.run("""
#                 MATCH (d:Document {url: $url})
#                 MATCH (c:Chunk {chunk_id: $chunk_id})
#                 MERGE (d)-[:CONTAINS]->(c)
#             """, url=url, chunk_id=chunk['chunk_id'])
    
#     def _create_chunk_relationships(self, session, chunks: List[Dict], url: str):
#         """Create relationships between chunks"""
#         for i, chunk in enumerate(chunks):
#             current_chunk_id = chunk['chunk_id']
            
#             # Link to next chunk
#             if i < len(chunks) - 1:
#                 next_chunk_id = chunks[i + 1]['chunk_id']
#                 session.run("""
#                     MATCH (c1:Chunk {chunk_id: $current_id})
#                     MATCH (c2:Chunk {chunk_id: $next_id})
#                     MERGE (c1)-[:NEXT]->(c2)
#                 """, current_id=current_chunk_id, next_id=next_chunk_id)
            
#             # Link to previous chunk
#             if i > 0:
#                 prev_chunk_id = chunks[i - 1]['chunk_id']
#                 session.run("""
#                     MATCH (c1:Chunk {chunk_id: $current_id})
#                     MATCH (c2:Chunk {chunk_id: $prev_id})
#                     MERGE (c1)-[:PREVIOUS]->(c2)
#                 """, current_id=current_chunk_id, prev_id=prev_chunk_id)
            
#             # Link chunks with same header (semantic relationship)
#             session.run("""
#                 MATCH (c1:Chunk {chunk_id: $current_id})
#                 MATCH (c2:Chunk)
#                 WHERE c1.header = c2.header 
#                   AND c1.chunk_id <> c2.chunk_id
#                   AND c1.level = c2.level
#                 MERGE (c1)-[:SAME_SECTION]->(c2)
#             """, current_id=current_chunk_id)
    
#     def semantic_search(self, query: str, url: Optional[str] = None, top_k: int = 7) -> List[Dict]:
#         """
#         Perform semantic search to retrieve relevant chunks
#         """
#         try:
#             # Generate query embedding
#             query_embedding = self.embedding_model.encode(query).tolist()
#             query_embedding_np = np.array(query_embedding)

#             with self.driver.session() as session:
#                 # Fetch all relevant chunks and their embeddings
#                 cypher_query = """
#                     MATCH (c:Chunk)
#                 """
#                 parameters = {}
#                 if url:
#                     cypher_query += """
#                     MATCH (d:Document {url: $url})-[:CONTAINS]->(c)
#                     """
#                     parameters['url'] = url

#                 cypher_query += """
#                     RETURN c.chunk_id as chunk_id,
#                            c.content as content,
#                            c.header as header,
#                            c.level as level,
#                            c.chunk_index as chunk_index,
#                            c.token_count as token_count,
#                            c.embedding_vector as embedding
#                 """

#                 result = session.run(cypher_query, parameters)
#                 chunks = []
#                 for record in result:
#                     embedding = record.get("embedding")
#                     if embedding is not None:
#                         chunks.append({
#                             "chunk_id": record["chunk_id"],
#                             "content": record["content"],
#                             "header": record["header"],
#                             "level": record["level"],
#                             "chunk_index": record["chunk_index"],
#                             "token_count": record["token_count"],
#                             "embedding": np.array(embedding)
#                         })

#                 # Compute cosine similarity in Python
#                 for chunk in chunks:
#                     emb = chunk["embedding"]
#                     if emb is not None and len(emb) == len(query_embedding_np):
#                         sim = np.dot(query_embedding_np, emb) / (np.linalg.norm(query_embedding_np) * np.linalg.norm(emb) + 1e-8)
#                     else:
#                         sim = 0.0
#                     chunk["similarity"] = float(sim)

#                 # Filter and sort by similarity
#                 filtered_chunks = [c for c in chunks if c["similarity"] > 0.3]
#                 filtered_chunks.sort(key=lambda x: x["similarity"], reverse=True)
#                 top_chunks = filtered_chunks[:top_k]

#                 # Add related chunks
#                 for chunk in top_chunks:
#                     chunk["related_chunks"] = self._get_related_chunks(session, chunk["chunk_id"])
#                     # Remove embedding from output
#                     del chunk["embedding"]

#                 # If not enough, fallback to keyword search
#                 if len(top_chunks) < top_k:
#                     keyword_chunks = self._keyword_search(session, query, url, top_k - len(top_chunks))
#                     existing_ids = {chunk['chunk_id'] for chunk in top_chunks}
#                     for chunk in keyword_chunks:
#                         if chunk['chunk_id'] not in existing_ids:
#                             top_chunks.append(chunk)

#                 return top_chunks[:top_k]

#         except Exception as e:
#             print(f"Error in semantic search: {e}")
#             return self._fallback_search(query, url, top_k)
    
#     def _get_related_chunks(self, session, chunk_id: str) -> List[str]:
#         """Get related chunk IDs for context"""
#         result = session.run("""
#             MATCH (c:Chunk {chunk_id: $chunk_id})
#             OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
#             OPTIONAL MATCH (c)-[:PREVIOUS]->(prev:Chunk)
#             OPTIONAL MATCH (c)-[:SAME_SECTION]->(same:Chunk)
#             RETURN COLLECT(DISTINCT next.chunk_id) + 
#                    COLLECT(DISTINCT prev.chunk_id) + 
#                    COLLECT(DISTINCT same.chunk_id) as related_ids
#         """, chunk_id=chunk_id)
        
#         record = result.single()
#         if record and record['related_ids']:
#             return [rid for rid in record['related_ids'] if rid is not None]
#         return []
    
#     def _keyword_search(self, session, query: str, url: Optional[str], limit: int) -> List[Dict]:
#         """Fallback keyword search"""
#         query_words = query.lower().split()
        
#         cypher_query = """
#             MATCH (c:Chunk)
#         """
        
#         parameters = {'limit': limit}
        
#         if url:
#             cypher_query += """
#             MATCH (d:Document {url: $url})-[:CONTAINS]->(c)
#             """
#             parameters['url'] = url
        
#         # Create WHERE conditions for keyword matching
#         where_conditions = []
#         for i, word in enumerate(query_words[:5]):  # Limit to 5 words
#             param_name = f'word_{i}'
#             where_conditions.append(f"toLower(c.content) CONTAINS ${param_name}")
#             parameters[param_name] = word
        
#         if where_conditions:
#             cypher_query += f"""
#             WHERE {' OR '.join(where_conditions)}
#             RETURN c.chunk_id as chunk_id,
#                    c.content as content,
#                    c.header as header,
#                    c.level as level,
#                    c.chunk_index as chunk_index,
#                    c.token_count as token_count,
#                    0.5 as similarity
#             ORDER BY c.chunk_index
#             LIMIT $limit
#             """
            
#             result = session.run(cypher_query, parameters)
#             return [dict(record) for record in result]
        
#         return []
    
#     def _fallback_search(self, query: str, url: Optional[str], top_k: int) -> List[Dict]:
#         """Simple fallback search without embeddings"""
#         try:
#             with self.driver.session() as session:
#                 cypher_query = """
#                     MATCH (c:Chunk)
#                 """
                
#                 parameters = {'top_k': top_k}
                
#                 if url:
#                     cypher_query += """
#                     MATCH (d:Document {url: $url})-[:CONTAINS]->(c)
#                     """
#                     parameters['url'] = url
                
#                 cypher_query += """
#                     WHERE toLower(c.content) CONTAINS toLower($query)
#                        OR toLower(c.header) CONTAINS toLower($query)
#                     RETURN c.chunk_id as chunk_id,
#                            c.content as content,
#                            c.header as header,
#                            c.level as level,
#                            c.chunk_index as chunk_index,
#                            c.token_count as token_count,
#                            0.4 as similarity
#                     ORDER BY c.chunk_index
#                     LIMIT $top_k
#                 """
                
#                 parameters['query'] = query
#                 result = session.run(cypher_query, parameters)
#                 return [dict(record) for record in result]
                
#         except Exception as e:
#             print(f"Error in fallback search: {e}")
#             return []
    
#     def get_document_info(self, url: str) -> Optional[Dict]:
#         """Get document information"""
#         try:
#             with self.driver.session() as session:
#                 result = session.run("""
#                     MATCH (d:Document {url: $url})
#                     RETURN d.title as title,
#                            d.total_chunks as total_chunks,
#                            d.created_at as created_at,
#                            d.updated_at as updated_at
#                 """, url=url)
                
#                 record = result.single()
#                 if record:
#                     return dict(record)
#                 return None
                
#         except Exception as e:
#             print(f"Error getting document info: {e}")
#             return None
    
#     def delete_document(self, url: str) -> bool:
#         """Delete document and all its chunks"""
#         try:
#             with self.driver.session() as session:
#                 session.run("""
#                     MATCH (d:Document {url: $url})
#                     OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
#                     DETACH DELETE d, c
#                 """, url=url)
#                 return True
                
#         except Exception as e:
#             print(f"Error deleting document: {e}")
#             return False
    
#     def get_all_documents(self) -> List[Dict]:
#         """Get all stored documents"""
#         try:
#             with self.driver.session() as session:
#                 result = session.run("""
#                     MATCH (d:Document)
#                     RETURN d.url as url,
#                            d.title as title,
#                            d.total_chunks as total_chunks,
#                            d.created_at as created_at
#                     ORDER BY d.created_at DESC
#                 """)
                
#                 return [dict(record) for record in result]
                
#         except Exception as e:
#             print(f"Error getting documents: {e}")
#             return []
    
#     def test_connection(self) -> bool:
#         """Test Neo4j connection"""
#         try:
#             with self.driver.session() as session:
#                 result = session.run("RETURN 1 as test")
#                 return result.single()["test"] == 1
#         except Exception as e:
#             print(f"Connection test failed: {e}")
#             return False


from neo4j import GraphDatabase
from typing import List, Dict, Optional, Tuple
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import re
from collections import defaultdict

class Neo4jManager:
    def __init__(self, uri: str, username: str, password: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_model = SentenceTransformer(embedding_model)
        self._create_constraints()
    
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def _create_constraints(self):
        """Create optimized constraints and indexes"""
        with self.driver.session() as session:
            constraints = [
                # Core constraints
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.url IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
                
                # Performance indexes
                "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.similarity_score)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.level)",
                "CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.title)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.keywords)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)",
                
                # Vector index for semantic search (if supported)
                # "CREATE VECTOR INDEX IF NOT EXISTS chunk_embeddings FOR (c:Chunk) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint/Index creation note: {e}")
    
    def store_document_chunks(self, chunks: List[Dict], url: str, title: str) -> bool:
        """
        Store document chunks with intelligent relationships
        """
        try:
            with self.driver.session() as session:
                # Start transaction
                with session.begin_transaction() as tx:
                    # 1. Create document node
                    self._create_document_node(tx, url, title, len(chunks))
                    
                    # 2. Analyze document structure
                    sections, topics = self._analyze_document_structure(chunks)
                    
                    # 3. Create section nodes
                    self._create_section_nodes(tx, sections, url)
                    
                    # 4. Create topic nodes
                    self._create_topic_nodes(tx, topics, url)
                    
                    # 5. Store chunks with embeddings
                    self._store_chunks_with_embeddings(tx, chunks, url)
                    
                    # 6. Create intelligent relationships
                    self._create_intelligent_relationships(tx, chunks, sections, topics, url)
                    
                    # 7. Create semantic clusters
                    self._create_semantic_clusters(tx, chunks, url)
                
                return True
                
        except Exception as e:
            print(f"Error storing document chunks: {e}")
            return False
    
    def _create_document_node(self, tx, url: str, title: str, total_chunks: int):
        """Create or update document node"""
        tx.run("""
            MERGE (d:Document {url: $url})
            SET d.title = $title,
                d.total_chunks = $total_chunks,
                d.created_at = datetime(),
                d.updated_at = datetime(),
                d.word_count = $word_count,
                d.avg_chunk_size = $avg_chunk_size
        """, url=url, title=title, total_chunks=total_chunks, 
             word_count=0, avg_chunk_size=0)
    
    def _analyze_document_structure(self, chunks: List[Dict]) -> Tuple[Dict, Dict]:
        """Analyze document to identify sections and topics"""
        sections = {}
        topics = defaultdict(list)
        
        # Group by headers/sections
        for chunk in chunks:
            header = chunk.get('header', 'Unknown')
            level = chunk.get('level', 1)
            
            # Create section identifier
            section_id = hashlib.md5(f"{header}_{level}".encode()).hexdigest()[:12]
            
            if section_id not in sections:
                sections[section_id] = {
                    'section_id': section_id,
                    'title': header,
                    'level': level,
                    'chunk_count': 0,
                    'keywords': self._extract_keywords(header)
                }
            
            sections[section_id]['chunk_count'] += 1
            
            # Extract topics from content
            chunk_topics = self._extract_topics(chunk['content'])
            for topic in chunk_topics:
                topics[topic].append(chunk['chunk_id'])
        
        # Convert topics to proper format
        topic_dict = {}
        for topic, chunk_ids in topics.items():
            if len(chunk_ids) >= 2:  # Only topics that appear in multiple chunks
                topic_id = hashlib.md5(topic.encode()).hexdigest()[:12]
                topic_dict[topic_id] = {
                    'topic_id': topic_id,
                    'name': topic,
                    'chunk_ids': chunk_ids,
                    'keywords': self._extract_keywords(topic)
                }
        
        return sections, topic_dict
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'are', 'is', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
        keywords = [word for word in words if word not in stop_words]
        return list(set(keywords))[:10]  # Top 10 unique keywords
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content using simple NLP"""
        # Extract noun phrases and important terms
        sentences = content.split('.')
        topics = []
        
        for sentence in sentences:
            # Look for patterns like "X is", "X are", "X can", etc.
            patterns = [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|can|will|has|have)',
                r'(?:the|The)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)',
                r'([A-Z][a-z]+)\s+(?:system|method|process|approach|technique)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, sentence)
                topics.extend(matches)
        
        # Clean and filter topics
        clean_topics = []
        for topic in topics:
            if isinstance(topic, str) and 3 <= len(topic) <= 50:
                clean_topics.append(topic.strip())
        
        return list(set(clean_topics))[:5]  # Top 5 unique topics
    
    def _create_section_nodes(self, tx, sections: Dict, url: str):
        """Create section nodes"""
        for section_id, section_data in sections.items():
            tx.run("""
                MERGE (s:Section {section_id: $section_id})
                SET s.title = $title,
                    s.level = $level,
                    s.chunk_count = $chunk_count,
                    s.keywords = $keywords
            """, **section_data)
            
            # Link section to document
            tx.run("""
                MATCH (d:Document {url: $url})
                MATCH (s:Section {section_id: $section_id})
                MERGE (d)-[:HAS_SECTION]->(s)
            """, url=url, section_id=section_id)
    
    def _create_topic_nodes(self, tx, topics: Dict, url: str):
        """Create topic nodes"""
        for topic_id, topic_data in topics.items():
            tx.run("""
                MERGE (t:Topic {topic_id: $topic_id})
                SET t.name = $name,
                    t.keywords = $keywords,
                    t.chunk_count = $chunk_count
            """, topic_id=topic_id, name=topic_data['name'], 
                keywords=topic_data['keywords'], 
                chunk_count=len(topic_data['chunk_ids']))
            
            # Link topic to document
            tx.run("""
                MATCH (d:Document {url: $url})
                MATCH (t:Topic {topic_id: $topic_id})
                MERGE (d)-[:DISCUSSES]->(t)
            """, url=url, topic_id=topic_id)
    
    def _store_chunks_with_embeddings(self, tx, chunks: List[Dict], url: str):
        """Store chunks with their embeddings"""
        for chunk in chunks:
            # Generate embedding
            embedding = self.embedding_model.encode(chunk['content']).tolist()
            
            # Calculate additional metrics
            content_quality = self._calculate_content_quality(chunk['content'])
            
            # Store chunk
            tx.run("""
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.content = $content,
                    c.header = $header,
                    c.level = $level,
                    c.token_count = $token_count,
                    c.chunk_index = $chunk_index,
                    c.total_chunks = $total_chunks,
                    c.hash = $hash,
                    c.embedding = $embedding,
                    c.content_quality = $content_quality,
                    c.word_count = $word_count,
                    c.created_at = datetime(),
                    c.updated_at = datetime()
            """, embedding=embedding, content_quality=content_quality,
                word_count=len(chunk['content'].split()), **chunk)
            
            # Link chunk to document
            tx.run("""
                MATCH (d:Document {url: $url})
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (d)-[:CONTAINS]->(c)
            """, url=url, chunk_id=chunk['chunk_id'])
    
    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score"""
        score = 0.0
        
        # Length score (optimal around 300-800 chars)
        length = len(content)
        if 300 <= length <= 800:
            score += 0.3
        elif 100 <= length <= 1200:
            score += 0.2
        
        # Sentence structure
        sentences = content.split('.')
        if 2 <= len(sentences) <= 8:
            score += 0.2
        
        # Word variety
        words = content.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            variety_ratio = len(unique_words) / len(words)
            score += min(variety_ratio * 0.5, 0.3)
        
        # Has numbers/facts
        if re.search(r'\d+', content):
            score += 0.1
        
        # Has proper nouns
        if re.search(r'[A-Z][a-zA-Z]+', content):
            score += 0.1
        
        return min(score, 1.0)
    
    def _create_intelligent_relationships(self, tx, chunks: List[Dict], sections: Dict, topics: Dict, url: str):
        """Create intelligent relationships between entities"""
        
        # 1. Sequential relationships (only immediate neighbors)
        for i, chunk in enumerate(chunks):
            current_id = chunk['chunk_id']
            
            # Next chunk (if exists and in same section)
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                if chunk.get('header') == next_chunk.get('header'):
                    tx.run("""
                        MATCH (c1:Chunk {chunk_id: $current_id})
                        MATCH (c2:Chunk {chunk_id: $next_id})
                        MERGE (c1)-[:FOLLOWS]->(c2)
                    """, current_id=current_id, next_id=next_chunk['chunk_id'])
        
        # 2. Section relationships
        section_mapping = {}
        for section_id, section_data in sections.items():
            section_mapping[section_data['title']] = section_id
        
        for chunk in chunks:
            header = chunk.get('header', 'Unknown')
            if header in section_mapping:
                section_id = section_mapping[header]
                tx.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MATCH (s:Section {section_id: $section_id})
                    MERGE (s)-[:CONTAINS]->(c)
                """, chunk_id=chunk['chunk_id'], section_id=section_id)
        
        # 3. Topic relationships
        for topic_id, topic_data in topics.items():
            for chunk_id in topic_data['chunk_ids']:
                tx.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MATCH (t:Topic {topic_id: $topic_id})
                    MERGE (c)-[:RELATES_TO]->(t)
                """, chunk_id=chunk_id, topic_id=topic_id)
        
        # 4. Hierarchical section relationships
        section_levels = defaultdict(list)
        for section_data in sections.values():
            section_levels[section_data['level']].append(section_data['section_id'])
        
        # Parent-child relationships between sections
        for level in sorted(section_levels.keys()):
            if level + 1 in section_levels:
                for parent_id in section_levels[level]:
                    for child_id in section_levels[level + 1]:
                        # Check if they're related by content
                        parent_section = sections[parent_id]
                        child_section = sections[child_id]
                        
                        # Simple heuristic: if child title contains parent keywords
                        if any(keyword in child_section['title'].lower() 
                               for keyword in parent_section['keywords'][:3]):
                            tx.run("""
                                MATCH (p:Section {section_id: $parent_id})
                                MATCH (c:Section {section_id: $child_id})
                                MERGE (p)-[:HAS_SUBSECTION]->(c)
                            """, parent_id=parent_id, child_id=child_id)
    
    def _create_semantic_clusters(self, tx, chunks: List[Dict], url: str):
        """Create semantic similarity clusters"""
        if len(chunks) < 2:
            return
            
        # Get embeddings for all chunks
        embeddings = []
        chunk_ids = []
        
        for chunk in chunks:
            embedding = self.embedding_model.encode(chunk['content'])
            embeddings.append(embedding)
            chunk_ids.append(chunk['chunk_id'])
        
        embeddings = np.array(embeddings)
        
        # Calculate similarity matrix
        similarities = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        similarities = similarities / (norms[:, None] * norms[None, :])
        
        # Create relationships for highly similar chunks
        similarity_threshold = 0.7
        
        for i in range(len(chunk_ids)):
            for j in range(i + 1, len(chunk_ids)):
                similarity = similarities[i, j]
                if similarity > similarity_threshold:
                    tx.run("""
                        MATCH (c1:Chunk {chunk_id: $chunk_id1})
                        MATCH (c2:Chunk {chunk_id: $chunk_id2})
                        MERGE (c1)-[:SIMILAR_TO {similarity: $similarity}]->(c2)
                    """, chunk_id1=chunk_ids[i], chunk_id2=chunk_ids[j], 
                         similarity=float(similarity))
    
    def semantic_search(self, query: str, url: Optional[str] = None, top_k: int = 7) -> List[Dict]:
        """
        Enhanced semantic search with intelligent ranking
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            query_embedding_np = np.array(query_embedding)
            
            # Extract query topics for better matching
            query_topics = self._extract_topics(query)
            query_keywords = self._extract_keywords(query)

            with self.driver.session() as session:
                # Get chunks with their context
                cypher_query = """
                    MATCH (c:Chunk)
                """
                parameters = {}
                
                if url:
                    cypher_query += """
                    MATCH (d:Document {url: $url})-[:CONTAINS]->(c)
                    """
                    parameters['url'] = url

                cypher_query += """
                    OPTIONAL MATCH (s:Section)-[:CONTAINS]->(c)
                    OPTIONAL MATCH (c)-[:RELATES_TO]->(t:Topic)
                    RETURN c.chunk_id as chunk_id,
                           c.content as content,
                           c.header as header,
                           c.level as level,
                           c.chunk_index as chunk_index,
                           c.token_count as token_count,
                           c.content_quality as content_quality,
                           c.embedding as embedding,
                           s.title as section_title,
                           collect(DISTINCT t.name) as topics
                    ORDER BY c.chunk_index
                """

                result = session.run(cypher_query, parameters)
                chunks = []
                
                for record in result:
                    embedding = record.get("embedding")
                    if embedding is not None:
                        chunk_data = {
                            "chunk_id": record["chunk_id"],
                            "content": record["content"],
                            "header": record["header"],
                            "level": record["level"],
                            "chunk_index": record["chunk_index"],
                            "token_count": record["token_count"],
                            "content_quality": record.get("content_quality", 0.5),
                            "section_title": record.get("section_title"),
                            "topics": record.get("topics", []),
                            "embedding": np.array(embedding)
                        }
                        chunks.append(chunk_data)

                # Enhanced similarity calculation
                for chunk in chunks:
                    # Base semantic similarity
                    emb = chunk["embedding"]
                    if emb is not None and len(emb) == len(query_embedding_np):
                        semantic_sim = np.dot(query_embedding_np, emb) / (
                            np.linalg.norm(query_embedding_np) * np.linalg.norm(emb) + 1e-8
                        )
                    else:
                        semantic_sim = 0.0
                    
                    # Topic relevance boost
                    topic_boost = 0.0
                    chunk_topics = chunk.get("topics", [])
                    if chunk_topics and query_topics:
                        topic_matches = sum(1 for topic in query_topics 
                                          if any(topic.lower() in chunk_topic.lower() 
                                               for chunk_topic in chunk_topics))
                        topic_boost = min(topic_matches * 0.1, 0.3)
                    
                    # Keyword relevance boost
                    keyword_boost = 0.0
                    content_lower = chunk["content"].lower()
                    keyword_matches = sum(1 for keyword in query_keywords 
                                        if keyword in content_lower)
                    keyword_boost = min(keyword_matches * 0.05, 0.2)
                    
                    # Content quality boost
                    quality_boost = chunk.get("content_quality", 0.5) * 0.1
                    
                    # Header relevance boost
                    header_boost = 0.0
                    if chunk.get("header") and any(keyword in chunk["header"].lower() 
                                                 for keyword in query_keywords):
                        header_boost = 0.15
                    
                    # Calculate final score
                    final_score = (
                        semantic_sim * 0.6 +  # 60% semantic similarity
                        topic_boost +          # Topic relevance
                        keyword_boost +        # Keyword relevance
                        quality_boost +        # Content quality
                        header_boost           # Header relevance
                    )
                    
                    chunk["similarity"] = float(final_score)

                # Filter and sort
                filtered_chunks = [c for c in chunks if c["similarity"] > 0.2]
                filtered_chunks.sort(key=lambda x: x["similarity"], reverse=True)
                top_chunks = filtered_chunks[:top_k]

                # Add context chunks for top results
                for chunk in top_chunks:
                    chunk["context_chunks"] = self._get_context_chunks(session, chunk["chunk_id"])
                    # Remove embedding from output
                    del chunk["embedding"]

                # If not enough results, add keyword fallback
                if len(top_chunks) < top_k:
                    keyword_chunks = self._enhanced_keyword_search(session, query, url, top_k - len(top_chunks))
                    existing_ids = {chunk['chunk_id'] for chunk in top_chunks}
                    for chunk in keyword_chunks:
                        if chunk['chunk_id'] not in existing_ids:
                            top_chunks.append(chunk)

                return top_chunks[:top_k]

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return self._fallback_search(query, url, top_k)
    
    def _get_context_chunks(self, session, chunk_id: str) -> List[str]:
        """Get contextually relevant chunks"""
        result = session.run("""
            MATCH (c:Chunk {chunk_id: $chunk_id})
            
            // Get sequential context
            OPTIONAL MATCH (c)<-[:FOLLOWS]-(prev:Chunk)
            OPTIONAL MATCH (c)-[:FOLLOWS]->(next:Chunk)
            
            // Get similar chunks
            OPTIONAL MATCH (c)-[:SIMILAR_TO]-(similar:Chunk)
            WHERE similar.chunk_id <> $chunk_id
            
            // Get section siblings
            OPTIONAL MATCH (s:Section)-[:CONTAINS]->(c)
            OPTIONAL MATCH (s)-[:CONTAINS]->(sibling:Chunk)
            WHERE sibling.chunk_id <> $chunk_id
            
            WITH c, prev, next, 
                 collect(DISTINCT similar.chunk_id)[..2] as similar_chunks,
                 collect(DISTINCT sibling.chunk_id)[..2] as sibling_chunks
            
            RETURN coalesce(prev.chunk_id, '') as prev_id,
                   coalesce(next.chunk_id, '') as next_id,
                   similar_chunks,
                   sibling_chunks
        """, chunk_id=chunk_id)
        
        record = result.single()
        context_ids = []
        
        if record:
            if record['prev_id']:
                context_ids.append(record['prev_id'])
            if record['next_id']:
                context_ids.append(record['next_id'])
            context_ids.extend(record.get('similar_chunks', []))
            context_ids.extend(record.get('sibling_chunks', []))
        
        return [cid for cid in context_ids if cid][:5]  # Max 5 context chunks
    
    def _enhanced_keyword_search(self, session, query: str, url: Optional[str], limit: int) -> List[Dict]:
        """Enhanced keyword search with better ranking"""
        query_words = self._extract_keywords(query)
        
        if not query_words:
            return []
        
        cypher_query = """
            MATCH (c:Chunk)
        """
        
        parameters = {'limit': limit}
        
        if url:
            cypher_query += """
            MATCH (d:Document {url: $url})-[:CONTAINS]->(c)
            """
            parameters['url'] = url
        
        # Build scoring logic
        score_parts = []
        for i, word in enumerate(query_words[:3]):  # Max 3 words for performance
            param_name = f'word_{i}'
            parameters[param_name] = word.lower()
            score_parts.append(f"""
                CASE WHEN toLower(c.content) CONTAINS ${param_name} THEN 1 ELSE 0 END +
                CASE WHEN toLower(c.header) CONTAINS ${param_name} THEN 2 ELSE 0 END
            """)
        
        if score_parts:
            scoring_logic = " + ".join(score_parts)
            cypher_query += f"""
            WITH c, ({scoring_logic}) as keyword_score
            WHERE keyword_score > 0
            RETURN c.chunk_id as chunk_id,
                   c.content as content,
                   c.header as header,
                   c.level as level,
                   c.chunk_index as chunk_index,
                   c.token_count as token_count,
                   (keyword_score * 0.05 + coalesce(c.content_quality, 0.5) * 0.1) as similarity
            ORDER BY keyword_score DESC, c.content_quality DESC
            LIMIT $limit
            """
            
            result = session.run(cypher_query, parameters)
            return [dict(record) for record in result]
        
        return []
    
    def _fallback_search(self, query: str, url: Optional[str], top_k: int) -> List[Dict]:
        """Improved fallback search"""
        try:
            with self.driver.session() as session:
                cypher_query = """
                    MATCH (c:Chunk)
                """
                
                parameters = {'top_k': top_k, 'query': query.lower()}
                
                if url:
                    cypher_query += """
                    MATCH (d:Document {url: $url})-[:CONTAINS]->(c)
                    """
                    parameters['url'] = url
                
                cypher_query += """
                    WHERE toLower(c.content) CONTAINS $query
                       OR toLower(c.header) CONTAINS $query
                    RETURN c.chunk_id as chunk_id,
                           c.content as content,
                           c.header as header,
                           c.level as level,
                           c.chunk_index as chunk_index,
                           c.token_count as token_count,
                           0.3 as similarity
                    ORDER BY c.content_quality DESC, c.chunk_index ASC
                    LIMIT $top_k
                """
                
                result = session.run(cypher_query, parameters)
                return [dict(record) for record in result]
                
        except Exception as e:
            print(f"Error in fallback search: {e}")
            return []
    
    def get_document_info(self, url: str) -> Optional[Dict]:
        """Get document information with enhanced stats"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document {url: $url})
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                    OPTIONAL MATCH (d)-[:DISCUSSES]->(t:Topic)
                    RETURN d.title as title,
                           d.total_chunks as total_chunks,
                           d.created_at as created_at,
                           d.updated_at as updated_at,
                           count(DISTINCT s) as section_count,
                           count(DISTINCT t) as topic_count
                """, url=url)
                
                record = result.single()
                if record:
                    return dict(record)
                return None
                
        except Exception as e:
            print(f"Error getting document info: {e}")
            return None
    
    def delete_document(self, url: str) -> bool:
        """Delete document and all related entities"""
        try:
            with self.driver.session() as session:
                session.run("""
                    MATCH (d:Document {url: $url})
                    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                    OPTIONAL MATCH (d)-[:DISCUSSES]->(t:Topic)
                    DETACH DELETE d, c, s, t
                """, url=url)
                return True
                
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_all_documents(self) -> List[Dict]:
        """Get all stored documents with enhanced info"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                    OPTIONAL MATCH (d)-[:DISCUSSES]->(t:Topic)
                    RETURN d.url as url,
                           d.title as title,
                           d.total_chunks as total_chunks,
                           d.created_at as created_at,
                           count(DISTINCT s) as section_count,
                           count(DISTINCT t) as topic_count
                    ORDER BY d.created_at DESC
                """)
                
                return [dict(record) for record in result]
                
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_document_structure(self, url: str) -> Dict:
        """Get document structure for analysis"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document {url: $url})
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                    OPTIONAL MATCH (d)-[:DISCUSSES]->(t:Topic)
                    OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk)
                    RETURN d.title as document_title,
                           collect(DISTINCT {
                               section_id: s.section_id,
                               title: s.title,
                               level: s.level,
                               chunk_count: s.chunk_count
                           }) as sections,
                           collect(DISTINCT {
                               topic_id: t.topic_id,
                               name: t.name,
                               chunk_count: t.chunk_count
                           }) as topics,
                           count(DISTINCT c) as total_chunks
                """, url=url)
                
                record = result.single()
                if record:
                    return {
                        'document_title': record['document_title'],
                        'sections': [s for s in record['sections'] if s['section_id']],
                        'topics': [t for t in record['topics'] if t['topic_id']],
                        'total_chunks': record['total_chunks']
                    }
                return {}
                
        except Exception as e:
            print(f"Error getting document structure: {e}")
            return {}
    
    def get_chunk_relationships(self, chunk_id: str) -> Dict:
        """Get all relationships for a specific chunk"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    
                    // Sequential relationships
                    OPTIONAL MATCH (c)-[:FOLLOWS]->(next:Chunk)
                    OPTIONAL MATCH (prev:Chunk)-[:FOLLOWS]->(c)
                    
                    // Section relationship
                    OPTIONAL MATCH (s:Section)-[:CONTAINS]->(c)
                    
                    // Topic relationships
                    OPTIONAL MATCH (c)-[:RELATES_TO]->(t:Topic)
                    
                    // Similar chunks
                    OPTIONAL MATCH (c)-[:SIMILAR_TO]-(similar:Chunk)
                    
                    RETURN c.content as content,
                           c.header as header,
                           next.chunk_id as next_chunk,
                           prev.chunk_id as prev_chunk,
                           s.title as section_title,
                           collect(DISTINCT t.name) as topics,
                           collect(DISTINCT similar.chunk_id) as similar_chunks
                """, chunk_id=chunk_id)
                
                record = result.single()
                if record:
                    return dict(record)
                return {}
                
        except Exception as e:
            print(f"Error getting chunk relationships: {e}")
            return {}
    
    def optimize_relationships(self, url: str) -> bool:
        """Optimize relationships for better retrieval performance"""
        try:
            with self.driver.session() as session:
                # Remove weak similarity relationships
                session.run("""
                    MATCH (d:Document {url: $url})-[:CONTAINS]->(c1:Chunk)
                    MATCH (c1)-[r:SIMILAR_TO]-(c2:Chunk)
                    WHERE r.similarity < 0.6
                    DELETE r
                """, url=url)
                
                # Create importance scores based on relationships
                session.run("""
                    MATCH (d:Document {url: $url})-[:CONTAINS]->(c:Chunk)
                    OPTIONAL MATCH (c)-[:RELATES_TO]->(t:Topic)
                    OPTIONAL MATCH (s:Section)-[:CONTAINS]->(c)
                    OPTIONAL MATCH (c)-[:SIMILAR_TO]-(similar:Chunk)
                    
                    WITH c, 
                         count(DISTINCT t) as topic_count,
                         count(DISTINCT similar) as similarity_count,
                         coalesce(c.content_quality, 0.5) as quality
                    
                    SET c.importance_score = (
                        topic_count * 0.3 + 
                        similarity_count * 0.2 + 
                        quality * 0.5
                    )
                """, url=url)
                
                return True
                
        except Exception as e:
            print(f"Error optimizing relationships: {e}")
            return False
    
    def get_search_analytics(self, query: str, results: List[Dict]) -> Dict:
        """Get analytics for search results to improve future searches"""
        try:
            analytics = {
                'query': query,
                'result_count': len(results),
                'avg_similarity': 0.0,
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'section_coverage': set(),
                'topic_coverage': set()
            }
            
            if not results:
                return analytics
            
            similarities = [r.get('similarity', 0) for r in results]
            analytics['avg_similarity'] = sum(similarities) / len(similarities)
            
            for result in results:
                # Quality distribution
                quality = result.get('content_quality', 0.5)
                if quality > 0.7:
                    analytics['quality_distribution']['high'] += 1
                elif quality > 0.4:
                    analytics['quality_distribution']['medium'] += 1
                else:
                    analytics['quality_distribution']['low'] += 1
                
                # Coverage
                if result.get('section_title'):
                    analytics['section_coverage'].add(result['section_title'])
                
                topics = result.get('topics', [])
                analytics['topic_coverage'].update(topics)
            
            # Convert sets to lists for JSON serialization
            analytics['section_coverage'] = list(analytics['section_coverage'])
            analytics['topic_coverage'] = list(analytics['topic_coverage'])
            
            return analytics
            
        except Exception as e:
            print(f"Error getting search analytics: {e}")
            return {'error': str(e)}