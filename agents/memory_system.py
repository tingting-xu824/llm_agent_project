"""
Memory System for RAG (Retrieval-Augmented Generation)
This module handles embedding generation, memory storage, and retrieval
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional
from agents.database import save_memory_vector, get_relevant_memories

class SimpleEmbeddingModel:
    """Simple embedding model using TF-IDF-like approach"""
    
    def __init__(self):
        self.word_vectors = {}
        self.vocab_size = 1000
        print("Simple embedding model initialized")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for given text"""
        try:
            # Simple bag-of-words with basic preprocessing
            words = text.lower().split()
            embedding = [0.0] * self.vocab_size
            
            for word in words:
                # Simple hash-based word embedding
                word_hash = hash(word) % self.vocab_size
                embedding[word_hash] += 1.0
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * self.vocab_size

class MemorySystem:
    """Memory system for RAG functionality"""
    
    def __init__(self):
        # Initialize simple embedding model
        self.embedding_model = SimpleEmbeddingModel()
        print("Memory system initialized with simple embedding model")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        return self.embedding_model.generate_embedding(text)
    
    def store_memory(self, user_id: int, content: str, metadata: Dict = None) -> Optional[int]:
        """Store a memory with embedding"""
        try:
            # Generate embedding
            embedding = self.generate_embedding(content)
            if not embedding:
                return None
            
            # Save to database
            vector_id = save_memory_vector(user_id, content, embedding, metadata)
            return vector_id
        except Exception as e:
            print(f"Error storing memory: {e}")
            return None
    
    def retrieve_relevant_memories(self, user_id: int, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories for a query"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Get relevant memories
            memories = get_relevant_memories(user_id, query_embedding, top_k)
            return memories
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []
    
    def create_memory_context(self, user_id: int, query: str, top_k: int = 3) -> str:
        """Create context string from relevant memories"""
        try:
            memories = self.retrieve_relevant_memories(user_id, query, top_k)
            
            if not memories:
                return ""
            
            context_parts = []
            for i, memory in enumerate(memories, 1):
                content = memory['content']
                similarity = memory['similarity']
                timestamp = memory['timestamp']
                
                context_parts.append(
                    f"Memory {i} (relevance: {similarity:.2f}, from: {timestamp.strftime('%Y-%m-%d %H:%M')}):\n{content}"
                )
            
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"Error creating memory context: {e}")
            return ""

# Global memory system instance
memory_system = MemorySystem()
