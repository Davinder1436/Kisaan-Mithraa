import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re

# Simple in-memory vector database
# In production, you would use a proper vector database like Pinecone, Weaviate, or Chroma
class VectorDatabase:
    def __init__(self, dimension=384):  # Default dimension for sentence embeddings
        self.vectors = []
        self.metadata = []
        self.dimension = dimension
        
    def add_document(self, vector: List[float], metadata: Dict[str, Any]) -> int:
        """Add a document to the vector database"""
        # Validate vector dimension
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        
        # Add the document
        doc_id = len(self.vectors)
        self.vectors.append(vector)
        self.metadata.append(metadata)
        
        return doc_id
    
    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search the database for similar vectors"""
        if not self.vectors:
            return []
        
        # Convert to numpy for efficient computation
        query_vector_np = np.array(query_vector)
        vectors_np = np.array(self.vectors)
        
        # Compute cosine similarity
        dot_product = np.dot(vectors_np, query_vector_np)
        query_norm = np.linalg.norm(query_vector_np)
        doc_norm = np.linalg.norm(vectors_np, axis=1)
        
        # Handle zero norms to avoid division by zero
        query_norm = max(query_norm, 1e-10)
        doc_norm = np.maximum(doc_norm, 1e-10)
        
        similarities = dot_product / (query_norm * doc_norm)
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx]), self.metadata[idx]))
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save the vector database to a file"""
        data = {
            "dimension": self.dimension,
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """Load the vector database from a file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.dimension = data.get("dimension", self.dimension)
            self.vectors = data.get("vectors", [])
            self.metadata = data.get("metadata", [])

# Mock sentence transformer for embeddings
# In a real implementation, you would use a proper embedding model
class MockSentenceTransformer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384  # Default for the mentioned model
        
    def encode(self, text: str) -> List[float]:
        """Mock encoding function that creates deterministic embeddings based on text content"""
        # This is a simple hash-based mock - not for real use!
        # In a real implementation, you would use a proper embedding model
        hash_val = hash(text) % 10000
        np.random.seed(hash_val)
        return list(np.random.random(self.dimension).astype(float))

# RAG Service for context enhancement
class RAGService:
    def __init__(self, app_context):
        self.app_context = app_context
        self.vector_db = VectorDatabase()
        self.encoder = MockSentenceTransformer()
        
        # Initialize the vector database with app context
        self._initialize_vector_db()
        
    def _initialize_vector_db(self):
        """Initialize the vector database with app context data"""
        # Process navigation contexts
        for nav in self.app_context.get("navigation_contexts", []):
            # Create a document for each route
            text = f"{nav.get('title', '')}. {nav.get('description', '')}."
            
            # Add sample questions if available
            sample_questions = nav.get("sample_questions", [])
            if sample_questions:
                text += " Sample questions: " + " ".join(sample_questions)
            
            # Add user needs if available
            user_needs = nav.get("user_needs", [])
            if user_needs:
                text += " User needs: " + " ".join(user_needs)
            
            # Create vector embedding
            vector = self.encoder.encode(text)
            
            # Add to vector database
            self.vector_db.add_document(vector, {
                "type": "navigation",
                "route": nav.get("route"),
                "title": nav.get("title"),
                "description": nav.get("description")
            })
        
        # Add more context from meta information
        if "meta" in self.app_context:
            meta = self.app_context["meta"]
            app_description = f"{meta.get('app_name', '')} - {meta.get('app_description', '')}"
            vector = self.encoder.encode(app_description)
            self.vector_db.add_document(vector, {
                "type": "meta",
                "content": app_description
            })
            
            # Add key features
            for feature in meta.get("key_features", []):
                vector = self.encoder.encode(feature)
                self.vector_db.add_document(vector, {
                    "type": "feature",
                    "content": feature
                })
    
    def get_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant context based on the query"""
        # Encode the query
        query_vector = self.encoder.encode(query)
        
        # Search for similar documents
        results = self.vector_db.search(query_vector, top_k=top_k)
        
        # Extract and return metadata
        return [metadata for _, _, metadata in results]
    
    def enhance_query_with_context(self, query: str) -> str:
        """Enhance a query with relevant context"""
        relevant_context = self.get_relevant_context(query)
        
        # Extract navigation routes
        relevant_routes = []
        for ctx in relevant_context:
            if ctx.get("type") == "navigation" and "route" in ctx:
                relevant_routes.append({
                    "route": ctx["route"],
                    "title": ctx.get("title", ""),
                    "description": ctx.get("description", "")
                })
        
        # Create enhanced query
        if relevant_routes:
            routes_text = "\n".join([f"- {r['route']}: {r['title']} - {r['description']}" 
                                   for r in relevant_routes])
            enhanced_query = f"{query}\n\nRelevant app features:\n{routes_text}"
            return enhanced_query
        
        return query