import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: Optional[str] = None
    
    # Groq API Configuration
    groq_api_key: Optional[str] = None
    groq_model: str = "llama3-8b-8192"
    
    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Chunking Configuration
    max_chunk_size: int = 500
    overlap_size: int = 50
    
    # Search Configuration
    default_top_k: int = 7
    similarity_threshold: float = 0.3
    
    # LLM Configuration
    max_tokens: int = 2048
    temperature: float = 0.3
    top_p: float = 0.9
    
    # Web Scraping Configuration
    request_timeout: int = 30
    max_content_length: int = 1000000  # 1MB
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", cls.neo4j_uri),
            neo4j_username=os.getenv("NEO4J_USERNAME", cls.neo4j_username),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", cls.groq_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", str(cls.max_chunk_size))),
            overlap_size=int(os.getenv("OVERLAP_SIZE", str(cls.overlap_size))),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", str(cls.default_top_k))),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", str(cls.similarity_threshold))),
            max_tokens=int(os.getenv("MAX_TOKENS", str(cls.max_tokens))),
            temperature=float(os.getenv("TEMPERATURE", str(cls.temperature))),
            top_p=float(os.getenv("TOP_P", str(cls.top_p))),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", str(cls.request_timeout))),
            max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", str(cls.max_content_length)))
        )
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.neo4j_password:
            print("❌ NEO4J_PASSWORD is required")
            return False
        
        if not self.groq_api_key:
            print("❌ GROQ_API_KEY is required")
            return False
        
        if self.max_chunk_size <= 0:
            print("❌ MAX_CHUNK_SIZE must be positive")
            return False
        
        if self.overlap_size >= self.max_chunk_size:
            print("❌ OVERLAP_SIZE must be less than MAX_CHUNK_SIZE")
            return False
        
        if not (0 <= self.temperature <= 2):
            print("❌ TEMPERATURE must be between 0 and 2")
            return False
        
        if not (0 <= self.top_p <= 1):
            print("❌ TOP_P must be between 0 and 1")
            return False
        
        return True