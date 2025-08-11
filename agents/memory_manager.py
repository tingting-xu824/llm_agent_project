import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional, Any, Callable
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustConnectionPool:
    """Enhanced connection pool with recovery mechanisms"""
    
    def __init__(self, database_url: str, minconn: int = 5, maxconn: int = 20, 
                 connect_timeout: int = 10, retry_attempts: int = 3, 
                 retry_delay: float = 1.0, health_check_interval: int = 300):
        """
        Initialize robust connection pool
        
        Args:
            database_url: PostgreSQL connection string
            minconn: Minimum connections in pool
            maxconn: Maximum connections in pool
            connect_timeout: Connection timeout in seconds
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retries in seconds
            health_check_interval: Health check interval in seconds
        """
        self.database_url = database_url
        self.minconn = minconn
        self.maxconn = maxconn
        self.connect_timeout = connect_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        self.last_health_check = 0
        self.pool = None
        self.is_healthy = True
        self.connection_errors = 0
        self.max_connection_errors = 10
        
        # Initialize pool
        self._initialize_pool()
        
        # Start health check task
        self._start_health_check()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.minconn,
                maxconn=self.maxconn,
                dsn=self.database_url,
                connect_timeout=self.connect_timeout,
                application_name="hackathon_memory_system"
            )
            self.is_healthy = True
            self.connection_errors = 0
            logger.info(f"Database connection pool initialized with {self.minconn}-{self.maxconn} connections")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            self.is_healthy = False
            raise
    
    def _start_health_check(self):
        """Start periodic health check"""
        async def health_check_loop():
            while True:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self._perform_health_check()
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
        
        # Start health check in background
        asyncio.create_task(health_check_loop())
    
    async def _perform_health_check(self):
        """Perform health check on the connection pool"""
        if not self.pool:
            return
        
        try:
            conn = self.pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            self.pool.putconn(conn)
            
            self.is_healthy = True
            self.connection_errors = 0
            self.last_health_check = time.time()
            logger.debug("Database connection pool health check passed")
            
        except Exception as e:
            logger.warning(f"Database connection pool health check failed: {e}")
            self.connection_errors += 1
            self.is_healthy = False
            
            # Attempt pool recovery if too many errors
            if self.connection_errors >= self.max_connection_errors:
                await self._recover_pool()
    
    async def _recover_pool(self):
        """Attempt to recover the connection pool"""
        logger.warning("Attempting to recover database connection pool...")
        
        try:
            # Close existing pool
            if self.pool:
                self.pool.closeall()
            
            # Wait before reinitializing
            await asyncio.sleep(self.retry_delay * 2)
            
            # Reinitialize pool
            self._initialize_pool()
            logger.info("Database connection pool recovered successfully")
            
        except Exception as e:
            logger.error(f"Failed to recover database connection pool: {e}")
            self.is_healthy = False
    
    def get_connection(self):
        """Get a connection from the pool with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                if not self.is_healthy:
                    logger.warning("Connection pool is unhealthy, attempting recovery...")
                    # Note: Recovery is handled asynchronously in health check
                    # For sync context, we'll just try to get a connection
                
                conn = self.pool.getconn()
                return conn
                
            except psycopg2.pool.PoolError as e:
                logger.warning(f"Connection pool exhausted (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error("All connection pool retry attempts failed")
                    raise Exception("Database temporarily unavailable - connection pool exhausted")
            
            except Exception as e:
                logger.error(f"Unexpected error getting connection: {e}")
                raise
    
    def return_connection(self, conn):
        """Return a connection to the pool safely"""
        try:
            if conn and self.pool:
                self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
            # Try to close the connection if we can't return it to the pool
            try:
                conn.close()
            except:
                pass
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status"""
        if not self.pool:
            return {"status": "not_initialized"}
        
        try:
            return {
                "status": "healthy" if self.is_healthy else "unhealthy",
                "min_connections": self.minconn,
                "max_connections": self.maxconn,
                "connection_errors": self.connection_errors,
                "last_health_check": self.last_health_check,
                "pool_size": self.pool.get_size() if hasattr(self.pool, 'get_size') else "unknown"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def close(self):
        """Close the connection pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")

class MemoryManager:
    """Manages vector memory storage and retrieval for the hackathon system"""
    
    def __init__(self, database_url: str, embedding_generator: Callable, memory_extractor: Callable, pool_size: int = 20):
        """
        Initialize MemoryManager with external dependencies
        
        Args:
            database_url: PostgreSQL connection string
            embedding_generator: Function that takes text and returns embedding vector
            memory_extractor: Function that takes conversation text and memory type, returns summary
            pool_size: Maximum database connections in pool
        """
        self.database_url = database_url
        self.embedding_generator = embedding_generator
        self.memory_extractor = memory_extractor
        self.embedding_cache = {}  # Simple in-memory cache for embeddings
        self.cache_size_limit = 1000
        
        # Initialize robust database connection pool
        self.connection_pool = RobustConnectionPool(
            database_url=database_url,
            minconn=5,
            maxconn=pool_size,
            connect_timeout=10,
            retry_attempts=3,
            retry_delay=1.0,
            health_check_interval=300
        )
        
        # Memory trigger thresholds
        self.chat_message_threshold = 10
        self.inactivity_threshold = 1800  # 30 minutes in seconds
        self.important_keywords = [
            'important', 'decision', 'problem', 'solution', 'change', 'modify',
            'key', 'critical', 'issue', 'update', 'final', 'conclusion'
        ]
    
    @contextmanager
    def get_db_connection(self):
        """Safe database connection context manager with recovery"""
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.return_connection(conn)
    
    async def get_embedding_with_cache(self, text: str) -> List[float]:
        """Get embedding vector for text with caching"""
        # Create hash for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            logger.debug("Using cached embedding")
            return self.embedding_cache[text_hash]
        
        try:
            # Call external embedding generator
            embedding = await self.embedding_generator(text)
            
            # Cache the result with size limit
            if len(self.embedding_cache) < self.cache_size_limit:
                self.embedding_cache[text_hash] = embedding
            
            logger.debug(f"Generated and cached embedding for text length: {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def vector_to_sql(self, vector: List[float]) -> str:
        """Convert vector to PostgreSQL format"""
        return f"[{','.join(map(str, vector))}]"
    
    def get_conversations_by_user(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Retrieve recent conversations for a user"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT conversation_id, message_type, content, timestamp, sequence_number
                    FROM conversation 
                    WHERE user_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (user_id, limit))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {e}")
            return []
    
    def get_conversations_by_ids(self, conversation_ids: List[int]) -> List[Dict]:
        """Get specific conversations by their IDs"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT conversation_id, message_type, content, timestamp, sequence_number
                    FROM conversation 
                    WHERE conversation_id = ANY(%s)
                    ORDER BY timestamp ASC
                """, (conversation_ids,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get conversations by IDs: {e}")
            return []
    
    def format_conversations_for_memory(self, conversations: List[Dict]) -> str:
        """Format conversations into text for memory extraction"""
        formatted = []
        for conv in reversed(conversations):  # Reverse to chronological order
            role = "User" if conv['message_type'] == 'user_input' else "Agent"
            formatted.append(f"{role}: {conv['content']}")
        return "\n".join(formatted)
    
    async def extract_memory_content(self, conversation_text: str, memory_type: str) -> str:
        """Extract key memory points from conversations using external memory extractor"""
        try:
            # Call external memory extraction function
            memory_content = await self.memory_extractor(conversation_text, memory_type)
            logger.debug(f"Extracted memory content for type: {memory_type}")
            return memory_content
            
        except Exception as e:
            logger.error(f"Failed to extract memory: {e}")
            return f"Memory extraction failed for conversation at {datetime.now()}"
    
    def store_memory(self, user_id: int, memory_type: str, source_conversations: str,
                          memory_content: str, embedding: List[float], metadata: Optional[Dict] = None) -> int:
        """Store memory vector in database"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memory_vectors 
                    (user_id, memory_type, source_conversations, memory_content, embedding, _metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING memory_id
                """, (
                    user_id,
                    memory_type,
                    source_conversations,
                    memory_content,
                    self.vector_to_sql(embedding),
                    json.dumps(metadata) if metadata else None
                ))
                memory_id = cursor.fetchone()[0]
                logger.info(f"Stored memory {memory_id} for user {user_id}")
                return memory_id
                
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def retrieve_relevant_memories(self, user_id: int, query_text: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant memories for a query using vector similarity"""
        try:
            query_embedding = await self.get_embedding_with_cache(query_text)
            
            # Use asyncio to run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._retrieve_memories_sync, user_id, query_embedding, top_k)
            logger.debug(f"Retrieved {len(results)} relevant memories for user {user_id}")
            return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def _retrieve_memories_sync(self, user_id: int, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Synchronous version of retrieve_relevant_memories for thread pool execution"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT memory_id, memory_content, memory_type, _metadata, created_at,
                           1 - (embedding <=> %s::VECTOR(1536)) as similarity
                    FROM memory_vectors 
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::VECTOR(1536)
                    LIMIT %s
                """, (
                    self.vector_to_sql(query_embedding),
                    user_id,
                    self.vector_to_sql(query_embedding),
                    top_k
                ))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to retrieve memories (sync): {e}")
            return []
    
    def detect_important_content(self, message: str) -> bool:
        """Detect if message contains important content that should trigger memory creation"""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.important_keywords)
    
    async def check_memory_trigger(self, user_id: int, mode: str, **kwargs) -> Tuple[bool, List[str]]:
        """Check if memory should be created based on mode and conditions"""
        if mode == "eval":
            return await self._check_eval_trigger(user_id, **kwargs)
        elif mode == "chat":
            return await self._check_chat_trigger(user_id, **kwargs)
        return False, []
    
    async def _check_eval_trigger(self, user_id: int, round_completed: bool = False) -> Tuple[bool, List[str]]:
        """Check evaluation phase memory triggers"""
        if round_completed:
            return True, ["round_completed"]
        return False, []
    
    async def _check_chat_trigger(self, user_id: int, message_count: int = 0, 
                                 important_content: bool = False, session_closing: bool = False) -> Tuple[bool, List[str]]:
        """Check chat phase memory triggers"""
        triggers = []
        
        # Trigger 1: Message count threshold
        if message_count > 0 and message_count % self.chat_message_threshold == 0:
            triggers.append("message_count")
        
        # Trigger 2: Important content detected
        if important_content:
            triggers.append("important_content")
        
        # Trigger 3: Session closing
        if session_closing:
            triggers.append("session_closing")
        
        # Trigger 4: Inactivity check (could be implemented with additional tracking)
        # This would require storing last activity timestamps in database
        
        return len(triggers) > 0, triggers
    
    async def create_memory_from_conversations(self, user_id: int, memory_type: str, 
                                              conversation_ids: Optional[List[int]] = None,
                                              agent_type: Optional[int] = None) -> Optional[int]:
        """Create memory from recent conversations"""
        try:
            # Get conversations (now synchronous)
            if conversation_ids:
                conversations = self.get_conversations_by_ids(conversation_ids)
            else:
                conversations = self.get_conversations_by_user(user_id, limit=20)
            
            if not conversations:
                logger.warning(f"No conversations found for user {user_id}")
                return None
            
            # Format conversations for memory extraction
            conversation_text = self.format_conversations_for_memory(conversations)
            
            # Extract memory content using external function
            memory_content = await self.extract_memory_content(conversation_text, memory_type)
            
            # Generate embedding using external function
            embedding = await self.get_embedding_with_cache(memory_content)
            
            # Prepare metadata
            metadata = {
                "agent_type": agent_type,
                "conversation_count": len(conversations),
                "created_by": "system"
            }
            
            # Store memory (now synchronous, run in thread pool)
            loop = asyncio.get_event_loop()
            memory_id = await loop.run_in_executor(None, self.store_memory,
                user_id, memory_type, json.dumps([c['conversation_id'] for c in conversations]),
                memory_content, embedding, metadata
            )
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to create memory for user {user_id}: {e}")
            return None
    
    async def create_memory_async(self, user_id: int, mode: str, triggers: List[str], 
                                 agent_type: Optional[int] = None) -> None:
        """Asynchronously create memory without blocking API response"""
        try:
            memory_type = "round_summary" if mode == "eval" else "conversation_chunk"
            await self.create_memory_from_conversations(user_id, memory_type, agent_type=agent_type)
            logger.info(f"Memory created for user {user_id}, triggers: {triggers}")
        except Exception as e:
            logger.error(f"Background memory creation failed for user {user_id}: {e}")
    
    def get_user_memory_count(self, user_id: int) -> int:
        """Get total number of memories stored for a user"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memory_vectors WHERE user_id = %s", (user_id,))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get memory count for user {user_id}: {e}")
            return 0
    
    def get_recent_memories(self, user_id: int, limit: int = 5) -> List[Dict]:
        """Get recent memories for a user"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT memory_id, memory_content, memory_type, _metadata, created_at
                    FROM memory_vectors 
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, limit))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get recent memories for user {user_id}: {e}")
            return []
    
    async def build_memory_context(self, memories: List[Dict]) -> str:
        """Build context string from retrieved memories"""
        if not memories:
            return ""
        
        context_parts = []
        for i, memory in enumerate(memories, 1):
            memory_text = memory['memory_content']
            memory_type = memory['memory_type']
            context_parts.append(f"Memory {i} ({memory_type}): {memory_text}")
        
        return "\n\n".join(context_parts)
    
    def clear_embedding_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status for monitoring"""
        return self.connection_pool.get_pool_status()
    
    def close(self):
        """Close all database connections and cleanup"""
        self.connection_pool.close()
        self.clear_embedding_cache()