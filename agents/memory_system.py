import os
import asyncio
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from .memory_manager import MemoryManager
from .runner import ai_service_manager # Imports the global AI service manager instance

load_dotenv()

class MemorySystem:
    """Memory system wrapper providing asynchronous API interfaces for the hackathon system"""

    def __init__(self):
        """Initialize MemorySystem with optional database configuration"""
        self.database_url = os.getenv("DATABASE_URL", "")
        self.memory_manager = None

        # Initialize memory manager if database is available
        if self.database_url:
            try:
                self.memory_manager = MemoryManager(
                    database_url=self.database_url,
                    embedding_generator=self._generate_embedding,
                    memory_extractor=self._extract_memory_content,
                    pool_size=20
                )
                print("MemoryManager initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize MemoryManager: {e}")
                print("Memory system will operate in fallback mode")
                self.memory_manager = None
        else:
            print("Warning: No DATABASE_URL found. Memory system will operate in fallback mode")

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text using AI service manager"""
        try:
            return await ai_service_manager.generate_embedding(text)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 1536 # Fallback

    async def _extract_memory_content(self, conversation_text: str, memory_type: str) -> str:
        """Extract key memory content from conversation using AI service manager"""
        try:
            return await ai_service_manager.extract_memory_content(conversation_text, memory_type)
        except Exception as e:
            print(f"Error extracting memory content: {e}")
            return f"Memory extraction failed: {str(e)}"

    async def create_memory_context(self, user_id: int, user_message: str, top_k: int = 3) -> str:
        """Create memory context for user message (async)"""
        if not self.memory_manager:
            return ""
        
        try:
            memories = await self.memory_manager.retrieve_relevant_memories(user_id, user_message, top_k)
            if not memories:
                return ""
            
            context_parts = []
            for i, memory in enumerate(memories, 1):
                memory_type = memory.get('memory_type', 'unknown')
                content = memory.get('memory_content', '')
                similarity = memory.get('similarity', 0)
                context_parts.append(f"Memory {i} ({memory_type}, relevance: {similarity:.2f}): {content}")
            
            return "\n".join(context_parts)
        except Exception as e:
            print(f"Error creating memory context: {e}")
            return ""

    async def store_memory(self, user_id: int, content: str, metadata: Optional[Dict] = None) -> bool:
        """Store memory for user (async)"""
        if not self.memory_manager:
            return False
        
        try:
            embedding = await self._generate_embedding(content)
            # Run synchronous store_memory in thread pool
            loop = asyncio.get_event_loop()
            memory_id = await loop.run_in_executor(None, self.memory_manager.store_memory,
                user_id,
                metadata.get('type', 'general') if metadata else 'general',
                "",
                content,
                embedding,
                metadata or {}
            )
            return memory_id is not None
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    async def retrieve_relevant_memories(self, user_id: int, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories for query (async)"""
        if not self.memory_manager:
            return []
        
        try:
            return await self.memory_manager.retrieve_relevant_memories(user_id, query, top_k)
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

    async def check_memory_trigger(self, user_id: int, mode: str, **kwargs) -> tuple[bool, List[str]]:
        """Check if memory should be created based on conditions (async)"""
        if not self.memory_manager:
            return False, []
        
        try:
            return await self.memory_manager.check_memory_trigger(user_id, mode, **kwargs)
        except Exception as e:
            print(f"Error checking memory trigger: {e}")
            return False, []

    async def create_memory_async(self, user_id: int, mode: str, triggers: List[str], agent_type: Optional[int] = None) -> None:
        """Asynchronously create memory from conversations"""
        if not self.memory_manager:
            return
        
        try:
            await self.memory_manager.create_memory_async(user_id, mode, triggers, agent_type)
        except Exception as e:
            print(f"Error creating memory asynchronously: {e}")

    async def get_memory_count(self, user_id: int) -> int:
        """Get total number of memories stored for a user (async)"""
        if not self.memory_manager:
            return 0
        
        try:
            # Run synchronous get_user_memory_count in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.memory_manager.get_user_memory_count, user_id)
        except Exception as e:
            print(f"Error getting memory count: {e}")
            return 0

    async def get_recent_memories(self, user_id: int, limit: int = 5) -> List[Dict]:
        """Get user's recent memories (async)"""
        if not self.memory_manager:
            return []
        
        try:
            # Run synchronous get_recent_memories in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.memory_manager.get_recent_memories, user_id, limit)
        except Exception as e:
            print(f"Error getting recent memories: {e}")
            return []

    def is_available(self) -> bool:
        """Check if memory system is available (sync - no async operations)"""
        return self.memory_manager is not None

    def get_status(self) -> Dict[str, Any]:
        """Get memory system status (sync - no async operations)"""
        return {
            "available": self.is_available(),
            "database_url": self.database_url if self.database_url else "Not configured",
            "memory_manager_initialized": self.memory_manager is not None
        }

# Global instance for easy access
memory_system = MemorySystem()
