import os
import asyncio
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from .memory_manager import MemoryManager
from .runner import ai_service_manager

load_dotenv()

class MemorySystem:
    """Memory system wrapper providing simplified API interfaces for the hackathon system"""
    
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
        """
        Generate embedding vector for text using AI service manager
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            return await ai_service_manager.generate_embedding(text)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a default embedding as fallback
            return [0.0] * 1536
    
    async def _extract_memory_content(self, conversation_text: str, memory_type: str) -> str:
        """
        Extract key memory content from conversation using AI service manager
        
        Args:
            conversation_text: Full conversation text
            memory_type: Type of memory to extract
            
        Returns:
            str: Extracted memory content
        """
        try:
            return await ai_service_manager.extract_memory_content(conversation_text, memory_type)
        except Exception as e:
            print(f"Error extracting memory content: {e}")
            return f"Memory extraction failed: {str(e)}"
    
    def create_memory_context(self, user_id: int, user_message: str, top_k: int = 3) -> str:
        """
        Create memory context for user message (synchronous wrapper)
        
        Args:
            user_id: User ID
            user_message: Current user message
            top_k: Number of relevant memories to retrieve
            
        Returns:
            str: Memory context string
        """
        if not self.memory_manager:
            return ""
        
        try:
            # Run async function in sync context
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._create_memory_context_async(user_id, user_message, top_k)
            )
        except RuntimeError:
            # If no event loop, create one
            return asyncio.run(
                self._create_memory_context_async(user_id, user_message, top_k)
            )
    
    async def _create_memory_context_async(self, user_id: int, user_message: str, top_k: int = 3) -> str:
        """
        Create memory context for user message (async implementation)
        
        Args:
            user_id: User ID
            user_message: Current user message
            top_k: Number of relevant memories to retrieve
            
        Returns:
            str: Memory context string
        """
        if not self.memory_manager:
            return ""
        
        try:
            # Retrieve relevant memories
            memories = await self.memory_manager.retrieve_relevant_memories(
                user_id, user_message, top_k
            )
            
            if not memories:
                return ""
            
            # Build context string
            context_parts = []
            for i, memory in enumerate(memories, 1):
                memory_type = memory.get('memory_type', 'unknown')
                content = memory.get('memory_content', '')
                similarity = memory.get('similarity', 0)
                
                context_parts.append(
                    f"Memory {i} ({memory_type}, relevance: {similarity:.2f}): {content}"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Error creating memory context: {e}")
            return ""
    
    def store_memory(self, user_id: int, content: str, metadata: Dict = None) -> bool:
        """
        Store memory for user (synchronous wrapper)
        
        Args:
            user_id: User ID
            content: Memory content
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        if not self.memory_manager:
            return False
        
        try:
            # Run async function in sync context
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._store_memory_async(user_id, content, metadata)
            )
        except RuntimeError:
            # If no event loop, create one
            return asyncio.run(
                self._store_memory_async(user_id, content, metadata)
            )
    
    async def _store_memory_async(self, user_id: int, content: str, metadata: Dict = None) -> bool:
        """
        Store memory for user (async implementation)
        
        Args:
            user_id: User ID
            content: Memory content
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        if not self.memory_manager:
            return False
        
        try:
            # Generate embedding
            embedding = await self._generate_embedding(content)
            
            # Store memory
            memory_id = await self.memory_manager.store_memory(
                user_id=user_id,
                memory_type=metadata.get('type', 'general') if metadata else 'general',
                source_conversations="",
                memory_content=content,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            return memory_id is not None
            
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False
    
    def retrieve_relevant_memories(self, user_id: int, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories for query (synchronous wrapper)
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of memories to retrieve
            
        Returns:
            List[Dict]: List of relevant memories
        """
        if not self.memory_manager:
            return []
        
        try:
            # Run async function in sync context
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.memory_manager.retrieve_relevant_memories(user_id, query, top_k)
            )
        except RuntimeError:
            # If no event loop, create one
            return asyncio.run(
                self.memory_manager.retrieve_relevant_memories(user_id, query, top_k)
            )
    
    async def check_memory_trigger(self, user_id: int, mode: str, **kwargs) -> tuple[bool, List[str]]:
        """
        Check if memory should be created based on conditions
        
        Args:
            user_id: User ID
            mode: Operation mode ("eval" or "chat")
            **kwargs: Additional trigger parameters
            
        Returns:
            tuple[bool, List[str]]: (should_create, triggers)
        """
        if not self.memory_manager:
            return False, []
        
        try:
            return await self.memory_manager.check_memory_trigger(user_id, mode, **kwargs)
        except Exception as e:
            print(f"Error checking memory trigger: {e}")
            return False, []
    
    async def create_memory_async(self, user_id: int, mode: str, triggers: List[str], agent_type: int = None) -> None:
        """
        Asynchronously create memory from conversations
        
        Args:
            user_id: User ID
            mode: Operation mode
            triggers: List of trigger reasons
            agent_type: Type of agent used
        """
        if not self.memory_manager:
            return
        
        try:
            await self.memory_manager.create_memory_async(user_id, mode, triggers, agent_type)
        except Exception as e:
            print(f"Error creating memory asynchronously: {e}")
    
    def get_memory_count(self, user_id: int) -> int:
        """
        Get total number of memories stored for a user (synchronous wrapper)
        
        Args:
            user_id: User ID
            
        Returns:
            int: Number of memories
        """
        if not self.memory_manager:
            return 0
        
        try:
            # Run async function in sync context
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.memory_manager.get_user_memory_count(user_id)
            )
        except RuntimeError:
            # If no event loop, create one
            return asyncio.run(
                self.memory_manager.get_user_memory_count(user_id)
            )
    
    def get_recent_memories(self, user_id: int, limit: int = 5) -> List[Dict]:
        """
        Get user's recent memories (synchronous wrapper)
        
        Args:
            user_id: User ID
            limit: Number of memories to retrieve
            
        Returns:
            List[Dict]: List of recent memories
        """
        if not self.memory_manager:
            return []
        
        try:
            # Run async function in sync context
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.memory_manager.get_recent_memories(user_id, limit)
            )
        except RuntimeError:
            # If no event loop, create one
            return asyncio.run(
                self.memory_manager.get_recent_memories(user_id, limit)
            )
    
    def is_available(self) -> bool:
        """
        Check if memory system is available
        
        Returns:
            bool: True if memory manager is initialized
        """
        return self.memory_manager is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get memory system status
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "available": self.is_available(),
            "database_url": self.database_url if self.database_url else "Not configured",
            "memory_manager_initialized": self.memory_manager is not None
        }

# Global instance for easy access
memory_system = MemorySystem()
