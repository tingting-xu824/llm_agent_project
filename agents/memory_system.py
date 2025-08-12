import os
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from .memory_manager import MemoryManager
from .runner import ai_service_manager # Imports the global AI service manager instance

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySystem:
    """Memory system wrapper providing asynchronous API interfaces for the hackathon system"""

    def __init__(self):
        """Initialize MemorySystem with optional database configuration"""
        self.database_url = os.getenv("DATABASE_URL", "")
        self.memory_manager = None
        self.fallback_mode = False
        self.last_error = None
        self.error_count = 0
        self.max_errors = 5
        self.error_threshold_time = datetime.now()
        
        # In-memory fallback storage for basic functionality
        self.fallback_memories = {}  # user_id -> List[Dict]
        self.fallback_contexts = {}  # user_id -> List[str]
        
        # Initialize memory manager if database is available
        if self.database_url:
            try:
                self.memory_manager = MemoryManager(
                    database_url=self.database_url,
                    embedding_generator=self._generate_embedding,
                    memory_extractor=self._extract_memory_content,
                    pool_size=20
                )
                logger.info("MemoryManager initialized successfully")
                self._notify_admin("Memory system initialized successfully", "INFO")
            except Exception as e:
                logger.error(f"Failed to initialize MemoryManager: {e}")
                self.last_error = str(e)
                self.fallback_mode = True
                self._notify_admin(f"Memory system operating in fallback mode: {e}", "WARNING")
        else:
            logger.warning("No DATABASE_URL found. Memory system will operate in fallback mode")
            self.fallback_mode = True
            self._notify_admin("Memory system operating in fallback mode: No DATABASE_URL", "WARNING")

    def _notify_admin(self, message: str, level: str = "INFO"):
        """Notify administrators about memory system status changes"""
        timestamp = datetime.now().isoformat()
        notification = f"[{timestamp}] Memory System {level}: {message}"
        
        # Log the notification
        if level == "ERROR":
            logger.error(notification)
        elif level == "WARNING":
            logger.warning(notification)
        else:
            logger.info(notification)
        
        # TODO: Add integration with monitoring systems (e.g., Sentry, DataDog, etc.)
        # TODO: Add email/SMS notifications for critical failures
        
        # For now, we'll just log to a dedicated file
        try:
            with open("memory_system_notifications.log", "a") as f:
                f.write(notification + "\n")
        except Exception as e:
            logger.error(f"Failed to write notification to file: {e}")

    def _handle_error(self, error: Exception, operation: str):
        """Handle errors and determine if we should switch to fallback mode"""
        self.error_count += 1
        self.last_error = str(error)
        
        # Check if we've exceeded error threshold
        if self.error_count >= self.max_errors:
            if not self.fallback_mode:
                self.fallback_mode = True
                self._notify_admin(f"Switched to fallback mode after {self.error_count} errors. Last error: {error}", "ERROR")
            logger.error(f"Memory system error in {operation}: {error}")
        else:
            logger.warning(f"Memory system warning in {operation}: {error}")

    def _reset_error_count(self):
        """Reset error count when operations succeed"""
        if self.error_count > 0:
            logger.info(f"Memory system recovered. Reset error count from {self.error_count} to 0")
            self.error_count = 0
            self.last_error = None

    def get_fallback_status(self) -> Dict[str, Any]:
        """Get detailed fallback mode status"""
        return {
            "fallback_mode": self.fallback_mode,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "fallback_memories_count": sum(len(memories) for memories in self.fallback_memories.values()),
            "fallback_contexts_count": sum(len(contexts) for contexts in self.fallback_contexts.values())
        }

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
        """Create memory context for user message (async) with fallback support"""
        if not self.memory_manager or self.fallback_mode:
            # Use fallback context from in-memory storage
            return self._get_fallback_context(user_id, user_message, top_k)
        
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
            
            self._reset_error_count()  # Reset error count on success
            return "\n".join(context_parts)
        except Exception as e:
            self._handle_error(e, "create_memory_context")
            # Fallback to in-memory context
            return self._get_fallback_context(user_id, user_message, top_k)
    
    def _get_fallback_context(self, user_id: int, user_message: str, top_k: int) -> str:
        """Get context from fallback in-memory storage"""
        if user_id not in self.fallback_contexts:
            return ""
        
        contexts = self.fallback_contexts[user_id][-top_k:]  # Get last top_k contexts
        if not contexts:
            return ""
        
        context_parts = []
        for i, context in enumerate(contexts, 1):
            context_parts.append(f"Recent Context {i}: {context}")
        
        return "\n".join(context_parts)

    async def store_memory(self, user_id: int, content: str, metadata: Optional[Dict] = None) -> bool:
        """Store memory for user (async) with fallback support"""
        if not self.memory_manager or self.fallback_mode:
            # Store in fallback in-memory storage
            return self._store_fallback_memory(user_id, content, metadata)
        
        try:
            embedding = await self._generate_embedding(content)
            # Run synchronous store_memory in thread pool
            loop = asyncio.get_event_loop()
            # Map metadata type to valid memory_type values
            metadata_type = metadata.get('type', 'general') if metadata else 'general'
            mode = metadata.get('mode', 'chat') if metadata else 'chat'
            
            # Use mode as memory_type since it's either 'eval' or 'chat'
            memory_type = mode
            
            memory_id = await loop.run_in_executor(None, self.memory_manager.store_memory,
                user_id,
                memory_type,
                "",
                content,
                embedding,
                metadata or {}
            )
            
            if memory_id is not None:
                self._reset_error_count()  # Reset error count on success
                return True
            else:
                self._handle_error(Exception("Failed to store memory - no memory_id returned"), "store_memory")
                return self._store_fallback_memory(user_id, content, metadata)
                
        except Exception as e:
            self._handle_error(e, "store_memory")
            # Fallback to in-memory storage
            return self._store_fallback_memory(user_id, content, metadata)
    
    def _store_fallback_memory(self, user_id: int, content: str, metadata: Optional[Dict] = None) -> bool:
        """Store memory in fallback in-memory storage"""
        try:
            if user_id not in self.fallback_memories:
                self.fallback_memories[user_id] = []
            
            # Map metadata type to valid memory_type values for fallback storage
            metadata_type = metadata.get('type', 'general') if metadata else 'general'
            mode = metadata.get('mode', 'chat') if metadata else 'chat'
            memory_type = mode
            
            memory_entry = {
                "memory_id": len(self.fallback_memories[user_id]) + 1,
                "memory_content": content,
                "memory_type": memory_type,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                "fallback": True
            }
            
            self.fallback_memories[user_id].append(memory_entry)
            
            # Also store as context for immediate use
            if user_id not in self.fallback_contexts:
                self.fallback_contexts[user_id] = []
            
            self.fallback_contexts[user_id].append(content)
            
            # Limit fallback storage size to prevent memory leaks
            if len(self.fallback_memories[user_id]) > 100:
                self.fallback_memories[user_id] = self.fallback_memories[user_id][-50:]
            if len(self.fallback_contexts[user_id]) > 50:
                self.fallback_contexts[user_id] = self.fallback_contexts[user_id][-25:]
            
            logger.info(f"Stored fallback memory for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store fallback memory: {e}")
            return False

    async def retrieve_relevant_memories(self, user_id: int, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories for query (async) with fallback support"""
        if not self.memory_manager or self.fallback_mode:
            # Return fallback memories from in-memory storage
            return self._get_fallback_memories(user_id, top_k)
        
        try:
            memories = await self.memory_manager.retrieve_relevant_memories(user_id, query, top_k)
            self._reset_error_count()  # Reset error count on success
            return memories
        except Exception as e:
            self._handle_error(e, "retrieve_relevant_memories")
            # Fallback to in-memory memories
            return self._get_fallback_memories(user_id, top_k)
    
    def _get_fallback_memories(self, user_id: int, top_k: int) -> List[Dict]:
        """Get memories from fallback in-memory storage"""
        if user_id not in self.fallback_memories:
            return []
        
        memories = self.fallback_memories[user_id][-top_k:]  # Get last top_k memories
        return memories

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
        """Get comprehensive memory system status (sync - no async operations)"""
        status = {
            "available": self.is_available(),
            "database_url": self.database_url if self.database_url else "Not configured",
            "memory_manager_initialized": self.memory_manager is not None,
            "fallback_mode": self.fallback_mode,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "fallback_memories_count": sum(len(memories) for memories in self.fallback_memories.values()),
            "fallback_contexts_count": sum(len(contexts) for contexts in self.fallback_contexts.values()),
            "system_health": self._get_system_health()
        }
        
        # Add user notification if in fallback mode
        if self.fallback_mode:
            status["user_notification"] = {
                "message": "Memory system is operating in limited mode. Some features may be temporarily unavailable.",
                "severity": "warning",
                "timestamp": datetime.now().isoformat()
            }
        
        return status
    
    def _get_system_health(self) -> str:
        """Get system health status"""
        if self.fallback_mode:
            if self.error_count >= self.max_errors:
                return "critical"
            else:
                return "degraded"
        elif self.memory_manager is None:
            return "unavailable"
        else:
            return "healthy"
    
    def get_user_status_message(self) -> Optional[Dict[str, Any]]:
        """Get user-friendly status message for frontend display"""
        if not self.fallback_mode:
            return None
        
        return {
            "type": "memory_system_status",
            "message": "Memory system is operating in limited mode. Your conversation history may not be fully preserved.",
            "severity": "warning",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "error_count": self.error_count,
                "last_error": self.last_error
            }
        }

    async def attempt_recovery(self) -> bool:
        """Attempt to recover from fallback mode"""
        if not self.fallback_mode:
            return True
        
        try:
            # Test database connection
            if self.database_url:
                # Try to reinitialize memory manager
                self.memory_manager = MemoryManager(
                    database_url=self.database_url,
                    embedding_generator=self._generate_embedding,
                    memory_extractor=self._extract_memory_content,
                    pool_size=20
                )
                
                # Test a simple operation
                test_memories = await self.memory_manager.retrieve_relevant_memories(0, "test", 1)
                
                # If we get here, recovery was successful
                self.fallback_mode = False
                self.error_count = 0
                self.last_error = None
                
                self._notify_admin("Memory system recovered successfully from fallback mode", "INFO")
                logger.info("Memory system recovered successfully")
                return True
                
        except Exception as e:
            logger.warning(f"Recovery attempt failed: {e}")
            self.last_error = str(e)
            return False
        
        return False
    
    def clear_fallback_data(self, user_id: Optional[int] = None):
        """Clear fallback data (use with caution)"""
        if user_id is None:
            # Clear all fallback data
            self.fallback_memories.clear()
            self.fallback_contexts.clear()
            logger.info("Cleared all fallback data")
        else:
            # Clear specific user's fallback data
            if user_id in self.fallback_memories:
                del self.fallback_memories[user_id]
            if user_id in self.fallback_contexts:
                del self.fallback_contexts[user_id]
            logger.info(f"Cleared fallback data for user {user_id}")

# Global instance for easy access
memory_system = MemorySystem()
