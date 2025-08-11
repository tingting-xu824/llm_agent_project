import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

# Load environment variables from .env file
load_dotenv()

class AIServiceManager:
    """Comprehensive AI service manager with overlapping key allocation strategy"""
    
    def __init__(self):
        # Multiple API keys for load balancing
        self.api_keys = [
            os.getenv("OPENAI_API_KEY_1"),
            os.getenv("OPENAI_API_KEY_2"), 
            os.getenv("OPENAI_API_KEY_3"),
            os.getenv("OPENAI_API_KEY_4"),
            os.getenv("OPENAI_API_KEY_5"),
        ]
        
        # Filter out None values and validate
        self.api_keys = [key for key in self.api_keys if key]
        if not self.api_keys:
            raise ValueError("No valid OpenAI API keys found. Please set OPENAI_API_KEY_1 through OPENAI_API_KEY_5")
        
        # Overlapping key allocation strategy
        # Key 1, 2, 3: Priority for chat
        # Key 2, 3, 4: Shared between chat and memory
        # Key 4, 5: Priority for memory
        self.chat_priority_indices = [0, 1, 2]  # Key 1, 2, 3
        self.shared_indices = [1, 2, 3]         # Key 2, 3, 4
        self.memory_priority_indices = [3, 4]   # Key 4, 5
        
        # Current indices for round-robin within each group
        self.chat_index = 0
        self.shared_index = 0
        self.memory_index = 0
        
        # Model configuration for different phases
        self.eval_model = "gpt-4o"  # Evaluation phase model
        self.chat_model = "gpt-4o"  # Chat phase model
        
        print(f"AI Service Manager initialized with {len(self.api_keys)} API keys")
        print(f"Chat priority keys: {[i+1 for i in self.chat_priority_indices]}")
        print(f"Shared keys: {[i+1 for i in self.shared_indices]}")
        print(f"Memory priority keys: {[i+1 for i in self.memory_priority_indices]}")
    
    def _get_next_chat_client(self) -> Optional[OpenAI]:
        """Get next client from chat priority keys"""
        if not self.chat_priority_indices:
            return None
        
        key_index = self.chat_priority_indices[self.chat_index]
        self.chat_index = (self.chat_index + 1) % len(self.chat_priority_indices)
        return OpenAI(api_key=self.api_keys[key_index])
    
    def _get_next_memory_client(self) -> Optional[OpenAI]:
        """Get next client from memory priority keys"""
        if not self.memory_priority_indices:
            return None
        
        key_index = self.memory_priority_indices[self.memory_index]
        self.memory_index = (self.memory_index + 1) % len(self.memory_priority_indices)
        return OpenAI(api_key=self.api_keys[key_index])
    
    def _get_next_shared_client(self) -> Optional[OpenAI]:
        """Get next client from shared keys"""
        if not self.shared_indices:
            return None
        
        key_index = self.shared_indices[self.shared_index]
        self.shared_index = (self.shared_index + 1) % len(self.shared_indices)
        return OpenAI(api_key=self.api_keys[key_index])
    
    async def _call_with_fallback(self, priority_clients: List[OpenAI], fallback_clients: List[OpenAI], 
                                 operation: str, **kwargs) -> Any:
        """Execute operation with priority and fallback clients"""
        # Try priority clients first
        for client in priority_clients:
            try:
                if operation == "chat_completion":
                    response = client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content.strip()
                elif operation == "embedding":
                    response = client.embeddings.create(**kwargs)
                    return response.data[0].embedding
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            except Exception as e:
                print(f"Priority client failed: {e}")
                continue
        
        # Try fallback clients if priority clients fail
        for client in fallback_clients:
            try:
                if operation == "chat_completion":
                    response = client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content.strip()
                elif operation == "embedding":
                    response = client.embeddings.create(**kwargs)
                    return response.data[0].embedding
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            except Exception as e:
                print(f"Fallback client failed: {e}")
                continue
        
        # If all clients fail, return error response
        if operation == "chat_completion":
            return f"[OpenAI Error] All API keys failed for chat completion"
        elif operation == "embedding":
            return [0.0] * 1536  # Default embedding
        else:
            raise Exception(f"All API keys failed for operation: {operation}")
    
    async def run_agent(self, agent, prompt: str, model: str | None = None) -> str:
        """
        Execute an agent with the given prompt using OpenAI API
        Priority: Chat keys (1,2,3) -> Shared keys (2,3,4)
        
        Args:
            agent: Agent instance with instructions and tools
            prompt (str): User prompt/message to send to the agent
            model (str): OpenAI model to use (if None, uses default based on phase)
        
        Returns:
            str: Agent's response message
        """
        try:
            # Use default model if not specified
            if model is None:
                model = self.chat_model
            
            # Prepare parameters
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": agent.instructions or ""},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add model-specific parameters
            if model == "o3":
                # o3 model only supports basic parameters
                pass
            else:
                # Standard models support more parameters
                params.update({
                    "temperature": 0.7,  # Controls response creativity
                    "max_tokens": 4000
                })
            
            # Get priority clients (chat keys)
            priority_clients = []
            for _ in range(len(self.chat_priority_indices)):
                client = self._get_next_chat_client()
                if client:
                    priority_clients.append(client)
            
            # Get fallback clients (shared keys)
            fallback_clients = []
            for _ in range(len(self.shared_indices)):
                client = self._get_next_shared_client()
                if client:
                    fallback_clients.append(client)
            
            # Execute with fallback
            return await self._call_with_fallback(
                priority_clients, fallback_clients, "chat_completion", **params
            )
            
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using OpenAI API
        Priority: Memory keys (4,5) -> Shared keys (2,3,4)
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Get priority clients (memory keys)
            priority_clients = []
            for _ in range(len(self.memory_priority_indices)):
                client = self._get_next_memory_client()
                if client:
                    priority_clients.append(client)
            
            # Get fallback clients (shared keys)
            fallback_clients = []
            for _ in range(len(self.shared_indices)):
                client = self._get_next_shared_client()
                if client:
                    fallback_clients.append(client)
            
            # Execute with fallback
            return await self._call_with_fallback(
                priority_clients, fallback_clients, "embedding",
                model="text-embedding-3-small",
                input=text
            )
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a default embedding (all zeros) as fallback
            return [0.0] * 1536
    
    async def extract_memory_content(self, conversation_text: str, memory_type: str) -> str:
        """
        Extract key memory content from conversation using OpenAI API
        Priority: Memory keys (4,5) -> Shared keys (2,3,4)
        
        Args:
            conversation_text: Full conversation text
            memory_type: Type of memory to extract ("eval" or "chat")
            
        Returns:
            str: Extracted memory content
        """
        try:
            # Create prompt based on memory type
            if memory_type == "eval":
                prompt = f"""Extract the key insights and decisions from this evaluation conversation. Focus on:
- Main ideas discussed
- Decisions made
- Problems identified
- Solutions proposed

Conversation:
{conversation_text}

Summary:"""
            else:  # chat mode
                prompt = f"""Extract the key technical details and implementation information from this conversation. Focus on:
- Technical solutions discussed
- Implementation details
- Code snippets or approaches
- Important decisions made

Conversation:
{conversation_text}

Summary:"""
            
            # Get priority clients (memory keys)
            priority_clients = []
            for _ in range(len(self.memory_priority_indices)):
                client = self._get_next_memory_client()
                if client:
                    priority_clients.append(client)
            
            # Get fallback clients (shared keys)
            fallback_clients = []
            for _ in range(len(self.shared_indices)):
                client = self._get_next_shared_client()
                if client:
                    fallback_clients.append(client)
            
            # Execute with fallback
            return await self._call_with_fallback(
                priority_clients, fallback_clients, "chat_completion",
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key information from conversations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
        except Exception as e:
            print(f"Error extracting memory content: {e}")
            return f"Memory extraction failed: {str(e)}"
    
    def get_model_for_phase(self, phase: str) -> str:
        """
        Get the appropriate model for a given phase
        
        Args:
            phase: "eval" or "chat"
            
        Returns:
            str: Model name
        """
        if phase == "eval":
            return self.eval_model
        elif phase == "chat":
            return self.chat_model
        else:
            return self.chat_model  # Default to chat model
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [self.eval_model, self.chat_model]
    
    def get_api_key_count(self) -> int:
        """Get number of available API keys"""
        return len(self.api_keys)
    
    def get_allocation_info(self) -> Dict[str, Any]:
        """Get current allocation strategy information"""
        return {
            "total_keys": len(self.api_keys),
            "chat_priority_keys": [i+1 for i in self.chat_priority_indices],
            "shared_keys": [i+1 for i in self.shared_indices],
            "memory_priority_keys": [i+1 for i in self.memory_priority_indices],
            "current_chat_index": self.chat_index,
            "current_shared_index": self.shared_index,
            "current_memory_index": self.memory_index
        }

# Global instance for easy access
ai_service_manager = AIServiceManager()

# Backward compatibility - keep the old Runner class for existing code
class Runner:
    """Static class for executing agents with OpenAI API (backward compatibility)"""
    
    @staticmethod
    async def run(agent, prompt: str, model: str) -> str:
        """
        Execute an agent with the given prompt using OpenAI API
        
        Args:
            agent: Agent instance with instructions and tools
            prompt (str): User prompt/message to send to the agent
            model (str): OpenAI model to use (e.g., "gpt-4o-mini", "o3")
        
        Returns:
            str: Agent's response message
            
        Raises:
            Exception: If OpenAI API call fails, returns error message
        """
        return await ai_service_manager.run_agent(agent, prompt, model)
