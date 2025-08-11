from .agent import Agent
from .runner import Runner
from .tools import WebSearchTool

# Import agents from agents_backend
from .agents_backend import agent1, agent2, chatbot_agent, run_agent

# Lazy import to avoid circular dependency
def get_memory_system():
    """Get memory system instance with lazy loading"""
    from .memory_system import memory_system
    return memory_system
