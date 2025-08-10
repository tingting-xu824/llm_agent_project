from agents import Agent, Runner, WebSearchTool
from .memory_system import memory_system
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_prompt(file_name: str) -> str:
    """Load agent instructions from text files in the instructions directory"""
    base_dir = os.path.join(os.path.dirname(__file__), "../instructions")
    file_path = os.path.join(base_dir, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""  # Return empty string if prompt file not found

# Agent 1: Originality Checker for idea evaluation
agent1 = Agent(
    name="Originality Checker",
    instructions=load_prompt("agent1.txt"),
    tools=[WebSearchTool()]  # Equipped with web search capability
)

# Agent 2: Creative Idea Generator for solution enhancement
agent2 = Agent(
    name="Creative Idea Generator",
    instructions=load_prompt("agent2.txt")
    # No tools needed for this agent
)

# Chatbot agent for general conversation
chatbot_agent = Agent(
    name="AI Chatbot",
    instructions=load_prompt("chatbot_agent.txt")
)

# Enhanced agent execution logic with memory integration
async def run_agent(agent, user_id: int, user_message: str, history: list[tuple[str, str]], model: str, mode: str = "chat") -> str:
    """
    Execute agent with conversation history and memory context
    
    Args:
        agent: The agent instance to run
        user_id: User ID for memory retrieval
        user_message: Current user input message
        history: List of (role, message) tuples representing conversation history
        model: OpenAI model to use (e.g., "gpt-4o-mini", "o3")
        mode: Operation mode ("eval" or "chat")
    
    Returns:
        str: Agent's response message
    """
    # Get relevant memories for context
    memory_context = memory_system.create_memory_context(user_id, user_message, top_k=3)
    
    # Build conversation prompt from history
    prompt = ""
    for role, msg in history:
        prompt += f"{role}: {msg}\n"
    
    # Add memory context if available
    if memory_context:
        prompt += f"\nPrevious relevant context from our conversations:\n{memory_context}\n"
    
    prompt += f"User: {user_message}\nAssistant:"

    # Execute agent with the enhanced prompt
    assistant_reply = await Runner.run(agent, prompt, model)
    return assistant_reply.strip()

# Backward compatibility function (for existing code)
async def run_agent_legacy(agent, user_message: str, history: list[tuple[str, str]], model: str) -> str:
    """
    Legacy agent execution function for backward compatibility
    
    Args:
        agent: The agent instance to run
        user_message: Current user input message
        history: List of (role, message) tuples representing conversation history
        model: OpenAI model to use (e.g., "gpt-4o-mini", "o3")
    
    Returns:
        str: Agent's response message
    """
    # Build conversation prompt from history
    prompt = ""
    for role, msg in history:
        prompt += f"{role}: {msg}\n"
    prompt += f"User: {user_message}\nAssistant:"

    # Execute agent with the constructed prompt
    assistant_reply = await Runner.run(agent, prompt, model)
    return assistant_reply.strip()