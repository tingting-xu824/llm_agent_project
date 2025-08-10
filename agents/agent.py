class Agent:
    """
    Agent class representing an AI agent with specific instructions and tools
    
    Attributes:
        name (str): Human-readable name of the agent
        instructions (str): System instructions/prompt for the agent
        tools (list): List of tools available to the agent
    """
    
    def __init__(self, name: str, instructions: str, tools=None):
        """
        Initialize an Agent instance
        
        Args:
            name (str): Name of the agent
            instructions (str): System instructions for the agent
            tools (list, optional): List of tools for the agent. Defaults to None.
        """
        self.name = name
        self.instructions = instructions
        self.tools = tools if tools else []

    def __repr__(self):
        """String representation of the Agent"""
        return f"<Agent name={self.name}>"
