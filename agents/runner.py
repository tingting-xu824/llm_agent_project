import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Runner:
    """Static class for executing agents with OpenAI API"""
    
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
        try:
            # Prepare base parameters
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
                # No additional parameters needed
                pass
            else:
                # Standard models support more parameters
                params.update({
                    "temperature": 0.7,  # Controls response creativity
                    "max_tokens": 4000
                })
            
            # Create chat completion with appropriate parameters
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"
