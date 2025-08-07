from agents import Agent, Runner, WebSearchTool
import os
from dotenv import load_dotenv

load_dotenv()

def load_prompt(file_name: str) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "../instructions")
    file_path = os.path.join(base_dir, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""  # Handle empty prompt if file not found

# Agent 1 for originality checking
agent1 = Agent(
    name="Originality Checker",
    instructions=load_prompt("agent1.txt"),
    tools=[WebSearchTool()]
)

# Agent 2 for idea generation
agent2 = Agent(
    name="Creative Idea Generator",
    instructions=load_prompt("agent2.txt")
)

# Chatbot agent (no prompt, default to empty string)
chatbot_agent = Agent(
    name="AI Chatbot",
    instructions=""  # or load_prompt("chatbot_agent.txt") if file exists
)

# Unified run logic
async def run_agent(agent, user_message: str, history: list[tuple[str, str]], model: str) -> str:
    prompt = ""
    for role, msg in history:
        prompt += f"{role}: {msg}\n"
    prompt += f"User: {user_message}\nAssistant:"

    assistant_reply = await Runner.run(agent, prompt, model)
    return assistant_reply.strip()