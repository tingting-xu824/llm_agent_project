import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Runner:
    @staticmethod
    async def run(agent, prompt: str, model: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,  # "gpt-3.5-turbo" or "gpt-4o"
                messages=[
                    {"role": "system", "content": agent.instructions or ""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"
