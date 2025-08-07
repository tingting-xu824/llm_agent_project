from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict
from agents.agents_backend import agent1, agent2, chatbot_agent, run_agent

app = FastAPI()

user_sessions: Dict[str, list[tuple[str, str]]] = {}
user_last_request_time: Dict[str, datetime] = {}

COOLDOWN_SECONDS = 3

class ChatInput(BaseModel):
    message: str
    mode: str  # either "eval" or "chat"

@app.post("/agent")
async def post_agent(input: ChatInput, request: Request):
    user_id = request.cookies.get("userID")
    session_id = request.cookies.get("sessionID")

    if not user_id or not session_id:
        raise HTTPException(status_code=400, detail="Missing userID or sessionID in cookies")

    now = datetime.utcnow()
    last_time = user_last_request_time.get(user_id)
    if last_time and (now - last_time).total_seconds() < COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait.")

    user_last_request_time[user_id] = now

    # Select agent and model based on mode
    if input.mode == "eval":
        assigned_agent = agent1 if int(user_id[-1]) % 2 == 1 else agent2
        model = "gpt-3.5-turbo"
    elif input.mode == "chat":
        assigned_agent = chatbot_agent
        model = "gpt-4o"
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    # Maintain user history
    if user_id not in user_sessions:
        user_sessions[user_id] = []

    history = user_sessions[user_id]
    user_msg = input.message
    history.append(("User", user_msg))

    # Call agent
    agent_reply = await run_agent(assigned_agent, user_msg, history, model)
    history.append(("Assistant", agent_reply))

    return [
        {
            "message": agent_reply,
            "type": "agent_response",
            "timestamp": (datetime.utcnow() + timedelta(seconds=1)).isoformat() + "Z"
        }
    ]

@app.get("/agent")
def get_history(request: Request):
    user_id = request.cookies.get("userID")
    if not user_id or user_id not in user_sessions:
        raise HTTPException(status_code=404, detail="No history found for this user.")

    history = user_sessions[user_id]
    response = []
    for role, msg in history:
        response.append({
            "message": msg,
            "type": "user_input" if role == "User" else "agent_response",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    return response
