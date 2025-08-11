from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List
import redis
import json
import os
import uuid
from agents.agents_backend import agent1, agent2, chatbot_agent, run_agent
from agents.database import (
    get_user_by_token, 
    get_user_by_email_and_dob, 
    create_user, 
    get_user_profile,
    save_conversation_message,
    get_user_conversations,
    create_evaluation_round_data,
    get_evaluation_data
)
from agents.constants import EVALUATION_ROUND_1_TIME
from .memory_system import memory_system
from fastapi import Query
from sqlalchemy import Column

app = FastAPI()

# CORS setup
origins = [
    "http://localhost:3231",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["OPTIONS", "GET", "POST"],
    allow_headers=["*"],
)

# Redis configuration for distributed caching
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
IS_LOCAL = os.getenv("ENV", "dev") == "dev"

# Initialize Redis client for distributed session storage
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    print("Redis connection successful")
except Exception as e:
    print(f"Warning: Redis connection failed: {e}")
    print("Falling back to in-memory storage (not suitable for production)")
    redis_client = None

# Fallback in-memory storage for when Redis is not available
in_memory_conversations = {}
in_memory_last_requests = {}

# Rate limiting configuration
COOLDOWN_SECONDS = 3

class ChatInput(BaseModel):
    """Request model for chat input"""
    message: str
    mode: str  # either "eval" or "chat"

class UserRegistration(BaseModel):
    """User registration model with all required fields"""
    first_name: str
    last_name: str
    email: EmailStr
    confirm_email: EmailStr
    date_of_birth: date
    gender: str
    field_of_education: Optional[str] = None
    current_level_of_education: Optional[str] = None
    disability_knowledge: str  # "yes" or "no"
    ai_course_experience: str  # "yes" or "no"
    
    @validator('confirm_email')
    def emails_must_match(cls, v, values):
        """Validate that email and confirm_email match"""
        if 'email' in values and v != values['email']:
            raise ValueError('Email and confirm email must match')
        return v
    
    @validator('gender')
    def validate_gender(cls, v):
        """Validate gender selection"""
        valid_genders = ['male', 'female', 'other', 'prefer_not_to_say']
        if v.lower() not in valid_genders:
            raise ValueError('Invalid gender selection')
        return v.lower()
    
    @validator('disability_knowledge', 'ai_course_experience')
    def validate_yes_no(cls, v):
        """Validate yes/no fields"""
        if v.lower() not in ['yes', 'no']:
            raise ValueError('Must be either "yes" or "no"')
        return v.lower()
    
    @validator('date_of_birth')
    def validate_date_of_birth(cls, v):
        """Validate date of birth (must be in the past)"""
        if v >= date.today():
            raise ValueError('Date of birth must be in the past')
        return v

class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    date_of_birth: date

class InterviewInit(BaseModel):
    """Interview initialization model"""
    user_id: int

class InterviewChat(BaseModel):
    """Interview chat model"""
    user_id: int
    user_message: str

class InterviewHistory(BaseModel):
    """Interview history request model"""
    user_id: int

async def get_current_user(request: Request) -> Dict:
    """Dependency to get current user from token"""
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(
            status_code=401, 
            detail="Authentication expired!"
        )

    user = get_user_by_token(token)
    
    if not user:
        raise HTTPException(
            status_code=401, 
            detail="Invalid token or user not found"
        )
    
    return user

def get_user_conversation_key(user_id: int) -> str:
    """Generate Redis key for user conversation history"""
    return f"user_conversation:{user_id}"

def get_user_last_request_key(user_id: int) -> str:
    """Generate Redis key for user rate limiting"""
    return f"user_last_request:{user_id}"

def get_user_conversation(user_id: int) -> list[tuple[str, str]]:
    """Get user conversation history from Redis, fallback to memory"""
    if redis_client:
        try:
            conversation_key = get_user_conversation_key(user_id)
            conversation_data = redis_client.get(conversation_key)
            if conversation_data:
                return json.loads(conversation_data)
            return []
        except Exception as e:
            print(f"Error getting user conversation from Redis: {e}")
    
    # Fallback to in-memory storage
    return in_memory_conversations.get(user_id, [])

def save_user_conversation(user_id: int, conversation: list[tuple[str, str]]):
    """Save user conversation history to Redis with 24-hour expiration, fallback to memory"""
    if redis_client:
        try:
            conversation_key = get_user_conversation_key(user_id)
            # Set expiration to 24 hours for automatic cleanup
            redis_client.setex(conversation_key, 86400, json.dumps(conversation))
            return
        except Exception as e:
            print(f"Error saving user conversation to Redis: {e}")
    
    # Fallback to in-memory storage
    in_memory_conversations[user_id] = conversation

def get_user_last_request_time(user_id: int) -> datetime | None :
    """Get user's last request time from Redis for rate limiting, fallback to memory"""
    if redis_client:
        try:
            request_key = get_user_last_request_key(user_id)
            last_time_str = redis_client.get(request_key)
            if last_time_str:
                return datetime.fromisoformat(last_time_str)
            return None
        except Exception as e:
            print(f"Error getting user last request time from Redis: {e}")
    
    # Fallback to in-memory storage
    return in_memory_last_requests.get(user_id)

def save_user_last_request_time(user_id: int, request_time: datetime):
    """Save user's last request time to Redis with 1-hour expiration, fallback to memory"""
    if redis_client:
        try:
            request_key = get_user_last_request_key(user_id)
            # Set expiration to 1 hour for rate limiting
            redis_client.setex(request_key, 3600, request_time.isoformat())
            return
        except Exception as e:
            print(f"Error saving user last request time to Redis: {e}")
    
    # Fallback to in-memory storage
    in_memory_last_requests[user_id] = request_time

# Async versions of the above functions for use in async endpoints
async def get_user_conversation_async(user_id: int) -> list[tuple[str, str]]:
    """Async version of get_user_conversation"""
    if redis_client:
        try:
            conversation_key = get_user_conversation_key(user_id)
            conversation_data = await redis_client.get(conversation_key)
            if conversation_data:
                return json.loads(conversation_data)
            return []
        except Exception as e:
            print(f"Error getting user conversation from Redis: {e}")
    
    # Fallback to in-memory storage
    return in_memory_conversations.get(user_id, [])

async def save_user_conversation_async(user_id: int, conversation: list[tuple[str, str]]):
    """Async version of save_user_conversation"""
    if redis_client:
        try:
            conversation_key = get_user_conversation_key(user_id)
            # Set expiration to 24 hours for automatic cleanup
            await redis_client.setex(conversation_key, 86400, json.dumps(conversation))
            return
        except Exception as e:
            print(f"Error saving user conversation to Redis: {e}")
    
    # Fallback to in-memory storage
    in_memory_conversations[user_id] = conversation

async def get_user_last_request_time_async(user_id: int) -> datetime | None:
    """Async version of get_user_last_request_time"""
    if redis_client:
        try:
            request_key = get_user_last_request_key(user_id)
            last_time_str = await redis_client.get(request_key)
            if last_time_str:
                return datetime.fromisoformat(last_time_str)
            return None
        except Exception as e:
            print(f"Error getting user last request time from Redis: {e}")
    
    # Fallback to in-memory storage
    return in_memory_last_requests.get(user_id)

async def save_user_last_request_time_async(user_id: int, request_time: datetime):
    """Async version of save_user_last_request_time"""
    if redis_client:
        try:
            request_key = get_user_last_request_key(user_id)
            # Set expiration to 1 hour for rate limiting
            await redis_client.setex(request_key, 3600, request_time.isoformat())
            return
        except Exception as e:
            print(f"Error saving user last request time to Redis: {e}")
    
    # Fallback to in-memory storage
    in_memory_last_requests[user_id] = request_time

async def save_conversation_message_async(user_id: int, message: str, role: str, mode: str, agent_type: int) -> Optional[Column[int]]:
    """Async version of save_conversation_message - optimized to minimize thread pool usage"""
    from agents.database import save_conversation_message_async as db_save_conversation
    return await db_save_conversation(user_id, message, role, mode, agent_type)

@app.post("/users/register")
async def register_user(user_data: UserRegistration):
    """Register a new user with comprehensive profile information"""
    try:
        # Generate a UUID token
        token = str(uuid.uuid4())
        
        # Assign agent_type based on user ID for load balancing (will be set after user creation)
        # We'll set this after getting the user_id
        agent_type = None
        
        # Prepare user data for database (without agent_type initially)
        db_user_data = {
            "email": user_data.email,
            "dob": user_data.date_of_birth,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "agent_type": 1,  # Temporary value, will be updated
            "gender": user_data.gender,
            "education_field": user_data.field_of_education,
            "education_level": user_data.current_level_of_education,
            "disability_knowledge": user_data.disability_knowledge,
            "genai_course_exp": user_data.ai_course_experience,
            "token": token
        }
        
        # Create user using database module
        user_id = create_user(db_user_data)
        create_evaluation_round_data(user_id=user_id, problem="", solution="", ai_feedback=None, round="round-1",time_remaining=EVALUATION_ROUND_1_TIME)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="Registration failed - duplicate data")
        
        # Assign agent_type based on user ID for load balancing
        agent_type = 1 if int(str(user_id)[-1]) % 2 == 1 else 2
        
        # Update the user's agent_type in the database
        from agents.database import update_user_agent_type
        update_user_agent_type(user_id, agent_type)
         # Prepare response data
        resp_data = {
            "user_id": user_id,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "email": user_data.email,
            "token": token,
            "message": "User registered successfully"
        }
        
        response = JSONResponse(content=resp_data)
        response.set_cookie(
            key="auth_token",
            value=token,
            httponly=True,      # JS can't access cookie
            secure=not IS_LOCAL,        # only over HTTPS
            samesite="none" if not IS_LOCAL else "strict",     # adjust if cross-site needed
            max_age=60 * 60 * 24 * 7  # 7 days
        )
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/users/login")
async def login_user(login_data: UserLogin):
    """Login user with email and date of birth"""
    try:
        user = get_user_by_email_and_dob(login_data.email, login_data.date_of_birth)
        
        if not user:
            raise HTTPException(
                status_code=401, 
                detail="Invalid email or date of birth"
            )
        
        response_data = {
            "user_id": user["user_id"],
            "email": user["email"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "token": user["token"],
            "message": "Login successful"
        }

        response  = JSONResponse(content=response_data)
        response.set_cookie(
            key="auth_token",
            value=response_data["token"],
            httponly=True,      # JS can't access cookie
            secure=not IS_LOCAL,        # only over HTTPS
            samesite="none" if not IS_LOCAL else "strict",     # adjust if cross-site needed
            max_age=60 * 60 * 24 * 7  # 7 days
        )
        return response
    
    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.get("/evaluation")
async def get_evaluation_by_round(
    current_user: Dict = Depends(get_current_user),
    round: Optional[int] = Query(None, alias="round", ge=1, le=3, description="Agent round (1, 2, or 3)")
):
    """
    Get evaluation rounds for the current user, optionally filtered by round.
    Use ?round=1, ?round=2, ?round=3 or omit for all rounds.
    """
    user_id = current_user["user_id"]
    return get_evaluation_data(user_id, round)

@app.post("/agent")
async def post_agent(input: ChatInput, current_user: Dict = Depends(get_current_user)):
    """Handle chat requests with token-based authentication, rate limiting, and RAG memory"""
    user_id = current_user["user_id"]
    
    # Rate limiting check
    now = datetime.utcnow()
    last_time = await get_user_last_request_time_async(user_id)
    if last_time and (now - last_time).total_seconds() < COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait.")

    await save_user_last_request_time_async(user_id, now)

    # Select agent and model based on mode
    if input.mode == "eval":
        # Get user's assigned agent_type from database
        from agents.database import get_user_profile
        user_profile = get_user_profile(user_id)
        agent_type = user_profile.get("agent_type", 1) if user_profile else 1
        
        # Assign agent based on stored agent_type
        assigned_agent = agent1 if agent_type == 1 else agent2
        model = "o3"  # Using o3 model with proper parameters
    elif input.mode == "chat":
        assigned_agent = chatbot_agent
        model = "gpt-4o-mini"
        agent_type = 3  # chatbot agent
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    # Get user conversation history from Redis (for immediate context)
    history = await get_user_conversation_async(user_id)
    user_msg = input.message
    
    # Store user message in database
    await save_conversation_message_async(user_id, user_msg, "user", input.mode, agent_type)
    
    # Get relevant memories for RAG
    memory_context = await memory_system.create_memory_context(user_id, user_msg, top_k=3)
    
    # Prepare enhanced prompt with memory context
    enhanced_prompt = user_msg
    if memory_context:
        enhanced_prompt = f"""Previous relevant context from our conversations:
{memory_context}

Current message: {user_msg}

Please respond to the current message while considering the relevant context from our previous conversations."""
    
    # Add to immediate history for agent
    history.append(("User", enhanced_prompt))

    # Call agent with enhanced prompt and conversation history
    agent_reply = await run_agent(assigned_agent, user_id, enhanced_prompt, history, model, input.mode)
    
    # Store agent reply in database
    await save_conversation_message_async(user_id, agent_reply, "assistant", input.mode, agent_type)
    
    # Store important parts in memory system for future RAG
    # Store user message if it contains important information
    if len(user_msg) > 20:  # Only store substantial messages
        await memory_system.store_memory(
            user_id, 
            user_msg, 
            metadata={
                "type": "user_message",
                "mode": input.mode,
                "agent_type": agent_type,
                "timestamp": now.isoformat()
            }
        )
    
    # Store agent reply if it contains substantial information
    if len(agent_reply) > 30:  # Only store substantial responses
        await memory_system.store_memory(
            user_id, 
            agent_reply, 
            metadata={
                "type": "agent_response",
                "mode": input.mode,
                "agent_type": agent_type,
                "timestamp": now.isoformat()
            }
        )
    
    # Update immediate history for Redis
    history.append(("Assistant", agent_reply))
    await save_user_conversation_async(user_id, history)

    return [
        {
            "message": agent_reply,
            "type": "agent_response",
            "timestamp": (datetime.utcnow() + timedelta(seconds=1)).isoformat() + "Z",
            "memory_context_used": bool(memory_context)
        }
    ]

@app.get("/agent")
async def get_history(current_user: Dict = Depends(get_current_user)):
    """Get user's conversation history from Redis"""
    user_id = current_user["user_id"]
    
    history = await get_user_conversation_async(user_id)
    if not history:
        raise HTTPException(status_code=404, detail="No history found for this user.")

    response = []
    for role, msg in history:
        response.append({
            "message": msg,
            "type": "user_input" if role == "User" else "agent_response",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    return response

@app.get("/conversations")
async def get_conversation_history(current_user: Dict = Depends(get_current_user), limit: int = 50):
    """Get user's conversation history from database"""
    user_id = current_user["user_id"]
    
    # Use optimized async function to minimize thread pool usage
    from agents.database import get_user_conversations_async
    conversations = await get_user_conversations_async(user_id, limit)
    
    if not conversations:
        raise HTTPException(status_code=404, detail="No conversation history found for this user.")
    
    return conversations

@app.get("/memories")
async def get_user_memories(current_user: Dict = Depends(get_current_user), query: str = "", top_k: int = 5):
    """Get user's relevant memories for a query"""
    user_id = current_user["user_id"]
    
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    
    memories = await memory_system.retrieve_relevant_memories(user_id, query, top_k)
    return {
        "query": query,
        "memories": memories,
        "count": len(memories)
    }

@app.get("/users/profile")
async def get_user_profile_endpoint(current_user: Dict = Depends(get_current_user)):
    """Get current user's complete profile information"""
    try:
        # Use optimized async function to minimize thread pool usage
        from agents.database import get_user_profile_async
        profile = await get_user_profile_async(current_user["user_id"])
        
        if profile:
            return profile
        else:
            raise HTTPException(status_code=404, detail="User profile not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")

# Interview-related storage (in-memory for now, can be moved to database later)
interview_sessions = {}
interview_chat_history = {}

def get_interview_session_key(user_id: int) -> str:
    """Generate key for interview session"""
    return f"interview_session:{user_id}"

def get_interview_history_key(user_id: int) -> str:
    """Generate key for interview chat history"""
    return f"interview_history:{user_id}"

def get_interview_session(user_id: int) -> Dict:
    """Get interview session for user"""
    if redis_client:
        try:
            session_key = get_interview_session_key(user_id)
            session_data = redis_client.get(session_key)
            if session_data:
                return json.loads(session_data)
        except Exception as e:
            print(f"Error getting interview session from Redis: {e}")
    
    return interview_sessions.get(user_id, {})

def save_interview_session(user_id: int, session_data: Dict):
    """Save interview session for user"""
    if redis_client:
        try:
            session_key = get_interview_session_key(user_id)
            redis_client.setex(session_key, 3600, json.dumps(session_data))  # 1 hour expiration
            return
        except Exception as e:
            print(f"Error saving interview session to Redis: {e}")
    
    interview_sessions[user_id] = session_data

def get_interview_chat_history(user_id: int) -> List[Dict]:
    """Get interview chat history for user"""
    if redis_client:
        try:
            history_key = get_interview_history_key(user_id)
            history_data = redis_client.get(history_key)
            if history_data:
                return json.loads(history_data)
        except Exception as e:
            print(f"Error getting interview history from Redis: {e}")
    
    return interview_chat_history.get(user_id, [])

def save_interview_chat_history(user_id: int, history: List[Dict]):
    """Save interview chat history for user"""
    if redis_client:
        try:
            history_key = get_interview_history_key(user_id)
            redis_client.setex(history_key, 86400, json.dumps(history))  # 24 hour expiration
            return
        except Exception as e:
            print(f"Error saving interview history to Redis: {e}")
    
    interview_chat_history[user_id] = history

def add_interview_message(user_id: int, role: str, content: str):
    """Add a message to interview chat history"""
    history = get_interview_chat_history(user_id)
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    history.append(message)
    save_interview_chat_history(user_id, history)

@app.post("/interview/init")
async def initialize_interview(request: InterviewInit, current_user: Dict = Depends(get_current_user)):
    """Initialize a chat session for an interview"""
    try:
        user_id = request.user_id
        
        # Verify the user_id matches the authenticated user
        if user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="User ID mismatch")
        
        # Check if user exists
        user_profile = get_user_profile(user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Initialize interview session
        session_data = {
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        save_interview_session(user_id, session_data)
        
        # Create initial interview question
        initial_question = "Hello! I'm here to conduct an interview with you about your experience with our AI agents. Let's start with a simple question: How would you rate your overall experience with the AI agents you interacted with today? Please rate it on a scale of 1-10, where 1 is very poor and 10 is excellent."
        
        # Add initial question to chat history
        add_interview_message(user_id, "assistant", initial_question)
        
        return {
            "response": initial_question,
            "interview_id": user_id  # 使用user_id作为interview标识
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize interview: {str(e)}")

@app.post("/interview/chat")
async def interview_chat(request: InterviewChat, current_user: Dict = Depends(get_current_user)):
    """Send a user message to the Interview Bot and receive the assistant's reply"""
    try:
        user_id = request.user_id
        user_message = request.user_message
        
        # Verify the user_id matches the authenticated user
        if user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="User ID mismatch")
        
        # Check if interview session exists
        session = get_interview_session(user_id)
        if not session or session.get("status") != "active":
            raise HTTPException(status_code=400, detail="No active interview session found. Please initialize the interview first.")
        
        # Add user message to history
        add_interview_message(user_id, "user", user_message)
        
        # Get chat history for context
        history = get_interview_chat_history(user_id)
        
        # Generate interview bot response based on context
        # This is a simple interview bot - you can enhance it with more sophisticated logic
        if len(history) <= 2:  # First exchange
            response = "Thank you for your rating! Could you tell me more about what aspects of the AI agents you found most helpful or most challenging?"
        elif len(history) <= 4:  # Second exchange
            response = "That's very helpful feedback. Did you notice any differences between the different AI agents you interacted with? If so, what were they?"
        elif len(history) <= 6:  # Third exchange
            response = "Interesting! One more question: How likely are you to use AI agents like these in the future, and what would make you more likely to use them?"
        elif len(history) <= 8:  # Fourth exchange
            response = "Thank you for sharing your thoughts! Is there anything else you'd like to tell us about your experience or any suggestions for improvement?"
        else:  # Final exchange
            response = "Thank you so much for participating in this interview! Your feedback is very valuable to us. Is there anything else you'd like to add before we conclude?"
            # Mark session as completed
            session["status"] = "completed"
            session["completed_at"] = datetime.utcnow().isoformat()
            save_interview_session(user_id, session)
        
        # Add assistant response to history
        add_interview_message(user_id, "assistant", response)
        
        return {
            "response": response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process interview chat: {str(e)}")

@app.post("/interview/history")
async def get_interview_history(request: InterviewHistory, current_user: Dict = Depends(get_current_user)):
    """Retrieve the chat history for the interview"""
    try:
        user_id = request.user_id
        
        # Verify the user_id matches the authenticated user
        if user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="User ID mismatch")
        
        # Get interview chat history
        chat_history = get_interview_chat_history(user_id)
        
        if not chat_history:
            raise HTTPException(status_code=404, detail="No interview history found for this user")
        
        return {
            "chat_history": chat_history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interview history: {str(e)}")
