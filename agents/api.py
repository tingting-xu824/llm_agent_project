from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta, date
from typing import Dict, Optional
import redis
import json
import os
import uuid
from agents.agents_backend import agent1, agent2, chatbot_agent, run_agent
from agents.database import (
    get_user_by_token, 
    get_user_by_email_and_dob, 
    create_user, 
    update_user_login_time, 
    get_user_profile,
    save_conversation_message,
    get_user_conversations
)
from agents.memory_system import memory_system

app = FastAPI()
security = HTTPBearer()

# Redis configuration for distributed caching
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

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

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Dependency to get current user from token"""
    token = credentials.credentials
    user = get_user_by_token(token)
    
    if not user:
        raise HTTPException(
            status_code=401, 
            detail="Invalid token or user not found"
        )
    
    # Update last login time
    update_user_login_time(user["user_id"])
    
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

def get_user_last_request_time(user_id: int) -> datetime:
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

@app.post("/users/register")
async def register_user(user_data: UserRegistration):
    """Register a new user with comprehensive profile information"""
    try:
        # Generate a UUID token
        token = str(uuid.uuid4())
        
        # Assign agent_type based on user data (1 or 2)
        agent_type = 1 if user_data.gender in ['male', 'female'] else 2
        
        # Prepare user data for database
        db_user_data = {
            "email": user_data.email,
            "dob": user_data.date_of_birth,
            "agent_type": agent_type,
            "gender": user_data.gender,
            "education_field": user_data.field_of_education,
            "education_level": user_data.current_level_of_education,
            "disability_knowledge": user_data.disability_knowledge,
            "genai_course_exp": user_data.ai_course_experience,
            "token": token
        }
        
        # Create user using database module
        user_id = create_user(db_user_data)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="Registration failed - duplicate data")
        
        return {
            "user_id": user_id,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "email": user_data.email,
            "token": token,
            "message": "User registered successfully"
        }
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
        
        # Update last login time
        update_user_login_time(user["user_id"])
        
        return {
            "user_id": user["user_id"],
            "email": user["email"],
            "token": user["token"],
            "message": "Login successful"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/agent")
async def post_agent(input: ChatInput, current_user: Dict = Depends(get_current_user)):
    """Handle chat requests with token-based authentication, rate limiting, and RAG memory"""
    user_id = current_user["user_id"]
    
    # Rate limiting check
    now = datetime.utcnow()
    last_time = get_user_last_request_time(user_id)
    if last_time and (now - last_time).total_seconds() < COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait.")

    save_user_last_request_time(user_id, now)

    # Select agent and model based on mode
    if input.mode == "eval":
        # Assign agent based on user ID for load balancing
        assigned_agent = agent1 if int(str(user_id)[-1]) % 2 == 1 else agent2
        model = "o3"  # Using o3 model with proper parameters
        agent_type = 1 if int(str(user_id)[-1]) % 2 == 1 else 2
    elif input.mode == "chat":
        assigned_agent = chatbot_agent
        model = "gpt-4o-mini"
        agent_type = 3  # chatbot agent
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    # Get user conversation history from Redis (for immediate context)
    history = get_user_conversation(user_id)
    user_msg = input.message
    
    # Store user message in database
    save_conversation_message(user_id, user_msg, "user", input.mode, agent_type)
    
    # Get relevant memories for RAG
    memory_context = memory_system.create_memory_context(user_id, user_msg, top_k=3)
    
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
    agent_reply = await run_agent(assigned_agent, enhanced_prompt, history, model)
    
    # Store agent reply in database
    save_conversation_message(user_id, agent_reply, "assistant", input.mode, agent_type)
    
    # Store important parts in memory system for future RAG
    # Store user message if it contains important information
    if len(user_msg) > 20:  # Only store substantial messages
        memory_system.store_memory(
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
        memory_system.store_memory(
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
    save_user_conversation(user_id, history)

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
    
    history = get_user_conversation(user_id)
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
    
    conversations = get_user_conversations(user_id, limit)
    if not conversations:
        raise HTTPException(status_code=404, detail="No conversation history found for this user.")
    
    return conversations

@app.get("/memories")
async def get_user_memories(current_user: Dict = Depends(get_current_user), query: str = "", top_k: int = 5):
    """Get user's relevant memories for a query"""
    user_id = current_user["user_id"]
    
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    
    memories = memory_system.retrieve_relevant_memories(user_id, query, top_k)
    return {
        "query": query,
        "memories": memories,
        "count": len(memories)
    }

@app.get("/users/profile")
async def get_user_profile_endpoint(current_user: Dict = Depends(get_current_user)):
    """Get current user's complete profile information"""
    try:
        profile = get_user_profile(current_user["user_id"])
        
        if profile:
            return profile
        else:
            raise HTTPException(status_code=404, detail="User profile not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")
