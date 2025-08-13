from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List
import redis.asyncio as aioredis
import json
import os
import uuid
from agents.agents_backend import agent1, agent2, chatbot_agent, run_agent
from agents.database import (
    update_evaluation_round_time,
<<<<<<< Updated upstream
=======
    get_evaluation_data,
>>>>>>> Stashed changes
    get_evaluation_record_by_round_async,
    update_evaluation_record_async,
    complete_evaluation_round_async,
    check_previous_round_completed_async
)
from agents.constants import (
    EVALUATION_ROUND_1_TIME,
    ROUND_1_PROBLEM_MIN_WORDS, ROUND_1_PROBLEM_MAX_WORDS, ROUND_1_SOLUTION_MIN_WORDS, ROUND_1_SOLUTION_MAX_WORDS,
    ROUND_2_PROBLEM_MIN_WORDS, ROUND_2_PROBLEM_MAX_WORDS, ROUND_2_SOLUTION_MIN_WORDS, ROUND_2_SOLUTION_MAX_WORDS,
    ROUND_3_PROBLEM_MIN_WORDS, ROUND_3_PROBLEM_MAX_WORDS, ROUND_3_SOLUTION_MIN_WORDS, ROUND_3_SOLUTION_MAX_WORDS,
    ROUND_4_PROBLEM_MIN_WORDS, ROUND_4_PROBLEM_MAX_WORDS, ROUND_4_SOLUTION_MIN_WORDS, ROUND_4_SOLUTION_MAX_WORDS
)
from .memory_system import memory_system
from fastapi import Query
from sqlalchemy import Column

ENV = os.getenv("ENV", "dev")

if ENV == "prod":
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
else:
    app = FastAPI()

# CORS setup
origins = [
    "http://localhost:3231",
    "https://b2g-hackathon.netlify.app"
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
redis_client = None

async def init_redis_client():
    """Initialize async Redis client"""
    global redis_client
    try:
        # Build Redis URL properly
        if REDIS_PASSWORD:
            redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        else:
            redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        
        redis_client = await aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        # Test connection
        await redis_client.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Warning: Redis connection failed: {e}")
        print("Falling back to in-memory storage (not suitable for production)")
        redis_client = None

# Initialize Redis on startup
@app.on_event("startup")
async def startup_event():
    await init_redis_client()

# Cleanup Redis connection on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()
        print("Redis connection closed")

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

class EvaluationSubmission(BaseModel):
    """Evaluation submission model for problem and solution"""
    problem: str
    solution: str

def get_round_word_requirements(round: int) -> tuple[int, int, int, int]:
    """Get word count requirements for problem and solution based on round"""
    if round == 1:
        return ROUND_1_PROBLEM_MIN_WORDS, ROUND_1_PROBLEM_MAX_WORDS, ROUND_1_SOLUTION_MIN_WORDS, ROUND_1_SOLUTION_MAX_WORDS
    elif round == 2:
        return ROUND_2_PROBLEM_MIN_WORDS, ROUND_2_PROBLEM_MAX_WORDS, ROUND_2_SOLUTION_MIN_WORDS, ROUND_2_SOLUTION_MAX_WORDS
    elif round == 3:
        return ROUND_3_PROBLEM_MIN_WORDS, ROUND_3_PROBLEM_MAX_WORDS, ROUND_3_SOLUTION_MIN_WORDS, ROUND_3_SOLUTION_MAX_WORDS
    elif round == 4:
        return ROUND_4_PROBLEM_MIN_WORDS, ROUND_4_PROBLEM_MAX_WORDS, ROUND_4_SOLUTION_MIN_WORDS, ROUND_4_SOLUTION_MAX_WORDS
    else:
        return ROUND_1_PROBLEM_MIN_WORDS, ROUND_1_PROBLEM_MAX_WORDS, ROUND_1_SOLUTION_MIN_WORDS, ROUND_1_SOLUTION_MAX_WORDS  # Default fallback

def count_words(text: str) -> int:
    """Count words in text (simple implementation)"""
    if not text:
        return 0
    # Split by whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)

async def get_current_user(request: Request) -> Dict:
    """Dependency to get current user from token"""
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(
            status_code=401, 
            detail="Authentication expired!"
        )

    # Get user by token
    from agents.database import get_user_by_token_async
    user = await get_user_by_token_async(token)
    
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

# Note: Synchronous Redis functions removed - use async versions instead

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
        except aioredis.ConnectionError as e:
            print(f"Redis connection error getting user conversation: {e}")
        except aioredis.RedisError as e:
            print(f"Redis error getting user conversation: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error in user conversation: {e}")
        except Exception as e:
            print(f"Unexpected error getting user conversation from Redis: {e}")
    
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
        except aioredis.ConnectionError as e:
            print(f"Redis connection error saving user conversation: {e}")
        except aioredis.RedisError as e:
            print(f"Redis error saving user conversation: {e}")
        except json.JSONEncodeError as e:
            print(f"JSON encode error in user conversation: {e}")
        except Exception as e:
            print(f"Unexpected error saving user conversation to Redis: {e}")
    
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
        except aioredis.ConnectionError as e:
            print(f"Redis connection error getting user last request time: {e}")
        except aioredis.RedisError as e:
            print(f"Redis error getting user last request time: {e}")
        except ValueError as e:
            print(f"Invalid datetime format in user last request time: {e}")
        except Exception as e:
            print(f"Unexpected error getting user last request time from Redis: {e}")
    
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
        except aioredis.ConnectionError as e:
            print(f"Redis connection error saving user last request time: {e}")
        except aioredis.RedisError as e:
            print(f"Redis error saving user last request time: {e}")
        except Exception as e:
            print(f"Unexpected error saving user last request time to Redis: {e}")
    
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
        from agents.database import create_user_async
        result = await create_user_async(db_user_data)
        user_id, error_type = result
        
        if not user_id:
            # Provide specific error messages based on error type
            if error_type == "email_already_exists":
                raise HTTPException(
                    status_code=409, 
                    detail="An account with this email address already exists. Please use a different email or try logging in."
                )
            elif error_type == "token_already_exists":
                raise HTTPException(
                    status_code=500, 
                    detail="Registration failed due to a system error. Please try again."
                )
            elif error_type == "missing_required_fields":
                raise HTTPException(
                    status_code=400, 
                    detail="Registration failed: Please fill in all required fields."
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Registration failed due to a system error. Please try again later."
                )
        
        # Assign agent_type based on user ID for load balancing
        agent_type = 1 if int(str(user_id)[-1]) % 2 == 1 else 2
        
        # Update the user's agent_type in the database
        from agents.database import update_user_agent_type_async, create_evaluation_record_async
        update_success = await update_user_agent_type_async(user_id, agent_type)
        if not update_success:
            print(f"Warning: Failed to update agent_type for user {user_id}")
            
        # Create evaluation round data
        await create_evaluation_record_async(user_id, "", "", None, 1, EVALUATION_ROUND_1_TIME)
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
        print(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed due to an unexpected error. Please try again later.")

@app.post("/users/login")
async def login_user(login_data: UserLogin):
    """Login user with email and date of birth"""
    try:
        # Get user by email and DOB
        from agents.database import get_user_by_email_and_dob_async
        user = await get_user_by_email_and_dob_async(login_data.email, login_data.date_of_birth)
        
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
        print(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed due to a system error. Please try again later.")

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
    # Get evaluation data
    from agents.database import get_evaluation_data_async
    return await get_evaluation_data_async(user_id, round)

@app.post("/evaluation/time")
async def update_evaluation_time_remaining(
    current_user: Dict = Depends(get_current_user),
    round: Optional[int] = Query(None, alias="round", ge=1, le=4, description="Agent round (1, 2, 3 or 4)")
):
    """ Update evaluation time for a user for specific round mentioned """
    if not round:
        raise HTTPException(
                status_code=400, 
                detail="Invalid round parameter"
            )
    user_id = current_user["user_id"]
    return update_evaluation_round_time(user_id, round)

@app.post("/evaluation")
async def submit_evaluation(
    submission: EvaluationSubmission,
    current_user: Dict = Depends(get_current_user),
    round: int = Query(..., ge=1, le=4, description="Evaluation round (1, 2, 3, or 4)")
):
    """
    Submit problem and solution for evaluation round and get AI feedback
    
    This endpoint:
    1. Checks if evaluation record exists for user_id and round
    2. Checks if previous round is completed (except for round 1)
    3. Updates problem and solution columns
    4. Calls AI agent to generate feedback
    5. Saves AI feedback and returns response
    """
    user_id = current_user["user_id"]
    
    try:
        # Check if previous round is completed (except for round 1)
        if round > 1:
            previous_completed = await check_previous_round_completed_async(user_id, round)
            if not previous_completed:
                raise HTTPException(
                    status_code=400,
                    detail=f"Previous round {round - 1} must be completed before starting round {round}"
                )
        
        # Check if evaluation record exists for this user and round
        evaluation_record = await get_evaluation_record_by_round_async(user_id, round)
        if not evaluation_record:
            raise HTTPException(
                status_code=400,
                detail=f"No evaluation record found for user {user_id} and round {round}"
            )
        
        # Get word count requirements for this round
        problem_min_words, problem_max_words, solution_min_words, solution_max_words = get_round_word_requirements(round)
        
        # Validate input data and word count
        if not submission.problem.strip():
            raise HTTPException(
                status_code=400,
                detail="Problem description cannot be empty"
            )
        
        if not submission.solution.strip():
            raise HTTPException(
                status_code=400,
                detail="Solution description cannot be empty"
            )
        
        # Check word count requirements
        problem_word_count = count_words(submission.problem)
        solution_word_count = count_words(submission.solution)
        
        if problem_word_count < problem_min_words:
            raise HTTPException(
                status_code=400,
                detail=f"Problem description must be at least {problem_min_words} words. Current: {problem_word_count} words"
            )
        
        if problem_word_count > problem_max_words:
            raise HTTPException(
                status_code=400,
                detail=f"Problem description must not exceed {problem_max_words} words. Current: {problem_word_count} words"
            )
        
        if solution_word_count < solution_min_words:
            raise HTTPException(
                status_code=400,
                detail=f"Solution description must be at least {solution_min_words} words. Current: {solution_word_count} words"
            )
        
        if solution_word_count > solution_max_words:
            raise HTTPException(
                status_code=400,
                detail=f"Solution description must not exceed {solution_max_words} words. Current: {solution_word_count} words"
            )
        
<<<<<<< Updated upstream
        # For round 4 (final first thought), don't generate AI feedback
        if round == 4:
            ai_feedback = None
            update_success = await update_evaluation_record_async(
                user_id, 
                round, 
                submission.problem, 
                submission.solution, 
                ai_feedback
            )
        else:
            # Get user's assigned agent_type for AI feedback
            agent_type = current_user.get("agent_type", 1)
            
            # Select agent based on user's agent_type
            assigned_agent = agent1 if agent_type == 1 else agent2
            
            # Prepare prompt for AI agent with round-specific requirements
            ai_prompt = f"""Please provide feedback on the following problem and solution for Round {round}:
=======
        # Get user's assigned agent_type for AI feedback
        from agents.database import get_user_profile_async
        user_profile = await get_user_profile_async(user_id)
        agent_type = user_profile.get("agent_type", 1) if user_profile else 1
        
        # Select agent based on user's agent_type
        assigned_agent = agent1 if agent_type == 1 else agent2
        
        # Prepare prompt for AI agent with round-specific requirements
        ai_prompt = f"""Please provide feedback on the following problem and solution for Round {round}:
>>>>>>> Stashed changes

Problem: {submission.problem}

Solution: {submission.solution}

Please provide constructive feedback focusing on:
1. Clarity and feasibility of the solution
2. Potential improvements or alternatives
3. Any concerns or considerations
4. Overall assessment of the idea

Round {round} Requirements:
- Problem: {problem_min_words} ≤ x ≤ {problem_max_words} words (Current: {problem_word_count} words)
- Solution: {solution_min_words} ≤ x ≤ {solution_max_words} words (Current: {solution_word_count} words)

<<<<<<< Updated upstream
Please provide a comprehensive but concise response (200-500 words)."""
            
            # Call AI agent to generate feedback
            ai_feedback = await run_agent(assigned_agent, user_id, ai_prompt, [], "o3", "eval")
            
            # Update evaluation record with problem, solution, and AI feedback
            update_success = await update_evaluation_record_async(
                user_id, 
                round, 
                submission.problem, 
                submission.solution, 
                ai_feedback
            )
=======
Please provide a comprehensive but concise response (200-400 words)."""
        
        # Call AI agent to generate feedback
        ai_feedback = await run_agent(assigned_agent, user_id, ai_prompt, [], "o3", "eval")
        
        # Update evaluation record with problem, solution, and AI feedback
        update_success = await update_evaluation_record_async(
            user_id, 
            round, 
            submission.problem, 
            submission.solution, 
            ai_feedback
        )
>>>>>>> Stashed changes
        
        if not update_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update evaluation record"
            )
        
<<<<<<< Updated upstream
        # For round 4, don't return AI feedback in response
        if round == 4:
            return {
                "message": "Final evaluation submitted successfully",
                "round": round,
            }
        else:
            return {
                "message": "Evaluation submitted successfully",
                "ai_feedback": ai_feedback,
                "round": round,
            }
=======
        return {
            "message": "Evaluation submitted successfully",
            "ai_feedback": ai_feedback,
            "round": round,
            "user_id": user_id
        }
>>>>>>> Stashed changes
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in submit_evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing evaluation"
        )

@app.post("/evaluation/complete")
async def complete_evaluation_round(
    current_user: Dict = Depends(get_current_user),
    round: int = Query(..., ge=1, le=4, description="Evaluation round to complete (1, 2, 3, or 4)")
):
    """
    Complete an evaluation round and create next round record
    
    This endpoint:
    1. Checks if evaluation record exists for user_id and round
    2. Checks if previous round is completed (except for round 1)
    3. Sets completed_at timestamp for current round
    4. Creates new record for next round (if not round 4)
    5. Returns success response
    """
    user_id = current_user["user_id"]
    
    try:
        # Check if previous round is completed (except for round 1)
        if round > 1:
            previous_completed = await check_previous_round_completed_async(user_id, round)
            if not previous_completed:
                raise HTTPException(
                    status_code=400,
                    detail=f"Previous round {round - 1} must be completed before completing round {round}"
                )
        
        # Check if evaluation record exists for this user and round
        evaluation_record = await get_evaluation_record_by_round_async(user_id, round)
        if not evaluation_record:
            raise HTTPException(
                status_code=400,
                detail=f"No evaluation record found for user {user_id} and round {round}"
            )
        
<<<<<<< Updated upstream
=======
        # Get word count requirements for this round
        problem_min_words, problem_max_words, solution_min_words, solution_max_words = get_round_word_requirements(round)
        
        # Check if current round has been submitted (has problem and solution)
        if not evaluation_record.problem.strip() or not evaluation_record.solution.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Round {round} must have problem and solution submitted before it can be completed"
            )
        
        # Check word count requirements for completion
        problem_word_count = count_words(evaluation_record.problem)
        solution_word_count = count_words(evaluation_record.solution)
        
        if problem_word_count < problem_min_words:
            raise HTTPException(
                status_code=400,
                detail=f"Round {round} problem must be at least {problem_min_words} words. Current: {problem_word_count} words"
            )
        
        if problem_word_count > problem_max_words:
            raise HTTPException(
                status_code=400,
                detail=f"Round {round} problem must not exceed {problem_max_words} words. Current: {problem_word_count} words"
            )
        
        if solution_word_count < solution_min_words:
            raise HTTPException(
                status_code=400,
                detail=f"Round {round} solution must be at least {solution_min_words} words. Current: {solution_word_count} words"
            )
        
        if solution_word_count > solution_max_words:
            raise HTTPException(
                status_code=400,
                detail=f"Round {round} solution must not exceed {solution_max_words} words. Current: {solution_word_count} words"
            )
        
>>>>>>> Stashed changes
        # Complete the round
        completion_success = await complete_evaluation_round_async(user_id, round)
        
        if not completion_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to complete evaluation round"
            )
        
<<<<<<< Updated upstream
        # Trigger memory creation for completed evaluation round
        try:
            # Get user's agent type for memory creation
            user_profile = current_user
            agent_type = user_profile.get("agent_type", 1) if user_profile else 1
            
            # Check memory triggers for eval mode (round completion)
            should_create_memory, triggers = await memory_system.check_memory_trigger(
                user_id, 
                "eval", 
                round_completed=True
            )
            
            # Create memory asynchronously if triggers are met
            if should_create_memory:
                await memory_system.create_memory_async(user_id, "eval", triggers, agent_type)
                print(f"Evaluation memory created for user {user_id}, round {round} with triggers: {triggers}")
                
        except Exception as e:
            print(f"Error in evaluation memory creation: {e}")
            # Continue execution even if memory creation fails
        
=======
>>>>>>> Stashed changes
        # Prepare response message
        if round == 4:
            message = f"Round {round} completed successfully. This was the final round."
        else:
            message = f"Round {round} completed successfully. Round {round + 1} is now available."
        
        return {
            "message": message,
            "round": round,
<<<<<<< Updated upstream
=======
            "user_id": user_id,
>>>>>>> Stashed changes
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in complete_evaluation_round: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while completing evaluation round"
        )

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
        user_profile = current_user
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
    memory_context = await memory_system.create_memory_context(
        user_id, user_msg, top_k=3, time_weight_factor=0.1
    )
    
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
    
    # Check memory triggers and create memory if needed
    try:
        # Get message count for this user in current mode
        from agents.database import get_user_message_count_async
        message_count = await get_user_message_count_async(user_id, input.mode)
        
        # Check user inactivity status
        last_activity = await get_user_last_request_time_async(user_id)
        inactivity_detected = False
        if last_activity:
            time_diff = (datetime.utcnow() - last_activity).total_seconds()
            inactivity_detected = time_diff > memory_system.memory_manager.inactivity_threshold
        
        # For eval mode, check current round completion status
        round_completed = False
        if input.mode == "eval":
            from agents.database import get_evaluation_data_async
            # Get all evaluation records for the user
            eval_data = await get_evaluation_data_async(user_id)
            if eval_data:
                # Find the current active round (the one that's not completed)
                current_round_record = None
                for record in eval_data:
                    if record.completed_at is None:
                        current_round_record = record
                        break
                
                # If no active round found, check if the latest round is completed
                if current_round_record is None and eval_data:
                    latest_record = max(eval_data, key=lambda x: x.round)
                    round_completed = latest_record.completed_at is not None
        
        # Check memory triggers
        should_create_memory, triggers = await memory_system.check_memory_trigger(
            user_id, 
            input.mode, 
            message_count=message_count,
            inactivity_detected=inactivity_detected,
            round_completed=round_completed
        )
        
        # Create memory asynchronously if triggers are met
        if should_create_memory:
            await memory_system.create_memory_async(user_id, input.mode, triggers, agent_type)
            print(f"Memory created for user {user_id} with triggers: {triggers}")
            
    except Exception as e:
        print(f"Error in memory trigger check: {e}")
        # Continue execution even if memory creation fails
    
    # Update immediate history for Redis
    history.append(("Assistant", agent_reply))
    await save_user_conversation_async(user_id, history)

    return [
        {
            "content": agent_reply,
            "message_type": "agent_response",
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

async def get_interview_session(user_id: int) -> Dict:
    """Get interview session for user"""
    if redis_client:
        try:
            session_key = get_interview_session_key(user_id)
            session_data = await redis_client.get(session_key)
            if session_data:
                return json.loads(session_data)
        except aioredis.ConnectionError as e:
            print(f"Redis connection error getting interview session: {e}")
        except aioredis.RedisError as e:
            print(f"Redis error getting interview session: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error in interview session: {e}")
        except Exception as e:
            print(f"Unexpected error getting interview session from Redis: {e}")
    
    return interview_sessions.get(user_id, {})

async def save_interview_session(user_id: int, session_data: Dict):
    """Save interview session for user"""
    if redis_client:
        try:
            session_key = get_interview_session_key(user_id)
            await redis_client.setex(session_key, 3600, json.dumps(session_data))  # 1 hour expiration
            return
        except aioredis.ConnectionError as e:
            print(f"Redis connection error saving interview session: {e}")
        except aioredis.RedisError as e:
            print(f"Redis error saving interview session: {e}")
        except json.JSONEncodeError as e:
            print(f"JSON encode error in interview session: {e}")
        except Exception as e:
            print(f"Unexpected error saving interview session to Redis: {e}")
    
    interview_sessions[user_id] = session_data

async def get_interview_chat_history(user_id: int) -> List[Dict]:
    """Get interview chat history for user"""
    if redis_client:
        try:
            history_key = get_interview_history_key(user_id)
            history_data = await redis_client.get(history_key)
            if history_data:
                return json.loads(history_data)
        except aioredis.ConnectionError as e:
            print(f"Redis connection error getting interview history: {e}")
        except aioredis.RedisError as e:
            print(f"Redis error getting interview history: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error in interview history: {e}")
        except Exception as e:
            print(f"Unexpected error getting interview history from Redis: {e}")
    
    return interview_chat_history.get(user_id, [])

async def save_interview_chat_history(user_id: int, history: List[Dict]):
    """Save interview chat history for user"""
    if redis_client:
        try:
            history_key = get_interview_history_key(user_id)
            await redis_client.setex(history_key, 86400, json.dumps(history))  # 24 hour expiration
            return
        except aioredis.ConnectionError as e:
            print(f"Redis connection error saving interview history: {e}")
        except aioredis.RedisError as e:
            print(f"Redis error saving interview history: {e}")
        except json.JSONEncodeError as e:
            print(f"JSON encode error in interview history: {e}")
        except Exception as e:
            print(f"Unexpected error saving interview history to Redis: {e}")
    
    interview_chat_history[user_id] = history

async def add_interview_message(user_id: int, role: str, content: str):
    """Add a message to interview chat history"""
    history = await get_interview_chat_history(user_id)
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    history.append(message)
    await save_interview_chat_history(user_id, history)

@app.post("/interview/init")
async def initialize_interview(request: InterviewInit, current_user: Dict = Depends(get_current_user)):
    """Initialize a chat session for an interview"""
    try:
        user_id = request.user_id
        
        # Verify the user_id matches the authenticated user
        if user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="User ID mismatch")
        
        # Check if user exists
        from agents.database import get_user_profile_async
        user_profile = await get_user_profile_async(user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Initialize interview session
        session_data = {
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        await save_interview_session(user_id, session_data)
        
        # Create initial interview question
        initial_question = "Hello! I'm here to conduct an interview with you about your experience with our AI agents. Let's start with a simple question: How would you rate your overall experience with the AI agents you interacted with today? Please rate it on a scale of 1-10, where 1 is very poor and 10 is excellent."
        
        # Add initial question to chat history
        await add_interview_message(user_id, "assistant", initial_question)
        
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
        session = await get_interview_session(user_id)
        if not session or session.get("status") != "active":
            raise HTTPException(status_code=400, detail="No active interview session found. Please initialize the interview first.")
        
        # Add user message to history
        await add_interview_message(user_id, "user", user_message)
        
        # Get chat history for context
        history = await get_interview_chat_history(user_id)
        
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
            await save_interview_session(user_id, session)
        
        # Add assistant response to history
        await add_interview_message(user_id, "assistant", response)
        
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
        chat_history = await get_interview_chat_history(user_id)
        
        if not chat_history:
            raise HTTPException(status_code=404, detail="No interview history found for this user")
        
        return {
            "chat_history": chat_history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interview history: {str(e)}")