from sqlalchemy import create_engine, update, Column, Integer, String, Date, DateTime, Text, TIMESTAMP, BigInteger, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import os
from typing import Optional, Dict, List, Any
from datetime import date
from agents.constants import EVALUATION_ROUND_TIMINGS

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not defined in .env")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Import json for database operations
import json

class User(Base):
    """User model for database operations"""
    __tablename__ = "user"
    
    user_id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    dob = Column(Date, nullable=False)
    agent_type = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    education_field = Column(String)
    education_level = Column(String)
    disability_knowledge = Column(String, nullable=False)
    genai_course_exp = Column(String, nullable=False)
    token = Column(String, unique=True, index=True)
    registration_time = Column(DateTime, default=func.now())

class IdeaEvaluation(Base):
    __tablename__ = "idea_evaluation"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    problem = Column(Text, nullable=False)
    solution = Column(Text, nullable=False)
    ai_feedback = Column(Text)
    round = Column(Integer, nullable=False)  # Matching your 'idea_evaluation_rounds'
    created_at = Column(
        TIMESTAMP, 
        server_default=func.current_timestamp(),
        nullable=False
    )
    time_remaining = Column(BigInteger)
    completed_at=Column(TIMESTAMP)

    __table_args__ = (
        UniqueConstraint("user_id", "round", name="uq_user_round"),
    )

    def __repr__(self):
        return f"<IdeaEvaluation(id={self.id}, user_id={self.user_id}, round={self.round})>"
    
class Conversation(Base):
    """Conversation model for storing chat messages"""
    __tablename__ = "conversation"
    
    conversation_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    message_type = Column(String, nullable=False)  # "user_input" or "agent_response" (existing)
    content = Column(Text, nullable=False)  # The actual message content
    timestamp = Column(DateTime, default=func.now())
    character_count = Column(Integer)
    sequence_number = Column(Integer)
    # New columns (will be added by SQL script)
    role = Column(String)  # "user" or "assistant" (new)
    mode = Column(String)  # "chat" or "eval" (new)
    agent_type = Column(Integer)  # Which agent was used (new)

class MemoryVector(Base):
    """Memory vector model for RAG functionality"""
    __tablename__ = "memory_vectors"
    
    memory_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    memory_type = Column(String(50), nullable=False)  # Type of memory with constraint
    source_conversations = Column(Text)  # Source conversation IDs
    memory_content = Column(Text, nullable=False)  # The text content
    embedding = Column(Text, nullable=False)  # VECTOR(1536) type for pgvector extension
    created_at = Column(DateTime, default=func.now())
    _metadata = Column(Text)  # JSONB metadata

class DatabaseManager:
    """Database manager for user operations"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    def get_user_by_token(self, token: str) -> Optional[Dict]:
        """Get user by token using ORM"""
        try:
            user = self.db.query(User).filter(User.token == token).first()
            if user:
                return {
                    "user_id": user.user_id,
                    "email": user.email,
                    "dob": user.dob,
                    "gender": user.gender,
                    "education_field": user.education_field,
                    "education_level": user.education_level,
                    "agent_type": user.agent_type,
                    "disability_knowledge": user.disability_knowledge,
                    "genai_course_exp": user.genai_course_exp,
                    "token": user.token,
                    "registration_time": user.registration_time
                }
            return None
        except Exception as e:
            print(f"Error getting user by token: {e}")
            return None
    
    def get_user_by_email_and_dob(self, email: str, date_of_birth: date) -> Optional[Dict]:
        """Get user by email and date of birth using ORM"""
        try:
            user = self.db.query(User).filter(
                User.email == email,
                User.dob == date_of_birth
            ).first()
            
            if user:
                return {
                    "user_id": user.user_id,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email,
                    "dob": user.dob,
                    "agent_type": user.agent_type,
                    "gender": user.gender,
                    "education_field": user.education_field,
                    "education_level": user.education_level,
                    "disability_knowledge": user.disability_knowledge,
                    "genai_course_exp": user.genai_course_exp,
                    "token": str(user.token),
                    "registration_time": user.registration_time
                }
            return None
        except Exception as e:
            print(f"Error getting user by email and DOB: {e}")
            return None
    
    def create_user(self, user_data: Dict) -> tuple[Optional[Column[int]], Optional[str]]:
        """Create new user using ORM"""
        try:
            new_user = User(**user_data)
            self.db.add(new_user)
            self.db.commit()
            self.db.refresh(new_user)
            return new_user.user_id, None
        except Exception as e:
            self.db.rollback()
            print(f"Error creating user: {e}")
            
            # Check for specific database constraint violations
            error_str = str(e).lower()
            if "unique constraint" in error_str and "email" in error_str:
                return None, "email_already_exists"
            elif "unique constraint" in error_str and "token" in error_str:
                return None, "token_already_exists"
            elif "not null constraint" in error_str:
                return None, "missing_required_fields"
            else:
                return None, "database_error"
    
    def update_user_agent_type(self, user_id: int, agent_type: int) -> bool:
        """Update user's agent type using ORM"""
        try:
            user = self.db.query(User).filter(User.user_id == user_id).first()
            if user:
                user.agent_type = agent_type
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            print(f"Error updating user agent type: {e}")
            return False
    
    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get complete user profile using ORM"""
        try:
            user = self.db.query(User).filter(User.user_id == user_id).first()
            if user:
                return {
                    "user_id": user.user_id,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email,
                    "dob": user.dob,
                    "agent_type": user.agent_type,
                    "gender": user.gender,
                    "education_field": user.education_field,
                    "education_level": user.education_level,
                    "disability_knowledge": user.disability_knowledge,
                    "genai_course_exp": user.genai_course_exp,
                    "token": user.token,
                    "registration_time": user.registration_time
                }
            return None
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None
    
    def create_evaluation_record(self, user_id: int, problem: str, solution: str, ai_feedback: str | None, round: int, time_remaining: int) -> Optional[dict]:
        try:
            new_idea_eval = IdeaEvaluation(user_id = user_id, problem = problem, solution = solution, ai_feedback = ai_feedback, round = round, time_remaining = time_remaining)
            self.db.add(new_idea_eval)
            self.db.commit()
            self.db.refresh(new_idea_eval)
            return new_idea_eval
        except Exception as e:
            self.db.rollback()
            print(f"Error creating evaluation record: {e}")
            return None

    def update_evaluation_round_time(self, user_id: int, round_number: int):
        try:
            stmt = (
                update(IdeaEvaluation)
                .where(IdeaEvaluation.user_id == user_id, IdeaEvaluation.round == round_number)
                .values(
                    time_remaining=func.greatest(IdeaEvaluation.time_remaining - 10, 0)
                )
                .returning(IdeaEvaluation.time_remaining)  # <-- This is the key
            )

            # Execute and fetch the updated value in the same operation
            result = self.db.execute(stmt).scalar()

            if result is None:
                self.db.rollback()
                return None  # No matching row found

            self.db.commit()
            return result
        except Exception as e:
            self.db.rollback()
            print(f"Error updating evaluation remaining time: {e}")
            return False

    def save_conversation_message(self, user_id: int, message: str, role: str, mode: str, agent_type: int) -> Optional[Column[int]]:
        """Save a conversation message to database"""
        try:
            # Get the next sequence number for this user
            last_conversation = self.db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.sequence_number.desc()).first()
            
            next_sequence = 1 if not last_conversation else last_conversation.sequence_number + 1
            
            # Map role to message_type format
            message_type = "user_input" if role == "user" else "agent_response"
            
            conversation = Conversation(
                user_id=user_id,
                message_type=message_type,  # "USER" or "ASSISTANT"
                content=message,
                character_count=len(message),
                sequence_number=next_sequence,
                role=role,  # "user" or "assistant"
                mode=mode,  # "chat" or "eval"
                agent_type=agent_type
            )
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            return conversation.conversation_id
        except Exception as e:
            self.db.rollback()
            print(f"Error saving conversation message: {e}")
            return None
    
    def get_user_conversations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get user's conversation history from database"""
        try:
            conversations = self.db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.timestamp.asc()).limit(limit).all()
            
            return [
                {
                    "conversation_id": conv.conversation_id,
                    "content": conv.content,
                    "message_type": conv.message_type,
                    "timestamp": conv.timestamp,
                }
                for conv in conversations
            ]
        except Exception as e:
            print(f"Error getting user conversations: {e}")
            return []
    
    def save_memory_vector(self, user_id: int, content: str, embedding: List[float], metadata: Dict | None = None) -> Optional[Column[int]]:
        """Save a memory vector for RAG functionality"""
        try:
            memory_vector = MemoryVector(
                user_id=user_id,
                memory_type="conversation_chunk",  # Use proper memory_type values
                memory_content=content,
                embedding=json.dumps(embedding),
                _metadata=json.dumps(metadata) if metadata else None
            )
            self.db.add(memory_vector)
            self.db.commit()
            self.db.refresh(memory_vector)
            return memory_vector.memory_id
        except Exception as e:
            self.db.rollback()
            print(f"Error saving memory vector: {e}")
            return None
    
    def get_relevant_memories(self, user_id: int, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Get relevant memories using vector similarity search"""
        try:
            import json
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get all memory vectors for the user
            memory_vectors = self.db.query(MemoryVector).filter(
                MemoryVector.user_id == user_id
            ).all()
            
            if not memory_vectors:
                return []
            
            # Calculate similarities
            similarities = []
            for mv in memory_vectors:
                try:
                    embedding = json.loads(str(mv.embedding))
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    similarities.append((similarity, mv))
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_memories = similarities[:top_k]
            
            return [
                {
                    "memory_id": mv.memory_id,
                    "content": mv.memory_content,  # Use memory_content field
                    "metadata": json.loads(mv._metadata) if mv._metadata else {},
                    "similarity": float(similarity),
                    "created_at": mv.created_at
                }
                for similarity, mv in top_memories
            ]
        except Exception as e:
            print(f"Error getting relevant memories: {e}")
            return []

    def get_evaluation_data(self, user_id: int, round: Optional[int] = None):
        try:
            if round is None:
                eval_data = self.db.query(IdeaEvaluation).filter(IdeaEvaluation.user_id == user_id).order_by(IdeaEvaluation.round.asc()).all()
                return eval_data
            else:
                # Convert int round to string for database query
                eval_data = self.db.query(IdeaEvaluation).filter(IdeaEvaluation.user_id == user_id, IdeaEvaluation.round == round).first() # "weired" round check because all possible enum values are
                return eval_data

        except Exception as e:
            print(f"Error getting evaluation data: {e}")
            return []

    def get_evaluation_record_by_round(self, user_id: int, round: int) -> Optional[IdeaEvaluation]:
        """Get specific evaluation record for user and round"""
        try:
            record = self.db.query(IdeaEvaluation).filter(
                IdeaEvaluation.user_id == user_id,
                IdeaEvaluation.round == round
            ).first()
            return record
        except Exception as e:
            print(f"Error getting evaluation record by round: {e}")
            return None

    def update_evaluation_record(self, user_id: int, round: int, problem: str, solution: str, ai_feedback: str) -> bool:
        """Update problem, solution, and ai_feedback for a specific evaluation record"""
        try:
            updated_rows = (
                self.db.query(IdeaEvaluation)
                .filter_by(user_id=user_id, round=round)
                .update({
                    IdeaEvaluation.problem: problem,
                    IdeaEvaluation.solution: solution,
                    IdeaEvaluation.ai_feedback: ai_feedback
                })
            )
            if updated_rows == 0:
                self.db.rollback()
                return False
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Error updating evaluation record: {e}")
            return False

    def complete_evaluation_round(self, user_id: int, round: int) -> bool:
        """Mark evaluation round as completed and create next round record"""
        try:
            # First, mark current round as completed
            updated_rows = (
                self.db.query(IdeaEvaluation)
                .filter_by(user_id=user_id, round=round)
                .update({
                    IdeaEvaluation.completed_at: func.current_timestamp()
                })
            )
            if updated_rows == 0:
                self.db.rollback()
                return False
            
            # Create next round record if not round 4 (final submission)
            next_round = round + 1
            if next_round <= 4:
                new_record = IdeaEvaluation(
                    user_id=user_id,
                    problem="",
                    solution="",
                    ai_feedback=None,
                    round=next_round,
                    time_remaining= EVALUATION_ROUND_TIMINGS.get(next_round)
                )
                self.db.add(new_record)
            
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Error completing evaluation round: {e}")
            return False

    def check_previous_round_completed(self, user_id: int, round: int) -> bool:
        """Check if previous round has been completed"""
        try:
            if round == 1:
                return True  # First round doesn't need previous completion check
            
            previous_record = self.db.query(IdeaEvaluation).filter(
                IdeaEvaluation.user_id == user_id,
                IdeaEvaluation.round == round - 1
            ).first()
            
            if not previous_record:
                return False
            
            return previous_record.completed_at is not None
        except Exception as e:
            print(f"Error checking previous round completion: {e}")
            return False

# Convenience functions
def get_user_by_token(token: str) -> Optional[Dict]:
    """Get user by token"""
    with DatabaseManager() as db:
        return db.get_user_by_token(token)

def get_user_by_email_and_dob(email: str, date_of_birth: date) -> Optional[Dict]:
    """Get user by email and date of birth"""
    with DatabaseManager() as db:
        return db.get_user_by_email_and_dob(email, date_of_birth)

def create_user(user_data: Dict) -> tuple[Optional[Column[int]], Optional[str]]:
    """Create new user"""
    with DatabaseManager() as db:
        return db.create_user(user_data)

def update_user_agent_type(user_id: int, agent_type: int) -> bool:
    """Update user's agent type"""
    with DatabaseManager() as db:
        return db.update_user_agent_type(user_id, agent_type)

def get_user_profile(user_id: int) -> Optional[Dict]:
    """Get complete user profile"""
    with DatabaseManager() as db:
        return db.get_user_profile(user_id)

def save_conversation_message(user_id: int, message: str, role: str, mode: str, agent_type: int) -> Optional[Column[int]]:
    """Save a conversation message"""
    with DatabaseManager() as db:
        return db.save_conversation_message(user_id, message, role, mode, agent_type)

def get_user_conversations(user_id: int, limit: int = 50) -> List[Dict]:
    """Get user's conversation history"""
    with DatabaseManager() as db:
        return db.get_user_conversations(user_id, limit)

def save_memory_vector(user_id: int, content: str, embedding: List[float], metadata: Dict | None = None) -> Optional[Column[int]]:
    """Save a memory vector"""
    with DatabaseManager() as db:
        return db.save_memory_vector(user_id, content, embedding, metadata)

def get_relevant_memories(user_id: int, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """Get relevant memories"""
    with DatabaseManager() as db:
        return db.get_relevant_memories(user_id, query_embedding, top_k)

def get_evaluation_data(user_id: int, round: Optional[int] = None):
    with DatabaseManager() as db:
        return db.get_evaluation_data(user_id, round)
    
def create_evaluation_round_data(user_id: int, problem: str, solution: str, ai_feedback: str | None, round: int, time_remaining: int):
    with DatabaseManager() as db:
        return db.create_evaluation_record(user_id, problem, solution, ai_feedback, round, time_remaining)
    
def update_evaluation_round_time(user_id: int, round: int):
    with DatabaseManager() as db:
        return db.update_evaluation_round_time(user_id, round)

def get_evaluation_record_by_round(user_id: int, round: int):
    """Get specific evaluation record for user and round"""
    with DatabaseManager() as db:
        return db.get_evaluation_record_by_round(user_id, round)

def update_evaluation_record(user_id: int, round: int, problem: str, solution: str, ai_feedback: str) -> bool:
    """Update problem, solution, and ai_feedback for a specific evaluation record"""
    with DatabaseManager() as db:
        return db.update_evaluation_record(user_id, round, problem, solution, ai_feedback)

def complete_evaluation_round(user_id: int, round: int) -> bool:
    """Mark evaluation round as completed and create next round record"""
    with DatabaseManager() as db:
        return db.complete_evaluation_round(user_id, round)

def check_previous_round_completed(user_id: int, round: int) -> bool:
    """Check if previous round has been completed"""
    with DatabaseManager() as db:
        return db.check_previous_round_completed(user_id, round)
        
# Async database functions using thread pool for synchronous operations
# Note: These are not truly async I/O but provide async interface for compatibility

async def save_conversation_message_async(user_id: int, message: str, role: str, mode: str, agent_type: int) -> Optional[Column[int]]:
    """Save conversation message using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, save_conversation_message, user_id, message, role, mode, agent_type)

async def get_user_conversations_async(user_id: int, limit: int = 50) -> List[Dict]:
    """Get user conversations using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_conversations, user_id, limit)

async def get_user_profile_async(user_id: int) -> Optional[Dict]:
    """Get user profile using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_profile, user_id)

async def update_user_agent_type_async(user_id: int, agent_type: int) -> bool:
    """Update user's agent type using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, update_user_agent_type, user_id, agent_type)

async def create_evaluation_record_async(user_id: int, problem: str, solution: str, ai_feedback: str | None, round: int, time_remaining: int) -> Optional[dict]:
    """Create evaluation record using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_evaluation_record, user_id, problem, solution, ai_feedback, round, time_remaining)

async def get_evaluation_data_async(user_id: int, round: Optional[int] = None):
    """Get evaluation data using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_evaluation_data, user_id, round)

async def get_evaluation_record_by_round_async(user_id: int, round: int):
    """Get specific evaluation record for user and round using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_evaluation_record_by_round, user_id, round)

async def update_evaluation_record_async(user_id: int, round: int, problem: str, solution: str, ai_feedback: str) -> bool:
    """Update evaluation record using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, update_evaluation_record, user_id, round, problem, solution, ai_feedback)

async def complete_evaluation_round_async(user_id: int, round: int) -> bool:
    """Complete evaluation round using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, complete_evaluation_round, user_id, round)

async def check_previous_round_completed_async(user_id: int, round: int) -> bool:
    """Check previous round completion using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, check_previous_round_completed, user_id, round)

async def create_user_async(user_data: Dict) -> tuple[Optional[Column[int]], Optional[str]]:
    """Create user using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_user, user_data)

async def get_user_by_token_async(token: str) -> Optional[Dict]:
    """Get user by token using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_by_token, token)

async def get_user_by_email_and_dob_async(email: str, date_of_birth: date) -> Optional[Dict]:
    """Get user by email and DOB using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_by_email_and_dob, email, date_of_birth)

# Batch operations for true optimization (when multiple operations need to be executed together)
async def execute_db_operations_batch(operations: List[tuple]) -> List[Any]:
    """
    Execute multiple database operations in a single thread pool call
    This is only used when multiple operations need to be executed together
    
    Args:
        operations: List of (function, *args) tuples
        
    Returns:
        List of results from each operation
    """
    import asyncio
    loop = asyncio.get_event_loop()
    
    def execute_operations_sync():
        results = []
        with DatabaseManager() as db:
            for func, args in operations:
                try:
                    if func == 'save_conversation_message':
                        result = db.save_conversation_message(*args)
                    elif func == 'get_user_conversations':
                        result = db.get_user_conversations(*args)
                    elif func == 'get_user_profile':
                        result = db.get_user_profile(*args)
                    elif func == 'save_memory_vector':
                        result = db.save_memory_vector(*args)
                    elif func == 'update_user_agent_type':
                        result = db.update_user_agent_type(*args)
                    elif func == 'create_evaluation_record':
                        result = db.create_evaluation_record(*args)
                    elif func == 'get_evaluation_data':
                        result = db.get_evaluation_data(*args)
                    else:
                        result = None
                    results.append(result)
                except Exception as e:
                    print(f"Error executing operation {func}: {e}")
                    results.append(None)
        return results
    
    return await loop.run_in_executor(None, execute_operations_sync)

def get_user_message_count(user_id: int, mode: str) -> int:
    """Get total message count for a user in a specific mode"""
    with DatabaseManager() as db:
        try:
            if mode == "eval":
                # For eval mode, count evaluation records from idea_evaluation table
                result = db.db.query(IdeaEvaluation).filter(
                    IdeaEvaluation.user_id == user_id
                ).count()
            else:
                # For chat mode, count conversation messages
                result = db.db.query(Conversation).filter(
                    Conversation.user_id == user_id,
                    Conversation.mode == mode
                ).count()
            return result
        except Exception as e:
            print(f"Error getting message count for user {user_id}, mode {mode}: {e}")
            return 0

async def get_user_message_count_async(user_id: int, mode: str) -> int:
    """Get total message count for a user in a specific mode using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_message_count, user_id, mode)
