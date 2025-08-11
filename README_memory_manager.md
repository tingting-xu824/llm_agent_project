# MemoryManager Class Documentation

## 📋 Overview

`MemoryManager` is an intelligent memory management system designed specifically for hackathon competition assistance systems. It uses vector databases to store and retrieve user conversation memories, supporting personalized AI services.

## 🏗️ System Architecture

### Core Components
- **Vectorized Storage**: Uses PostgreSQL + pgvector extension to store memory vectors
- **Intelligent Retrieval**: Memory retrieval based on semantic similarity
- **Asynchronous Processing**: Non-blocking memory creation and updates
- **Caching Mechanism**: Embedding vector cache for performance optimization

### Memory Types
- **eval phase**: `round_summary` - Records idea evolution process
- **chat phase**: `conversation_chunk` - Records technical implementation details

## 🔧 Initialization

```python
from memory_manager import MemoryManager

# Initialize MemoryManager
memory_manager = MemoryManager(
    database_url="postgresql://user:pass@localhost/db",
    embedding_generator=your_embedding_function,  # External embedding generation function
    memory_extractor=your_memory_extractor_function,  # External memory extraction function
    pool_size=20  # Database connection pool size
)
```

### Parameter Description
- `database_url`: PostgreSQL connection string
- `embedding_generator`: Text vectorization function (Callable)
- `memory_extractor`: Memory extraction function (Callable)
- `pool_size`: Database connection pool size (default: 20)

## 📚 Core Methods

### 1. Memory Creation Flow

#### `check_memory_trigger(user_id, mode, **kwargs)`
Check if memory should be created
```python
# eval phase: Trigger after each round of conversation
should_create, triggers = await memory_manager.check_memory_trigger(
    user_id=123,
    mode="eval",
    round_completed=True
)

# chat phase: Multiple trigger conditions
should_create, triggers = await memory_manager.check_memory_trigger(
    user_id=123,
    mode="chat",
    message_count=15,
    important_content=True,
    session_closing=False
)
```

#### `create_memory_async(user_id, mode, triggers, agent_type)`
Asynchronously create memory (non-blocking response)
```python
await memory_manager.create_memory_async(
    user_id=123,
    mode="chat",
    triggers=["message_count", "important_content"],
    agent_type=1
)
```

#### `create_memory_from_conversations(user_id, memory_type, conversation_ids, agent_type)`
Create complete memory from conversation records
```python
memory_id = await memory_manager.create_memory_from_conversations(
    user_id=123,
    memory_type="chat",
    conversation_ids=[101, 102, 103],  # Optional: specify conversation IDs
    agent_type=1
)
```

### 2. Memory Retrieval Flow

#### `retrieve_relevant_memories(user_id, query_text, top_k)`
Retrieve relevant memories based on semantic similarity
```python
relevant_memories = await memory_manager.retrieve_relevant_memories(
    user_id=123,
    query_text="How to implement user login functionality?",
    top_k=3
)
```

#### `get_recent_memories(user_id, limit)`
Get user's recent memories
```python
recent_memories = await memory_manager.get_recent_memories(
    user_id=123,
    limit=5
)
```

#### `build_memory_context(memories)`
Build memory context string
```python
context = await memory_manager.build_memory_context(relevant_memories)
# Output format:
# Memory 1 (chat): Participant uses Flask-Login for user authentication
# Memory 2 (eval): Participant plans to develop project-guided programming learning chatbot
```

### 3. Conversation Management

#### `get_conversations_by_user(user_id, limit)`
Get user's recent conversation records
```python
conversations = await memory_manager.get_conversations_by_user(
    user_id=123,
    limit=20
)
```

#### `get_conversations_by_ids(conversation_ids)`
Get specific conversations by IDs
```python
conversations = await memory_manager.get_conversations_by_ids([101, 102, 103])
```

### 4. Utility Methods

#### `get_user_memory_count(user_id)`
Get total number of user memories
```python
count = await memory_manager.get_user_memory_count(user_id=123)
```

#### `detect_important_content(message)`
Detect if message contains important content
```python
is_important = memory_manager.detect_important_content("This is an important decision")
```

## 🗄️ Database Structure

### memory_vectors table
```sql
CREATE TABLE memory_vectors (
    memory_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    memory_type VARCHAR(10) NOT NULL,  -- 'eval' or 'chat'
    source_conversations TEXT NOT NULL, -- Conversation ID array JSON
    memory_content TEXT NOT NULL,       -- Extracted memory content
    embedding VECTOR(1536) NOT NULL,    -- Vector embedding
    _metadata JSONB,                     -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔄 Complete Usage Flow

### 1. Memory Creation Flow
```python
# 1. Check trigger conditions
should_create, triggers = await memory_manager.check_memory_trigger(
    user_id=123,
    mode="chat",
    message_count=10,
    important_content=True
)

# 2. Asynchronously create memory
if should_create:
    await memory_manager.create_memory_async(
        user_id=123,
        mode="chat",
        triggers=triggers
    )
```

### 2. Memory Retrieval and Usage Flow
```python
# 1. Retrieve relevant memories
relevant_memories = await memory_manager.retrieve_relevant_memories(
    user_id=123,
    query_text=user_query,
    top_k=3
)

# 2. Build context
memory_context = await memory_manager.build_memory_context(relevant_memories)

# 3. Provide to AI model
enhanced_prompt = f"""
Answer user questions based on the following historical memories:

{memory_context}

User question: {user_query}
"""

response = await ai_model.generate(enhanced_prompt)
```

## ⚙️ Configuration Parameters

### Memory Trigger Thresholds
```python
self.chat_message_threshold = 10      # Chat message threshold
self.inactivity_threshold = 3600      # Inactivity threshold (seconds)
```

### Important Keywords
```python
self.important_keywords = [
    'important', 'decision', 'problem', 'solution', 'change', 'modify',
    'key', 'critical', 'issue', 'update', 'final', 'conclusion'
]
```

### Cache Settings
```python
self.cache_size_limit = 1000          # Embedding cache size limit
```

## 🚀 Performance Optimization

### 1. Database Indexes
```sql
-- User ID index
CREATE INDEX idx_memory_vectors_user_id ON memory_vectors(user_id);

-- Vector similarity search index
CREATE INDEX ON memory_vectors USING ivfflat (embedding vector_cosine_ops);

-- Time index
CREATE INDEX idx_memory_vectors_created_at ON memory_vectors(created_at);
```

### 2. Connection Pool Configuration
```python
# Connection pool parameters
minconn=5,           # Minimum connections
maxconn=20,          # Maximum connections
connect_timeout=10,  # Connection timeout
```

## 🛠️ Error Handling

### Exception Types
- **Database connection errors**: Automatic retry and rollback
- **Vector generation errors**: Log and return default values
- **Memory extraction errors**: Return error messages instead of crashing

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# Different log levels
logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning information")
logger.error("Error information")
```

## 🔧 Extension and Customization

### 1. Custom Memory Types
```python
# Add new memory type mapping in create_memory_async
memory_type = "custom_type" if mode == "custom" else "conversation_chunk"
```

### 2. Custom Trigger Conditions
```python
# Add new trigger logic in _check_chat_trigger
if custom_condition:
    triggers.append("custom_trigger")
```

### 3. Custom Memory Extraction
```python
# Implement custom memory_extractor function
async def custom_memory_extractor(conversation_text, memory_type):
    # Custom memory extraction logic
    return extracted_memory_content
```

## 📊 Monitoring and Debugging

### 1. Memory Statistics
```python
# Get user memory count
count = await memory_manager.get_user_memory_count(user_id)

# Get recent memories
recent = await memory_manager.get_recent_memories(user_id, limit=5)
```

### 2. Performance Monitoring
```python
# Check cache hit rate
cache_hit_rate = len(memory_manager.embedding_cache) / total_requests

# Monitor database connection pool status
pool_status = memory_manager.connection_pool.get_stats()
```

## 🔒 Security Considerations

### 1. User Isolation
- All queries include `user_id` conditions
- Ensure users can only access their own memories

### 2. SQL Injection Protection
- Use parameterized queries
- Avoid string concatenation

### 3. Data Validation
- Input parameter type checking
- Vector dimension validation

## 📝 Important Notes

1. **Asynchronous Operations**: Most methods are asynchronous, requiring `await`
2. **Resource Management**: Call `close()` method to clean up resources after use
3. **Error Handling**: All methods have exception handling and won't cause system crashes
4. **Performance Considerations**: Consider pagination and caching strategies for large memory volumes
