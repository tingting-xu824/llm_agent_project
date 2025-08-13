# AI Agent System with RAG and Database Integration

A comprehensive AI agent system with Retrieval-Augmented Generation (RAG), PostgreSQL database integration, and token-based authentication.

## Features

- **Multiple AI Agents**: Originality Checker, Creative Idea Generator, and Chatbot Agent
- **RAG Memory System**: Stores and retrieves conversation context using vector similarity
- **Token-based Authentication**: Secure user authentication with Bearer tokens
- **PostgreSQL Database**: Neon cloud database integration with SQLAlchemy ORM
- **Redis Caching**: Distributed caching for conversation history and rate limiting
- **User Management**: Registration and login with comprehensive user profiles
- **Conversation History**: Persistent storage of chat messages
- **Memory Vectors**: Vector-based memory storage for RAG functionality
- **Scalable Architecture**: Designed for multi-server deployment

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Neon          │
│   (Client)      │◄──►│   Backend       │◄──►│   PostgreSQL    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Redis Cache   │
                       │   (Optional)    │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (GPT Models)  │
                       └─────────────────┘
```

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tingting-xu824/llm_agent_project.git
   cd llm_agent_project/ai_agent
   ```

2. **Set up environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file with:
   ```bash
   DATABASE_URL=your_neon_postgresql_connection_string
   OPENAI_API_KEY=your_openai_api_key
   REDIS_HOST=localhost  # Optional
   REDIS_PORT=6379       # Optional
   ```

4. **Start the server**:
   ```bash
   uvicorn agents.api:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

## API Endpoints

### User Management
- `POST /users/register` - Register new user
- `POST /users/login` - User login
- `GET /users/profile` - Get user profile

### Agent Interaction
- `POST /agent` - Send message to AI agent
- `GET /conversations` - Get conversation history

## Database Schema

### User Table
- `user_id` (Primary Key)
- `email` (Unique)
- `dob` (Date of Birth)
- `gender`, `education_field`, `education_level`
- `disability_knowledge`, `genai_course_exp`
- `token` (Authentication token)
- `registration_time`

### Conversation Table
- `conversation_id` (Primary Key)
- `user_id` (Foreign Key)
- `message_type` (USER/ASSISTANT)
- `content` (Message text)
- `timestamp`, `character_count`, `sequence_number`
- `role`, `mode`, `agent_type`

### Memory Vectors Table
- `memory_id` (Primary Key)
- `user_id` (Foreign Key)
- `memory_type`, `source_conversations`
- `memory_content`, `content`
- `embedding` (JSON vector)
- `created_at`, `memory_metadata`

## Testing

Run the comprehensive test suite:

```bash
# Test basic API functionality
python3 test_api.py

# Test RAG system
python3 test_rag_system.py

# Test eval mode
python3 test_eval.py
```

## Configuration

### Environment Variables
- `DATABASE_URL`: Neon PostgreSQL connection string
- `OPENAI_API_KEY`: Your OpenAI API key
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`: Redis configuration (optional)

### Agent Types
- `1`: Originality Checker
- `2`: Creative Idea Generator  
- `3`: Chatbot Agent

### Modes
- `chat`: Interactive conversation mode
- `eval`: Evaluation mode (uses OpenAI "o3" model)

## Security Features

- Token-based authentication with Bearer tokens
- Rate limiting with Redis
- Input validation with Pydantic models
- SQL injection protection with SQLAlchemy ORM
- Environment variable protection

## Scalability Features

- Redis-based distributed caching
- Connection pooling with SQLAlchemy
- Stateless API design
- Horizontal scaling support

## Development

### Project Structure
```
ai_agent/
├── agents/
│   ├── api.py              # Main FastAPI application
│   ├── database.py         # Database models and operations
│   ├── memory_system.py    # RAG memory system
│   ├── agents_backend.py   # Agent definitions
│   ├── runner.py           # OpenAI API integration
│   ├── agent.py            # Base agent class
│   └── tools.py            # Agent tools
├── instructions/           # Agent prompt templates
├── test_*.py              # Test scripts
├── requirements.txt       # Python dependencies
├── API_README.md          # API documentation
├── SETUP.md              # Setup instructions
└── README.md             # This file
```

### Adding New Agents
1. Define agent in `agents/agents_backend.py`
2. Add prompt template in `instructions/`
3. Update API endpoints in `agents/api.py`

## License

This project is part of a Master's thesis research project.

## Support

For issues and questions, please refer to the API documentation or create an issue in the repository.
