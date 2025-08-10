# Setup Guide

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@host:port/database?sslmode=require

# Redis Configuration (optional, for caching)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in `.env` file

4. Start the server:
```bash
uvicorn agents.api:app --reload --host 0.0.0.0 --port 8000
```

## Database Setup

The system uses Neon PostgreSQL. Make sure your database has the following tables:
- `user` - for user management
- `conversation` - for chat history
- `memory_vectors` - for RAG memory storage

See the database schema in `agents/database.py` for details.
