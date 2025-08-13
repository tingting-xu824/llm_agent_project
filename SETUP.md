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

# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
AZURE_STORAGE_ACCOUNT_NAME=your_azure_storage_account_name
AZURE_STORAGE_ACCOUNT_KEY=your_azure_storage_account_key
AZURE_STORAGE_CONTAINER=finalreports

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
- `final_report` - for final report submissions

See the database schema in `agents/database.py` for details.

## Azure Storage Setup

The system uses Azure Blob Storage for file uploads. You need to:

1. Create an Azure Storage account
2. Create a container for storing files
3. Get the connection string and account key
4. Add the configuration to your `.env` file

The files will be stored with the following structure:
- Container: `finalreports` (configurable)
- File path: `user_{user_id}/{unique_id}{extension}`
- Access: Public URLs with SAS tokens (1 year expiry)
