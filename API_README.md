# Token-based API Documentation

## Overview

The API has been changed from session-based authentication to token-based authentication. Users are now authenticated through tokens stored in the User table.

## Major Changes

1. **Authentication Method**: Changed from cookies to Bearer token
2. **User Management**: Added user registration and profile management
3. **Database**: Using Neon PostgreSQL to store user information and tokens
4. **Session Management**: Still using Redis for distributed session storage

## API Endpoints

### 1. User Registration
```http
POST /users/register
Content-Type: application/json

{
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@example.com",
    "confirm_email": "john.doe@example.com",
    "date_of_birth": "1990-01-15",
    "gender": "male",
    "field_of_education": "Computer Science",
    "current_level_of_education": "Bachelor's Degree",
    "disability_knowledge": "no",
    "ai_course_experience": "yes"
}
```

**Required Fields:**
- `first_name` (string): First name
- `last_name` (string): Last name
- `email` (email): Email address
- `confirm_email` (email): Confirm email address
- `date_of_birth` (date): Date of birth (YYYY-MM-DD)
- `gender` (string): Gender ("male", "female", "other", "prefer_not_to_say")
- `disability_knowledge` (string): Disability knowledge ("yes" or "no")
- `ai_course_experience` (string): AI course experience ("yes" or "no")

**Optional Fields:**
- `field_of_education` (string): Field of education
- `current_level_of_education` (string): Current education level

**Response:**
```json
{
    "user_id": 1,
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@example.com",
    "token": "generated_token_here",
    "message": "User registered successfully"
}
```

### 2. User Login
```http
POST /users/login
Content-Type: application/json

{
    "email": "john.doe@example.com",
    "date_of_birth": "1990-01-15"
}
```

**Parameters:**
- `email` (email): Email address
- `date_of_birth` (date): Date of birth (YYYY-MM-DD)

**Response:**
```json
{
    "user_id": 1,
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@example.com",
    "token": "existing_token_here",
    "message": "Login successful"
}
```

### 3. Get User Profile
```http
GET /users/profile
Authorization: Bearer {token}
```

**Response:**
```json
{
    "id": 1,
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@example.com",
    "date_of_birth": "1990-01-15",
    "gender": "male",
    "field_of_education": "Computer Science",
    "current_level_of_education": "Bachelor's Degree",
    "disability_knowledge": "no",
    "ai_course_experience": "yes",
    "created_at": "2024-01-01T00:00:00",
    "last_login": "2024-01-01T12:00:00"
}
```

### 4. Send Message to Agent
```http
POST /agent
Authorization: Bearer {token}
Content-Type: application/json

{
    "message": "Hello, how are you?",
    "mode": "chat"  // or "eval"
}
```

**Response:**
```json
[
    {
        "message": "Hello! I'm doing well, thank you for asking.",
        "type": "agent_response",
        "timestamp": "2024-01-01T12:00:01Z"
    }
]
```

### 5. Get Chat History
```http
GET /agent
Authorization: Bearer {token}
```

**Response:**
```json
[
    {
        "message": "Hello, how are you?",
        "type": "user_input",
        "timestamp": "2024-01-01T12:00:00Z"
    },
    {
        "message": "Hello! I'm doing well, thank you for asking.",
        "type": "agent_response",
        "timestamp": "2024-01-01T12:00:01Z"
    }
]
```

## Usage Examples

### Python Example
```python
import requests
from datetime import date

# 1. Register user
register_data = {
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@example.com",
    "confirm_email": "john.doe@example.com",
    "date_of_birth": "1990-01-15",
    "gender": "male",
    "field_of_education": "Computer Science",
    "current_level_of_education": "Bachelor's Degree",
    "disability_knowledge": "no",
    "ai_course_experience": "yes"
}
response = requests.post("http://localhost:8000/users/register", json=register_data)
user_data = response.json()
token = user_data["token"]

# 2. Login user
login_data = {
    "email": "john.doe@example.com",
    "date_of_birth": "1990-01-15"
}
response = requests.post("http://localhost:8000/users/login", json=login_data)
login_result = response.json()
token = login_result["token"]

# 3. Set headers
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# 4. Send message
chat_data = {
    "message": "Hello, how are you?",
    "mode": "chat"
}
response = requests.post("http://localhost:8000/agent", json=chat_data, headers=headers)
result = response.json()
print(result[0]["message"])

# 5. Get history
response = requests.get("http://localhost:8000/agent", headers=headers)
history = response.json()
print(f"Total messages: {len(history)}")
```

### JavaScript Example
```javascript
// 1. Register user
const registerData = {
    first_name: "John",
    last_name: "Doe",
    email: "john.doe@example.com",
    confirm_email: "john.doe@example.com",
    date_of_birth: "1990-01-15",
    gender: "male",
    field_of_education: "Computer Science",
    current_level_of_education: "Bachelor's Degree",
    disability_knowledge: "no",
    ai_course_experience: "yes"
};

const registerResponse = await fetch('/users/register', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(registerData)
});
const userData = await registerResponse.json();
const token = userData.token;

// 2. Login user
const loginData = {
    email: "john.doe@example.com",
    date_of_birth: "1990-01-15"
};

const loginResponse = await fetch('/users/login', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(loginData)
});
const loginResult = await loginResponse.json();
const token = loginResult.token;

// 3. Send message
const response = await fetch('/agent', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        message: 'Hello, how are you?',
        mode: 'chat'
    })
});
const result = await response.json();
console.log(result[0].message);
```

## Environment Variables Configuration

```bash
# Neon PostgreSQL configuration
DATABASE_URL=postgresql://username:password@host:port/database

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password

# OpenAI configuration
OPENAI_API_KEY=your_openai_api_key

# Rate limiting
COOLDOWN_SECONDS=3
```

## Database Setup

1. **Create database in Neon**:
   - Login to Neon console
   - Create new project or use existing project
   - Get connection string

2. **Create users table**:
   - Create users table in Neon console SQL editor
   - Table structure includes: id, username, email, token, created_at, last_login

## Start Service

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (if using Docker)
docker run -d -p 6379:6379 redis:alpine

# Set environment variables
export DATABASE_URL="your_neon_connection_string"

# Start API service
uvicorn agents.api:app --host 0.0.0.0 --port 8000 --reload
```

## Test

Run test script:
```bash
python test_api.py
```

## Database Structure

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(50) NOT NULL,
    field_of_education VARCHAR(255),
    current_level_of_education VARCHAR(255),
    disability_knowledge VARCHAR(10) NOT NULL,
    ai_course_experience VARCHAR(10) NOT NULL,
    token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Security Considerations

1. **Token Security**: Tokens are generated using `secrets.token_urlsafe(32)` with sufficient randomness
2. **HTTPS**: Use HTTPS in production environment
3. **Token Expiration**: Consider adding token expiration mechanism
4. **Rate Limiting**: Implemented Redis-based request frequency limiting
5. **Input Validation**: All user inputs are validated

## Error Handling

Common error codes:
- `400`: Request parameter error
- `401`: Invalid token
- `404`: User history not found
- `429`: Too many requests
- `500`: Internal server error
