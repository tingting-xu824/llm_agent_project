# Evaluation API Endpoints Documentation

This document describes the new evaluation endpoints for handling idea evaluation submissions and round completion.

## Overview

The evaluation system consists of 4 rounds where users submit problems and solutions, receive AI feedback, and progress through the evaluation process. Each round must be completed before moving to the next.

## Endpoints

### 1. POST /evaluation?round=X

**Purpose**: Submit problem and solution for evaluation round and receive AI feedback

**Parameters**:
- `round` (query parameter): Evaluation round number (1, 2, 3, or 4)

**Request Body**:
```json
{
  "problem": "Description of the problem to be solved",
  "solution": "Proposed solution to the problem"
}
```

**Response**:
```json
{
  "message": "Evaluation submitted successfully",
  "ai_feedback": "AI agent's feedback on the problem and solution",
  "round": 1,
  "user_id": 123
}
```

**Note**: For Round 4 (final first thought), no AI feedback is generated. The response will be:
```json
{
  "message": "Final evaluation submitted successfully",
  "round": 4,
  "user_id": 123
}
```

**Error Responses**:
- `400 Bad Request`: 
  - Previous round not completed
  - No evaluation record found
  - Empty problem or solution
  - Invalid round number
- `500 Internal Server Error`: Database or AI agent error

**Logic Flow**:
1. Check if previous round is completed (except for round 1)
2. Verify evaluation record exists for user and round
3. Validate problem and solution are not empty
4. For rounds 1-3: Get user's assigned AI agent and generate AI feedback
5. For round 4: Skip AI feedback generation (final first thought submission)
6. Update database record with problem, solution, and AI feedback (if applicable)
7. Return success response with AI feedback (except for round 4)

### 2. POST /evaluation/complete?round=X

**Purpose**: Complete an evaluation round and create next round record

**Parameters**:
- `round` (query parameter): Evaluation round to complete (1, 2, 3, or 4)

**Request Body**: None (uses authentication token)

**Response**:
```json
{
  "message": "Round 1 completed successfully. Round 2 is now available.",
  "round": 1,
  "user_id": 123,
  "completed_at": "2024-01-15T10:30:00Z"
}
```

**Error Responses**:
- `400 Bad Request`:
  - Previous round not completed
  - No evaluation record found
  - Round not submitted (missing problem/solution)
- `500 Internal Server Error`: Database error

**Logic Flow**:
1. Check if previous round is completed (except for round 1)
2. Verify evaluation record exists for user and round
3. Ensure current round has problem and solution submitted
4. Set `completed_at` timestamp for current round
5. Create new record for next round (if not round 4)
6. Return success response

## Database Schema

The `idea_evaluation` table structure:

```sql
CREATE TABLE idea_evaluation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    problem TEXT NOT NULL,
    solution TEXT NOT NULL,
    ai_feedback TEXT,
    round INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    time_remaining BIGINT,
    completed_at TIMESTAMP,
    UNIQUE(user_id, round)
);
```

## Round Flow

1. **Round 1**: User submits problem and solution → AI feedback → Complete round
2. **Round 2**: User submits problem and solution → AI feedback → Complete round  
3. **Round 3**: User submits problem and solution → AI feedback → Complete round
4. **Round 4**: User submits problem and solution → No AI feedback → Complete round (final first thought)

## AI Agent Assignment

Users are assigned AI agents based on their user_id:
- If user_id ends with odd digit (1,3,5,7,9): Agent 1 (Originality Checker)
- If user_id ends with even digit (0,2,4,6,8): Agent 2 (Creative Idea Generator)

## Authentication

Both endpoints require authentication via cookie token. The user must be logged in and have a valid `auth_token` cookie.

## Example Usage

### Submit Evaluation (Round 1)
```bash
curl -X POST "http://localhost:8000/evaluation?round=1" \
  -H "Content-Type: application/json" \
  -H "Cookie: auth_token=your_token_here" \
  -d '{
    "problem": "How to reduce food waste in restaurants?",
    "solution": "Implement smart inventory management system with AI predictions"
  }'
```

### Complete Round
```bash
curl -X POST "http://localhost:8000/evaluation/complete?round=1" \
  -H "Cookie: auth_token=your_token_here"
```

## Error Handling

The endpoints include comprehensive error handling:

1. **Validation Errors**: Check input data validity
2. **Business Logic Errors**: Ensure proper round progression
3. **Database Errors**: Handle database operation failures
4. **AI Agent Errors**: Handle AI feedback generation failures

## Security Considerations

1. **Authentication**: All endpoints require valid user authentication
2. **Authorization**: Users can only access their own evaluation records
3. **Input Validation**: All inputs are validated for security and data integrity
4. **Rate Limiting**: Existing rate limiting applies to these endpoints

## Testing

Use the provided test script `test_evaluation_endpoints.py` to verify endpoint functionality:

```bash
python3 test_evaluation_endpoints.py
```

## Integration Notes

- These endpoints integrate with the existing authentication system
- They use the same AI agent infrastructure as the chat endpoints
- Database operations are optimized with async wrappers
- Error messages are detailed and user-friendly
