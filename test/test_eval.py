#!/usr/bin/env python3
"""
Simple test script for eval mode with o3 model
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_eval_mode():
    """Test eval mode with o3 model"""
    
    # First register a user to get a token
    print("1. Registering user...")
    register_data = {
        "first_name": "Test",
        "last_name": "User",
        "email": f"test.user.{int(time.time())}@example.com",
        "confirm_email": f"test.user.{int(time.time())}@example.com",
        "date_of_birth": "1990-01-01",
        "gender": "male",
        "field_of_education": "Computer Science",
        "current_level_of_education": "Bachelor",
        "disability_knowledge": "no",
        "ai_course_experience": "yes"
    }
    
    response = requests.post(f"{BASE_URL}/users/register", json=register_data)
    if response.status_code == 200:
        user_data = response.json()
        token = user_data["token"]
        print(f"   User registered with token: {token[:20]}...")
    else:
        print(f"   Registration failed: {response.text}")
        return
    
    # Wait a bit to avoid rate limiting
    print("2. Waiting 3 seconds to avoid rate limiting...")
    time.sleep(3)
    
    # Test eval mode
    print("3. Testing eval mode with o3 model...")
    eval_data = {
        "message": "This is a test message for eval mode",
        "mode": "eval"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/agent", json=eval_data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Eval mode successful!")
        print(f"   Response: {result[0]['message'][:100]}...")
    else:
        print(f"   Eval mode failed: {response.text}")

if __name__ == "__main__":
    test_eval_mode()
