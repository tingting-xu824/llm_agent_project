#!/usr/bin/env python3
"""
Test script for the token-based API

This script demonstrates how to use the API with token-based authentication.
It includes user registration, login, authentication, and API endpoint testing.
"""

import requests
import json
from datetime import date
import time

# API base URL - change this to your server URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Main test function that demonstrates all API endpoints"""
    print("=== Testing Token-based API ===\n")
    
    # Step 1: Register a new user with comprehensive profile
    print("1. Registering a new user...")
    # Use timestamp to make email unique
    timestamp = int(time.time())
    register_data = {
        "first_name": "John",
        "last_name": "Doe",
        "email": f"john.doe.{timestamp}@example.com",
        "confirm_email": f"john.doe.{timestamp}@example.com",
        "date_of_birth": "1990-01-15",
        "gender": "male",
        "field_of_education": "Computer Science",
        "current_level_of_education": "Bachelor's Degree",
        "disability_knowledge": "no",
        "ai_course_experience": "yes"
    }
    
    response = requests.post(f"{BASE_URL}/users/register", json=register_data)
    if response.status_code == 200:
        user_data = response.json()
        token = user_data["token"]
        user_id = user_data["user_id"]
        print(f"   User registered successfully!")
        print(f"   User ID: {user_id}")
        print(f"   Name: {user_data['first_name']} {user_data['last_name']}")
        print(f"   Email: {user_data['email']}")
        print(f"   Token: {token[:20]}...")
    else:
        print(f"   Registration failed: {response.text}")
        return
    
    # Set up authentication headers with Bearer token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Step 2: Get user profile to verify registration data
    print("\n2. Getting user profile...")
    response = requests.get(f"{BASE_URL}/users/profile", headers=headers)
    if response.status_code == 200:
        profile = response.json()
        print(f"   Profile retrieved successfully!")
        print(f"   User ID: {profile['user_id']}")
        print(f"   Email: {profile['email']}")
        print(f"   Gender: {profile['gender']}")
        print(f"   Education: {profile['education_field']}")
        print(f"   AI Experience: {profile['genai_course_exp']}")
    else:
        print(f"   Failed to get profile: {response.text}")
    
    # Step 3: Test login with email and date of birth
    print("\n3. Testing login...")
    login_data = {
        "email": f"john.doe.{timestamp}@example.com",
        "date_of_birth": "1990-01-15"
    }
    
    response = requests.post(f"{BASE_URL}/users/login", json=login_data)
    if response.status_code == 200:
        login_result = response.json()
        print(f"   Login successful!")
        print(f"   User ID: {login_result['user_id']}")
        print(f"   Email: {login_result['email']}")
        print(f"   Token: {login_result['token'][:20]}...")
        
        # Update token for subsequent requests
        token = login_result["token"]
        headers["Authorization"] = f"Bearer {token}"
    else:
        print(f"   Login failed: {response.text}")
    
    # Step 4: Send a chat message to test conversation
    print("\n4. Sending a chat message...")
    chat_data = {
        "message": "Hello, how are you?",
        "mode": "chat"
    }
    
    response = requests.post(f"{BASE_URL}/agent", json=chat_data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        print(f"   Chat message sent successfully!")
        print(f"   Response: {result[0]['message'][:100]}...")
    else:
        print(f"   Failed to send chat message: {response.text}")
    
    # Step 5: Send an eval message to test idea evaluation
    print("\n5. Sending an eval message...")
    eval_data = {
        "message": "I have an idea for a mobile app",
        "mode": "eval"
    }
    
    response = requests.post(f"{BASE_URL}/agent", json=eval_data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        print(f"   Eval message sent successfully!")
        print(f"   Response: {result[0]['message'][:100]}...")
    else:
        print(f"   Failed to send eval message: {response.text}")
    
    # Step 6: Get chat history to verify conversation persistence
    print("\n6. Getting chat history...")
    response = requests.get(f"{BASE_URL}/agent", headers=headers)
    if response.status_code == 200:
        history = response.json()
        print(f"   History retrieved successfully!")
        print(f"   Number of messages: {len(history)}")
        for i, msg in enumerate(history[:3]):  # Show first 3 messages
            print(f"   {i+1}. {msg['type']}: {msg['message'][:50]}...")
    else:
        print(f"   Failed to get history: {response.text}")
    
    print("\n=== Test completed ===")

def test_validation_errors():
    """Test various validation errors"""
    print("\n=== Testing Validation Errors ===\n")
    
    # Test 1: Email mismatch
    print("1. Testing email mismatch...")
    register_data = {
        "first_name": "Test",
        "last_name": "User",
        "email": "test@example.com",
        "confirm_email": "different@example.com",  # Mismatch
        "date_of_birth": "1995-05-20",
        "gender": "female",
        "disability_knowledge": "no",
        "ai_course_experience": "no"
    }
    
    response = requests.post(f"{BASE_URL}/users/register", json=register_data)
    if response.status_code == 422:
        print("   Email mismatch validation working correctly")
    else:
        print(f"   Email mismatch validation failed: {response.status_code}")
    
    # Test 2: Invalid gender
    print("\n2. Testing invalid gender...")
    register_data["confirm_email"] = "test@example.com"  # Fix email
    register_data["gender"] = "invalid_gender"
    
    response = requests.post(f"{BASE_URL}/users/register", json=register_data)
    if response.status_code == 422:
        print("   Gender validation working correctly")
    else:
        print(f"   Gender validation failed: {response.status_code}")
    
    # Test 3: Future date of birth
    print("\n3. Testing future date of birth...")
    register_data["gender"] = "female"
    register_data["date_of_birth"] = "2030-01-01"  # Future date
    
    response = requests.post(f"{BASE_URL}/users/register", json=register_data)
    if response.status_code == 422:
        print("   Date of birth validation working correctly")
    else:
        print(f"   Date of birth validation failed: {response.status_code}")
    
    print("\n=== Validation Tests Completed ===")

if __name__ == "__main__":
    test_api()
    test_validation_errors()
