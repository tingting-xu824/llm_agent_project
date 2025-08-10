#!/usr/bin/env python3
"""
Comprehensive test script for RAG system with conversation storage and memory
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_rag_system():
    """Test the complete RAG system"""
    
    print("=== Testing RAG System with Conversation Storage ===\n")
    
    # 1. Register a user
    print("1. Registering user...")
    register_data = {
        "first_name": "RAG",
        "last_name": "Test",
        "email": f"rag.test.{int(time.time())}@example.com",
        "confirm_email": f"rag.test.{int(time.time())}@example.com",
        "date_of_birth": "1990-01-01",
        "gender": "male",
        "field_of_education": "Computer Science",
        "current_level_of_education": "Master",
        "disability_knowledge": "no",
        "ai_course_experience": "yes"
    }
    
    response = requests.post(f"{BASE_URL}/users/register", json=register_data)
    if response.status_code == 200:
        user_data = response.json()
        token = user_data["token"]
        user_id = user_data["user_id"]
        print(f"   User registered with ID: {user_id}")
    else:
        print(f"   Registration failed: {response.text}")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Send first message (about personal information)
    print("\n2. Sending first message (personal info)...")
    time.sleep(2)  # Avoid rate limiting
    
    message1 = "Hi! My name is John and I'm studying computer science. I love programming in Python and I'm working on a machine learning project."
    response = requests.post(f"{BASE_URL}/agent", 
                           json={"message": message1, "mode": "chat"}, 
                           headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Response: {result[0]['message'][:100]}...")
        print(f"   Memory context used: {result[0].get('memory_context_used', False)}")
    else:
        print(f"   Failed: {response.text}")
        return
    
    # 3. Send second message (asking about previous info)
    print("\n3. Sending second message (asking about previous info)...")
    time.sleep(2)
    
    message2 = "What do you remember about me from our previous conversation?"
    response = requests.post(f"{BASE_URL}/agent", 
                           json={"message": message2, "mode": "chat"}, 
                           headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Response: {result[0]['message'][:100]}...")
        print(f"   Memory context used: {result[0].get('memory_context_used', False)}")
    else:
        print(f"   Failed: {response.text}")
        return
    
    # 4. Send third message (about project details)
    print("\n4. Sending third message (project details)...")
    time.sleep(2)
    
    message3 = "I'm working on a natural language processing project that involves sentiment analysis and text classification. I'm using transformers and PyTorch."
    response = requests.post(f"{BASE_URL}/agent", 
                           json={"message": message3, "mode": "chat"}, 
                           headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Response: {result[0]['message'][:100]}...")
        print(f"   Memory context used: {result[0].get('memory_context_used', False)}")
    else:
        print(f"   Failed: {response.text}")
        return
    
    # 5. Ask about project details (testing memory retrieval)
    print("\n5. Testing memory retrieval...")
    time.sleep(2)
    
    message4 = "Can you remind me what I told you about my project?"
    response = requests.post(f"{BASE_URL}/agent", 
                           json={"message": message4, "mode": "chat"}, 
                           headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Response: {result[0]['message'][:100]}...")
        print(f"   Memory context used: {result[0].get('memory_context_used', False)}")
    else:
        print(f"   Failed: {response.text}")
        return
    
    # 6. Get conversation history from database
    print("\n6. Getting conversation history from database...")
    response = requests.get(f"{BASE_URL}/conversations?limit=10", headers=headers)
    
    if response.status_code == 200:
        conversations = response.json()
        print(f"   Found {len(conversations)} conversations in database")
        for i, conv in enumerate(conversations[:3], 1):
            print(f"   {i}. {conv['role']}: {conv['message'][:50]}...")
    else:
        print(f"   Failed: {response.text}")
    
    # 7. Test memory retrieval API
    print("\n7. Testing memory retrieval API...")
    response = requests.get(f"{BASE_URL}/memories?query=machine learning&top_k=3", headers=headers)
    
    if response.status_code == 200:
        memories = response.json()
        print(f"   Found {memories['count']} relevant memories for 'machine learning'")
        for i, memory in enumerate(memories['memories'][:2], 1):
            print(f"   Memory {i}: {memory['content'][:50]}... (similarity: {memory['similarity']:.2f})")
    else:
        print(f"   Failed: {response.text}")
    
    # 8. Test eval mode with memory
    print("\n8. Testing eval mode with memory...")
    time.sleep(2)
    
    eval_message = "Based on what you know about me, what kind of programming challenges would I be interested in?"
    response = requests.post(f"{BASE_URL}/agent", 
                           json={"message": eval_message, "mode": "eval"}, 
                           headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Eval Response: {result[0]['message'][:100]}...")
        print(f"   Memory context used: {result[0].get('memory_context_used', False)}")
    else:
        print(f"   Failed: {response.text}")
    
    print("\n=== RAG System Test Completed ===")

if __name__ == "__main__":
    test_rag_system()
