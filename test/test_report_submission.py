#!/usr/bin/env python3
"""
Test script for final report submission functionality
"""
import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"

def test_report_submission():
    """Test the report submission functionality"""
    print("=== Testing Final Report Submission ===")
    
    # Step 1: Register a test user
    print("1. Registering test user...")
    register_data = {
        "first_name": "Test",
        "last_name": "User",
        "email": f"test.report.{int(time.time())}@example.com",
        "confirm_email": f"test.report.{int(time.time())}@example.com",
        "date_of_birth": "1990-01-15",
        "gender": "other",
        "field_of_education": "Computer Science",
        "current_level_of_education": "Bachelor's Degree",
        "disability_knowledge": "no",
        "ai_course_experience": "yes"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/users/register", json=register_data)
        if response.status_code != 200:
            print(f"Failed to register user: {response.text}")
            return False
        
        user_data = response.json()
        token = user_data["token"]
        user_id = user_data["user_id"]
        print(f"User registered successfully: ID {user_id}")
        
        # Set cookies for authenticated requests
        cookies = {
            "auth_token": token
        }
        
        # Step 2: Test report submission with file
        print("\n2. Testing report submission with file...")
        
        # Create a test file
        test_file_content = "This is a test report content."
        test_file_path = "test_report.txt"
        
        with open(test_file_path, "w") as f:
            f.write(test_file_content)
        
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_report.txt", f, "text/plain")}
            response = requests.post(
                f"{BASE_URL}/report/submit",
                files=files,
                cookies=cookies
            )
        
        # Clean up test file
        os.remove(test_file_path)
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Report submitted successfully!")
            print(f"Report ID: {result.get('report_id')}")
            print(f"File URL: {result.get('file_url')}")
        else:
            print(f"Failed to submit report: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Step 3: Test getting the report
        print("\n3. Testing report retrieval...")
        response = requests.get(f"{BASE_URL}/report/get", cookies=cookies)
        
        if response.status_code == 200:
            report = response.json()
            print("SUCCESS: Report retrieved successfully!")
            print(f"Report ID: {report.get('id')}")
            print(f"File URL: {report.get('file_url')}")
            print(f"Created at: {report.get('created_at')}")
        else:
            print(f"Failed to get report: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Step 4: Test duplicate submission (should fail)
        print("\n4. Testing duplicate submission (should fail)...")
        
        # Create another test file
        test_file_content2 = "This is another test report content."
        test_file_path2 = "test_report2.txt"
        
        with open(test_file_path2, "w") as f:
            f.write(test_file_content2)
        
        with open(test_file_path2, "rb") as f:
            files2 = {"file": ("test_report2.txt", f, "text/plain")}
            response = requests.post(
                f"{BASE_URL}/report/submit",
                files=files2,
                cookies=cookies
            )
        
        # Clean up test file
        os.remove(test_file_path2)
        
        if response.status_code == 400:
            print("SUCCESS: Duplicate submission correctly rejected!")
        else:
            print(f"Unexpected response for duplicate submission: {response.status_code}")
            print(f"Response: {response.text}")
        
        print("\n=== All tests completed successfully ===")
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("\n=== Testing File Upload ===")
    
    # Create a test file
    test_file_content = "This is a test file content for upload testing."
    test_file_path = "test_file.txt"
    
    try:
        with open(test_file_path, "w") as f:
            f.write(test_file_content)
        
        # Register a test user
        register_data = {
            "first_name": "File",
            "last_name": "Test",
            "email": f"file.test.{int(time.time())}@example.com",
            "confirm_email": f"file.test.{int(time.time())}@example.com",
            "date_of_birth": "1990-01-15",
            "gender": "other",
            "field_of_education": "Computer Science",
            "current_level_of_education": "Bachelor's Degree",
            "disability_knowledge": "no",
            "ai_course_experience": "yes"
        }
        
        response = requests.post(f"{BASE_URL}/users/register", json=register_data)
        if response.status_code != 200:
            print(f"Failed to register user: {response.text}")
            return False
        
        user_data = response.json()
        token = user_data["token"]
        user_id = user_data["user_id"]
        
        # Set cookies for authenticated requests
        cookies = {
            "auth_token": token
        }
        
        # Submit report with file
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_file.txt", f, "text/plain")}
            response = requests.post(
                f"{BASE_URL}/report/submit",
                files=files,
                cookies=cookies
            )
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Report with file uploaded successfully!")
            print(f"File URL: {result.get('file_url')}")
        else:
            print(f"Failed to submit report with file: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Clean up test file
        os.remove(test_file_path)
        
        print("=== File upload test completed successfully ===")
        return True
        
    except Exception as e:
        print(f"Error during file upload testing: {str(e)}")
        # Clean up test file if it exists
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        return False

if __name__ == "__main__":
    success1 = test_report_submission()
    success2 = test_file_upload()
    
    if success1 and success2:
        print("\n🎉 All tests PASSED!")
    else:
        print("\n❌ Some tests FAILED!")
