#!/usr/bin/env python3
"""
Test script for word count validation in evaluation endpoints
This script tests the word count requirements for different rounds
"""

import asyncio
import sys
import os

# Add the parent directory to the path to import from agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.api import get_round_word_requirements, count_words

def test_word_count_function():
    """Test the word count function"""
    print("=== Testing Word Count Function ===")
    
    # Test cases
    test_cases = [
        ("", 0),
        ("Hello world", 2),
        ("This is a test sentence with multiple words", 8),
        ("One", 1),
        ("   Multiple   spaces   ", 3),  # Should handle multiple spaces
    ]
    
    for text, expected in test_cases:
        result = count_words(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} Text: '{text}' -> Expected: {expected}, Got: {result}")

def test_round_requirements():
    """Test the round word requirements function"""
    print("\n=== Testing Round Word Requirements ===")
    
    expected_requirements = {
        1: (100, 500, 100, 500),  # Problem: 100-500, Solution: 100-500
        2: (100, 500, 150, 500),  # Problem: 100-500, Solution: 150-500
        3: (100, 500, 200, 500),  # Problem: 100-500, Solution: 200-500
        4: (100, 500, 250, 500),  # Problem: 100-500, Solution: 250-500
        5: (100, 500, 100, 500),  # Default fallback
    }
    
    for round_num, expected in expected_requirements.items():
        result = get_round_word_requirements(round_num)
        status = "✅" if result == expected else "❌"
        print(f"{status} Round {round_num}: Expected {expected}, Got {result}")

def test_validation_scenarios():
    """Test various validation scenarios"""
    print("\n=== Testing Validation Scenarios ===")
    
    # Test data for different rounds
    test_scenarios = [
        {
            "round": 1,
            "problem": "The accessibility challenges in public transportation systems present a significant barrier for individuals with disabilities, elderly populations, and those with limited mobility. Current systems often lack comprehensive features that address the diverse needs of all users, including those with visual, auditory, physical, and cognitive impairments. The problem encompasses inadequate information dissemination, insufficient physical infrastructure, and limited technological integration that could enhance the overall user experience for everyone. Additionally, there are significant gaps in the current transportation infrastructure that prevent equal access for all users. The lack of standardized accessibility features across different transportation modes creates confusion and frustration for users who rely on these services. Furthermore, the rapid advancement of technology has created new opportunities for improving accessibility, but many transportation systems have been slow to adopt these innovations. This technological lag further exacerbates the accessibility divide and prevents many individuals from fully participating in society.",
            "solution": "Implement a comprehensive accessibility enhancement system for public transportation that addresses multiple user needs through integrated technological and infrastructure solutions. The solution includes developing a multi-modal information system with voice announcements, tactile indicators, and mobile application integration that provides real-time updates and personalized navigation assistance. Physical infrastructure improvements involve installing accessible boarding platforms, clear signage with multiple formats, and emergency assistance systems. The mobile application should feature voice navigation, customizable interface options, real-time service updates, and integration with personal assistive devices. Additionally, staff training programs and community feedback mechanisms should be established to ensure continuous improvement and user satisfaction. This holistic approach ensures that public transportation becomes truly accessible for all users, regardless of their abilities or limitations.",
            "should_pass": True
        },
        {
            "round": 2,
            "problem": "The accessibility challenges in public transportation systems present a significant barrier for individuals with disabilities, elderly populations, and those with limited mobility. Current systems often lack comprehensive features that address the diverse needs of all users, including those with visual, auditory, physical, and cognitive impairments. The problem encompasses inadequate information dissemination, insufficient physical infrastructure, and limited technological integration that could enhance the overall user experience for everyone. Additionally, there are significant gaps in the current transportation infrastructure that prevent equal access for all users. The lack of standardized accessibility features across different transportation modes creates confusion and frustration for users who rely on these services. Furthermore, the rapid advancement of technology has created new opportunities for improving accessibility, but many transportation systems have been slow to adopt these innovations. This technological lag further exacerbates the accessibility divide and prevents many individuals from fully participating in society.",
            "solution": "Implement a comprehensive accessibility enhancement system for public transportation that addresses multiple user needs through integrated technological and infrastructure solutions. The solution includes developing a multi-modal information system with voice announcements, tactile indicators, and mobile application integration that provides real-time updates and personalized navigation assistance. Physical infrastructure improvements involve installing accessible boarding platforms, clear signage with multiple formats, and emergency assistance systems. The mobile application should feature voice navigation, customizable interface options, real-time service updates, and integration with personal assistive devices. Additionally, staff training programs and community feedback mechanisms should be established to ensure continuous improvement and user satisfaction. This holistic approach ensures that public transportation becomes truly accessible for all users, regardless of their abilities or limitations. The implementation process should be phased over multiple stages to ensure smooth transition and minimal disruption to existing services. Regular user feedback sessions and accessibility audits should be conducted to maintain high standards and address emerging needs.",
            "should_pass": True
        },
        {
            "round": 1,
            "problem": "Short problem",
            "solution": "Short solution",
            "should_pass": False
        },
        {
            "round": 2,
            "problem": "This is a test problem with enough words to meet the minimum requirement of one hundred words for round two evaluation. The problem should be clearly defined and provide enough context for the solution to be meaningful and comprehensive.",
            "solution": "Short solution that doesn't meet the 150 word requirement for round 2",
            "should_pass": False
        }
    ]
    
    for scenario in test_scenarios:
        round_num = scenario["round"]
        problem = scenario["problem"]
        solution = scenario["solution"]
        should_pass = scenario["should_pass"]
        
        problem_min, problem_max, solution_min, solution_max = get_round_word_requirements(round_num)
        problem_count = count_words(problem)
        solution_count = count_words(solution)
        
        problem_ok = problem_min <= problem_count <= problem_max
        solution_ok = solution_min <= solution_count <= solution_max
        validation_passed = problem_ok and solution_ok
        
        status = "✅" if validation_passed == should_pass else "❌"
        print(f"{status} Round {round_num}: Problem {problem_count}/{problem_min}-{problem_max}, Solution {solution_count}/{solution_min}-{solution_max} -> {'PASS' if validation_passed else 'FAIL'} (Expected: {'PASS' if should_pass else 'FAIL'})")

def test_edge_cases():
    """Test edge cases for word counting"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        ("", 0, "Empty string"),
        ("   ", 0, "Only spaces"),
        ("\n\t", 0, "Only whitespace"),
        ("a", 1, "Single character"),
        ("word", 1, "Single word"),
        ("word word", 2, "Two words"),
        ("word   word", 2, "Multiple spaces between words"),
        ("word.word", 1, "Word with punctuation"),
        ("word, word", 2, "Words with comma"),
        ("word! word?", 2, "Words with punctuation"),
    ]
    
    for text, expected, description in edge_cases:
        result = count_words(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} {description}: '{text}' -> Expected: {expected}, Got: {result}")

def main():
    """Main test function"""
    print("Starting word count validation tests...\n")
    
    test_word_count_function()
    test_round_requirements()
    test_validation_scenarios()
    test_edge_cases()
    
    print("\n=== Test Summary ===")
    print("All word count validation tests completed!")

if __name__ == "__main__":
    main()
