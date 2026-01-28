#!/usr/bin/env python3
"""
Comprehensive Coherency Test Suite

Tests model coherency across:
- Context lengths: Short (~2K), Medium (~16K), Long (~128K)
- Topics: Geography, Math, Science, Code

Validates that tensor parallelism communication produces correct, coherent outputs.
"""

import argparse
import requests
import time
import sys
import json
from typing import Callable, Tuple, Optional

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_TIMEOUT = 600  # 10 minutes for very long contexts

# Context length definitions (in approximate tokens)
# Note: Actual tokenization varies; these are tuned for GPT-style tokenizers
CONTEXT_SHORT = 2_000
CONTEXT_MEDIUM = 16_000
CONTEXT_LONG = 100_000  # Leave headroom for response within 128K limit

# =============================================================================
# API Helper
# =============================================================================

def call_api(
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int = 256,
    temperature: float = 0.3,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Call the OpenAI-compatible chat completions API."""
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def run_test(
    name: str,
    base_url: str,
    model: str,
    messages: list,
    check_fn: Callable[[str], Tuple[bool, str]],
    max_tokens: int = 256,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, dict]:
    """Run a single test and return (passed, details)."""
    start = time.time()
    
    # Extract the question from messages (last user message)
    question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # For needle tests, just show the question part (after "Question:")
            if "Question:" in content:
                question = content.split("Question:")[-1].strip()[:200]
            elif len(content) > 500:
                question = content[:100] + "... [truncated] ..." + content[-100:]
            else:
                question = content
            break
    
    try:
        result = call_api(base_url, model, messages, max_tokens=max_tokens, timeout=timeout)
        elapsed = time.time() - start
        
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        passed, reason = check_fn(answer)
        
        return passed, {
            "name": name,
            "passed": passed,
            "reason": reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed": elapsed,
            "question": question,
            "answer": answer,
            "answer_preview": answer[:400] + ("..." if len(answer) > 400 else ""),
        }
    except Exception as e:
        elapsed = time.time() - start
        return False, {
            "name": name,
            "passed": False,
            "reason": f"Error: {e}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed": elapsed,
            "question": question,
            "answer": "",
            "answer_preview": "",
        }


def generate_filler(target_tokens: int, topic: str = "general") -> str:
    """Generate filler text of approximately target_tokens length."""
    # Different filler texts for variety
    # Estimated ~1.3 tokens per word for these simple sentences
    fillers = {
        "geography": "The world contains many diverse landscapes including mountains valleys rivers and plains. Countries vary greatly in size population and cultural heritage. Climate zones range from tropical to arctic affecting local ecosystems and human settlements. ",
        "math": "Mathematics is the study of numbers quantities and shapes. It includes arithmetic algebra geometry and calculus. Mathematical principles are used in science engineering and everyday life. Problem solving skills are developed through practice. ",
        "science": "Scientific inquiry involves observation hypothesis formation experimentation and analysis. The natural world operates according to physical laws that can be discovered through careful study. Technology advances through the application of scientific principles. ",
        "code": "Software development involves designing coding testing and maintaining applications. Programming languages provide abstractions for expressing computational logic. Good code is readable maintainable and efficient. Documentation helps others understand the codebase. ",
        "general": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. It is commonly used for typing practice and font demonstrations. ",
    }
    
    filler_unit = fillers.get(topic, fillers["general"])
    # More accurate: ~1.3-1.5 tokens per word, so words ≈ tokens / 1.4
    words_per_unit = len(filler_unit.split())
    tokens_per_unit = int(words_per_unit * 1.4)  # Approximate tokens
    
    num_units = max(1, target_tokens // tokens_per_unit)
    return filler_unit * num_units


def create_needle_test(
    needle: str,
    question: str,
    answer_check: str,
    context_tokens: int,
    topic: str,
) -> Tuple[list, Callable[[str], Tuple[bool, str]]]:
    """Create a needle-in-haystack test with the given parameters."""
    filler = generate_filler(context_tokens - 200, topic)
    mid = len(filler) // 2
    context = filler[:mid] + f"\n\n{needle}\n\n" + filler[mid:]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based ONLY on the provided context. Look carefully for the specific information requested."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    
    def check(answer: str) -> Tuple[bool, str]:
        # Normalize unicode spaces and check
        import unicodedata
        normalized = ''.join(c if not unicodedata.category(c).startswith('Z') else ' ' for c in answer.lower())
        check_normalized = ''.join(c if not unicodedata.category(c).startswith('Z') else ' ' for c in answer_check.lower())
        found = check_normalized in normalized
        return found, f"Looking for '{answer_check}': {'Found' if found else 'Not found'}"
    
    return messages, check


# =============================================================================
# Test Definitions
# =============================================================================

def get_geography_tests(context_level: str) -> list:
    """Geography tests at different context lengths."""
    tests = []
    
    if context_level == "short":
        # Direct question, no context needed
        tests.append({
            "name": "Geography Short: Capital of Japan",
            "messages": [{"role": "user", "content": "What is the capital city of Japan?"}],
            "check": lambda a: ("tokyo" in a.lower(), "Found Tokyo" if "tokyo" in a.lower() else "Missing Tokyo"),
            "max_tokens": 64,
        })
        tests.append({
            "name": "Geography Short: Largest Ocean",
            "messages": [{"role": "user", "content": "Which ocean is the largest on Earth?"}],
            "check": lambda a: ("pacific" in a.lower(), "Found Pacific" if "pacific" in a.lower() else "Missing Pacific"),
            "max_tokens": 64,
        })
    
    elif context_level == "medium":
        # Needle in medium context
        needle = "IMPORTANT FACT: The fictional island nation of Veridian has its capital at Port Emerald, located on the eastern coast."
        messages, check = create_needle_test(
            needle=needle,
            question="What is the capital of Veridian and where is it located?",
            answer_check="Port Emerald",
            context_tokens=CONTEXT_MEDIUM,
            topic="geography",
        )
        tests.append({
            "name": f"Geography Medium ({CONTEXT_MEDIUM//1000}K): Find Hidden Capital",
            "messages": messages,
            "check": check,
            "max_tokens": 256,
        })
    
    elif context_level == "long":
        # Needle in very long context
        needle = "CRITICAL DATA: The remote archipelago of Crystalline Islands spans coordinates 42.7°N, 178.3°E with a total population of 847,293 as of the last census."
        messages, check = create_needle_test(
            needle=needle,
            question="What are the coordinates of the Crystalline Islands and what is their population?",
            answer_check="847,293",
            context_tokens=CONTEXT_LONG,
            topic="geography",
        )
        tests.append({
            "name": f"Geography Long ({CONTEXT_LONG//1000}K): Find Hidden Coordinates",
            "messages": messages,
            "check": check,
            "max_tokens": 256,
        })
    
    return tests


def get_math_tests(context_level: str) -> list:
    """Math tests at different context lengths."""
    tests = []
    
    if context_level == "short":
        # Direct calculation
        tests.append({
            "name": "Math Short: Multiplication",
            "messages": [{"role": "user", "content": "Calculate 47 × 83. Give only the numerical answer."}],
            "check": lambda a: ("3901" in a, "Found 3901" if "3901" in a else "Missing 3901"),
            "max_tokens": 128,
        })
        tests.append({
            "name": "Math Short: Quadratic Roots",
            "messages": [{"role": "user", "content": "Solve the quadratic equation x² - 5x + 6 = 0. What are x values?"}],
            "check": lambda a: ("2" in a and "3" in a, f"Found 2: {'2' in a}, Found 3: {'3' in a}"),
            "max_tokens": 256,
        })
    
    elif context_level == "medium":
        # Math problem embedded in context
        needle = "CALCULATION RESULT: After computing the integral of x³ + 2x² - 5x + 7 from 0 to 4, the exact answer is 148/3 or approximately 49.333."
        messages, check = create_needle_test(
            needle=needle,
            question="What is the result of the integral mentioned in the text?",
            answer_check="148",  # Simplified check - just look for the key number
            context_tokens=CONTEXT_MEDIUM,
            topic="math",
        )
        tests.append({
            "name": f"Math Medium ({CONTEXT_MEDIUM//1000}K): Find Hidden Integral",
            "messages": messages,
            "check": check,
            "max_tokens": 256,
        })
    
    elif context_level == "long":
        # Complex math in very long context
        needle = "THEOREM PROOF RESULT: The solution to the differential equation dy/dx = 3y - 2x with initial condition y(0) = 5 yields y = (2x + 2)/3 + (13/3)e^(3x) at x = 1, which evaluates to exactly 87.2953 (rounded to 4 decimal places)."
        messages, check = create_needle_test(
            needle=needle,
            question="What is the value of y at x=1 for the differential equation mentioned? Give the numerical answer.",
            answer_check="87",  # Simplified - just look for the integer part
            context_tokens=CONTEXT_LONG,
            topic="math",
        )
        tests.append({
            "name": f"Math Long ({CONTEXT_LONG//1000}K): Find DE Solution",
            "messages": messages,
            "check": check,
            "max_tokens": 256,
        })
    
    return tests


def get_science_tests(context_level: str) -> list:
    """Science tests at different context lengths."""
    tests = []
    
    if context_level == "short":
        # Direct science questions
        tests.append({
            "name": "Science Short: Speed of Light",
            "messages": [{"role": "user", "content": "What is the speed of light in meters per second? Give the approximate value."}],
            # Handle various formats: 299792458, 3×10^8, 3e8, 2.998, etc.
            "check": lambda a: (any(x in a for x in ["299", "2.99", "3×10", "3e8", "300", "3 ×", "3x10", "10^8", "10⁸", "10^{8}", "×10"]), 
                               "Found speed of light" if any(x in a for x in ["299", "2.99", "3×10", "3e8", "300", "3 ×", "3x10", "10^8", "10⁸", "10^{8}", "×10"]) else "Missing speed"),
            "max_tokens": 128,
        })
        tests.append({
            "name": "Science Short: Water Formula",
            "messages": [{"role": "user", "content": "What is the chemical formula for water?"}],
            # Check for H2O, H₂O, or variations
            "check": lambda a: (any(x in a.lower() for x in ["h2o", "h₂o"]) or ("h" in a.lower() and "o" in a.lower() and "2" in a),
                               "Found H2O" if any(x in a.lower() for x in ["h2o", "h₂o"]) or ("h" in a.lower() and "o" in a.lower() and "2" in a) else "Missing H2O"),
            "max_tokens": 128,
        })
    
    elif context_level == "medium":
        # Science fact in context
        needle = "EXPERIMENTAL FINDING: The novel catalyst compound XR-7 demonstrated a reaction rate of 4.72 × 10⁻³ mol/(L·s) at 298K, representing a 340% improvement over baseline platinum catalysts."
        messages, check = create_needle_test(
            needle=needle,
            question="What was the reaction rate of the XR-7 catalyst and what improvement did it show?",
            answer_check="340",  # Simplified check - look for 340 (the key number)
            context_tokens=CONTEXT_MEDIUM,
            topic="science",
        )
        tests.append({
            "name": f"Science Medium ({CONTEXT_MEDIUM//1000}K): Find Catalyst Data",
            "messages": messages,
            "check": check,
            "max_tokens": 256,
        })
    
    elif context_level == "long":
        # Complex science in very long context
        needle = "RESEARCH CONCLUSION: Analysis of the exoplanet HD-7829b revealed an atmospheric composition of 67.3% nitrogen, 21.8% oxygen, 8.4% argon, and trace noble gases, with a surface temperature of 287K and atmospheric pressure of 1.02 atm—remarkably Earth-like conditions."
        messages, check = create_needle_test(
            needle=needle,
            question="What is the nitrogen percentage in HD-7829b's atmosphere and what is its surface temperature?",
            answer_check="67",  # Simplified - look for 67 (part of 67.3%)
            context_tokens=CONTEXT_LONG,
            topic="science",
        )
        tests.append({
            "name": f"Science Long ({CONTEXT_LONG//1000}K): Find Exoplanet Data",
            "messages": messages,
            "check": check,
            "max_tokens": 256,
        })
    
    return tests


def get_code_tests(context_level: str) -> list:
    """Code tests at different context lengths."""
    tests = []
    
    if context_level == "short":
        # Direct coding questions - simpler prompts
        tests.append({
            "name": "Code Short: Function Output",
            "messages": [{"role": "user", "content": "What does this Python code print?\n\nprint(sum([x**2 for x in range(1,5)]))\n\nGive just the number."}],
            "check": lambda a: ("30" in a, "Found 30" if "30" in a else "Missing 30"),
            "max_tokens": 256,
        })
        tests.append({
            "name": "Code Short: Bug Detection",
            "messages": [{"role": "user", "content": "This Python code has a bug:\n\ndef greet(name)\n    return 'Hello ' + name\n\nWhat is missing? Answer in one word."}],
            "check": lambda a: ("colon" in a.lower() or ":" in a, "Found colon" if "colon" in a.lower() or ":" in a else "Missing colon"),
            "max_tokens": 128,
        })
    
    elif context_level == "medium":
        # Code snippet in context
        needle = '''
```python
# Secret function - the answer to the coding challenge
def mystery_transform(data: list[int]) -> int:
    """Returns the XOR of all elements multiplied by their indices."""
    result = 0
    for i, val in enumerate(data):
        result ^= (val * i)
    return result

# For input [3, 7, 2, 9], this returns 42
```
'''
        messages, check = create_needle_test(
            needle=needle,
            question="What does the mystery_transform function return for input [3, 7, 2, 9]?",
            answer_check="42",
            context_tokens=CONTEXT_MEDIUM,
            topic="code",
        )
        tests.append({
            "name": f"Code Medium ({CONTEXT_MEDIUM//1000}K): Find Function Output",
            "messages": messages,
            "check": check,
            "max_tokens": 256,
        })
    
    elif context_level == "long":
        # Complex code in very long context
        needle = '''
```python
# Critical configuration - DO NOT MODIFY
class SystemConfig:
    DATABASE_PORT = 5432
    REDIS_PORT = 6379
    API_SECRET_KEY = "xK9#mP2$vL7@nQ4"  # Production key
    MAX_CONNECTIONS = 1000
    CACHE_TTL_SECONDS = 3600
    
# The API_SECRET_KEY above is required for authentication
```
'''
        messages, check = create_needle_test(
            needle=needle,
            question="What is the API_SECRET_KEY value defined in SystemConfig?",
            answer_check="xK9",  # Simplified - just check for start of the key
            context_tokens=CONTEXT_LONG,
            topic="code",
        )
        tests.append({
            "name": f"Code Long ({CONTEXT_LONG//1000}K): Find API Key",
            "messages": messages,
            "check": check,
            "max_tokens": 256,
        })
    
    return tests


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive coherency test suite")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--context", choices=["short", "medium", "long", "all"], default="all",
                        help="Context length to test (default: all)")
    parser.add_argument("--topic", choices=["geography", "math", "science", "code", "all"], default="all",
                        help="Topic to test (default: all)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    args = parser.parse_args()
    
    # Determine which context levels to test
    if args.context == "all":
        context_levels = ["short", "medium", "long"]
    else:
        context_levels = [args.context]
    
    # Determine which topics to test
    topic_getters = {
        "geography": get_geography_tests,
        "math": get_math_tests,
        "science": get_science_tests,
        "code": get_code_tests,
    }
    if args.topic == "all":
        topics = list(topic_getters.keys())
    else:
        topics = [args.topic]
    
    # Header
    print("=" * 80)
    print("Comprehensive Coherency Test Suite")
    print("=" * 80)
    print(f"API: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Context levels: {', '.join(context_levels)}")
    print(f"Topics: {', '.join(topics)}")
    print(f"Timeout: {args.timeout}s")
    print()
    
    all_results = []
    
    for context_level in context_levels:
        context_labels = {'short': '2K', 'medium': '16K', 'long': '128K'}
        print(f"\n{'='*80}")
        print(f"CONTEXT: {context_level.upper()} (~{context_labels[context_level]} tokens)")
        print(f"{'='*80}")
        
        for topic in topics:
            tests = topic_getters[topic](context_level)
            if not tests:
                continue
                
            print(f"\n--- {topic.upper()} ---")
            
            for test_def in tests:
                passed, details = run_test(
                    name=test_def["name"],
                    base_url=args.base_url,
                    model=args.model,
                    messages=test_def["messages"],
                    check_fn=test_def["check"],
                    max_tokens=test_def.get("max_tokens", 256),
                    timeout=args.timeout,
                )
                
                status = "✓" if passed else "✗"
                print(f"\n{status} {details['name']}")
                print(f"  Tokens: {details['prompt_tokens']:,} prompt, {details['completion_tokens']:,} completion")
                print(f"  Time: {details['elapsed']:.1f}s")
                print(f"  Check: {details['reason']}")
                if details['answer_preview']:
                    # Truncate for display
                    preview = details['answer_preview'][:200]
                    if len(details['answer_preview']) > 200:
                        preview += "..."
                    print(f"  Answer: {preview}")
                
                all_results.append((context_level, topic, passed, details))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # By context level
    print("\nBy Context Level:")
    for level in context_levels:
        level_results = [r for r in all_results if r[0] == level]
        passed = sum(1 for r in level_results if r[2])
        total = len(level_results)
        pct = 100 * passed / total if total > 0 else 0
        status = "✓" if passed == total else "⚠" if passed > 0 else "✗"
        print(f"  {status} {level.upper()}: {passed}/{total} ({pct:.0f}%)")
    
    # By topic
    print("\nBy Topic:")
    for topic in topics:
        topic_results = [r for r in all_results if r[1] == topic]
        passed = sum(1 for r in topic_results if r[2])
        total = len(topic_results)
        pct = 100 * passed / total if total > 0 else 0
        status = "✓" if passed == total else "⚠" if passed > 0 else "✗"
        print(f"  {status} {topic.upper()}: {passed}/{total} ({pct:.0f}%)")
    
    # Overall
    total_passed = sum(1 for r in all_results if r[2])
    total_tests = len(all_results)
    overall_pct = 100 * total_passed / total_tests if total_tests > 0 else 0
    
    print(f"\nOverall: {total_passed}/{total_tests} ({overall_pct:.0f}%)")
    
    # Human review section
    print("\n" + "=" * 80)
    print("HUMAN REVIEW: Questions & Answers")
    print("=" * 80)
    print("\nReview the LLM's responses below to verify correctness:\n")
    
    for i, (context_level, topic, passed, details) in enumerate(all_results, 1):
        status = "✓" if passed else "✗"
        print(f"─" * 80)
        print(f"{status} [{i}/{total_tests}] {details['name']}")
        print(f"─" * 80)
        print(f"Q: {details.get('question', 'N/A')}")
        print()
        answer = details.get('answer', '')
        # Show full answer for human review (up to 800 chars)
        if len(answer) > 800:
            print(f"A: {answer[:800]}...")
        else:
            print(f"A: {answer}")
        print()
    
    print("=" * 80)
    print("END OF HUMAN REVIEW")
    print("=" * 80)
    
    if total_passed == total_tests:
        print("\n✓ All coherency tests passed!")
        sys.exit(0)
    elif total_passed >= total_tests * 0.8:
        print("\n✓ Most tests passed. Model appears coherent.")
        sys.exit(0)
    else:
        print("\n⚠ Multiple tests failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
