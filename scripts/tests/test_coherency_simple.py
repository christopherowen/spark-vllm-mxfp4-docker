#!/usr/bin/env python3
"""Simple coherency tests for TP=2 validation."""

import requests
import sys

BASE_URL = "http://localhost:8000/v1"
MODEL = "gpt-oss-120b"

def test(name, messages, check_fn, max_tokens=256):
    """Run a test and check the response."""
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7},
            timeout=120,
        )
        result = resp.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        usage = result.get("usage", {})
        
        passed, reason = check_fn(answer)
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        print(f"  Tokens: {usage.get('prompt_tokens', 0)} prompt, {usage.get('completion_tokens', 0)} completion")
        print(f"  Check: {reason}")
        print(f"  Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print()
        return passed
    except Exception as e:
        print(f"✗ {name}: Error - {e}")
        print()
        return False

def main():
    print("=" * 60)
    print("TP=2 Coherency Tests")
    print("=" * 60)
    print()
    
    results = []

    # Test 1: Basic math coherence
    results.append(test(
        "Basic Math (17 * 23 = 391)",
        [{"role": "user", "content": "What is 17 * 23? Show your work step by step, then give the final answer."}],
        lambda a: ("391" in a, "Found 391" if "391" in a else "Missing 391")
    ))

    # Test 2: Factual recall
    results.append(test(
        "Factual Recall (Capital of France)",
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        lambda a: ("paris" in a.lower(), "Found Paris" if "paris" in a.lower() else "Missing Paris")
    ))

    # Test 3: Instruction following
    results.append(test(
        "Instruction Following (Prime numbers)",
        [{"role": "user", "content": "List exactly 5 prime numbers less than 20, separated by commas."}],
        lambda a: (sum(1 for p in ["2", "3", "5", "7", "11", "13", "17", "19"] if p in a) >= 4, 
                   f"Found {sum(1 for p in ['2','3','5','7','11','13','17','19'] if p in a)} primes")
    ))

    # Test 4: Long context needle (4K tokens)
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    needle = "SECRET: The password is blueberry42."
    context = filler[:len(filler)//2] + "\n" + needle + "\n" + filler[len(filler)//2:]
    results.append(test(
        "4K Context Needle (find password)",
        [{"role": "user", "content": f"Read this text and find the password:\n\n{context}\n\nWhat is the password?"}],
        lambda a: ("blueberry42" in a.lower(), "Found password" if "blueberry42" in a.lower() else "Missing password"),
        max_tokens=64
    ))

    # Test 5: Coherent generation
    results.append(test(
        "Coherent Generation (haiku)",
        [{"role": "user", "content": "Write a haiku about the ocean."}],
        lambda a: (len(a.split()) >= 8, f"Generated {len(a.split())} words")
    ))

    # Test 6: Multi-turn coherence
    results.append(test(
        "Multi-turn Context",
        [
            {"role": "user", "content": "My name is Alice and I live in Seattle."},
            {"role": "assistant", "content": "Nice to meet you, Alice! Seattle is a beautiful city."},
            {"role": "user", "content": "What is my name and where do I live?"}
        ],
        lambda a: ("alice" in a.lower() and "seattle" in a.lower(), 
                   f"Found Alice: {'alice' in a.lower()}, Seattle: {'seattle' in a.lower()}")
    ))

    # Summary
    print("=" * 60)
    print(f"Summary: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All coherency tests passed! TP=2 communication is working correctly.")
        sys.exit(0)
    elif sum(results) >= len(results) - 1:
        print("\n✓ Most tests passed. TP=2 appears to be working correctly.")
        sys.exit(0)
    else:
        print("\n⚠ Multiple tests failed. Review output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
