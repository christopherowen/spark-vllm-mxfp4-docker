#!/usr/bin/env python3
"""
Long Context Coherency Test for TP=2 vLLM

Tests that the model produces coherent, accurate responses at various context lengths.
This validates that tensor parallelism and RDMA communication are working correctly.
"""

import argparse
import requests
import json
import time
import sys

def call_api(base_url: str, model: str, messages: list, max_tokens: int = 256, temperature: float = 0.0) -> dict:
    """Call the OpenAI-compatible API."""
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()

def test_needle_in_haystack(base_url: str, model: str, context_length: int) -> tuple[bool, str]:
    """
    Needle-in-a-haystack test: hide a fact in a long context and ask about it.
    """
    # Create filler text (approximately 4 chars per token)
    filler_unit = "The quick brown fox jumps over the lazy dog. " * 10  # ~100 tokens
    target_filler_tokens = context_length - 200  # Leave room for needle and question
    
    num_units = max(1, target_filler_tokens // 100)
    
    # The needle (secret fact)
    needle = "IMPORTANT SECRET: The capital of the fictional country Zephyria is Moonhaven City."
    
    # Place needle in the middle
    mid_point = num_units // 2
    filler_before = filler_unit * mid_point
    filler_after = filler_unit * (num_units - mid_point)
    
    context = f"{filler_before}\n\n{needle}\n\n{filler_after}"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What is the capital of Zephyria?"},
    ]
    
    try:
        start = time.time()
        result = call_api(base_url, model, messages, max_tokens=64)
        elapsed = time.time() - start
        
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if answer is None:
            answer = ""
        answer_lower = answer.lower()
        success = "moonhaven" in answer_lower
        
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        
        return success, f"Found: {success} | Tokens: {prompt_tokens} | Time: {elapsed:.1f}s | Answer: {answer[:100]}..."
    except Exception as e:
        return False, f"Error: {e}"

def test_arithmetic_chain(base_url: str, model: str, chain_length: int) -> tuple[bool, str]:
    """
    Test long arithmetic reasoning chains to verify coherent generation.
    """
    # Build a chain of arithmetic operations
    operations = []
    current = 100
    for i in range(chain_length):
        op = (i % 3)
        if op == 0:
            delta = (i + 1) * 7
            operations.append(f"Step {i+1}: Add {delta} to get {current + delta}")
            current += delta
        elif op == 1:
            delta = (i + 1) * 3
            operations.append(f"Step {i+1}: Subtract {delta} to get {current - delta}")
            current -= delta
        else:
            operations.append(f"Step {i+1}: Double the value to get {current * 2}")
            current *= 2
    
    context = "Follow this calculation chain:\n" + "\n".join(operations)
    
    messages = [
        {"role": "user", "content": f"{context}\n\nWhat is the final value after all steps?"},
    ]
    
    try:
        start = time.time()
        result = call_api(base_url, model, messages, max_tokens=128)
        elapsed = time.time() - start
        
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if answer is None:
            answer = ""
        # Check if the correct final value appears in the answer
        success = str(current) in answer
        
        return success, f"Expected: {current} | Found: {success} | Time: {elapsed:.1f}s | Answer: {answer[:150]}..."
    except Exception as e:
        return False, f"Error: {e}"

def test_repetition_coherence(base_url: str, model: str, num_items: int) -> tuple[bool, str]:
    """
    Test that the model can accurately recall items from a list.
    """
    items = [f"Item_{i:03d}_{'ALPHA' if i % 2 == 0 else 'BETA'}" for i in range(num_items)]
    target_idx = num_items // 2
    target_item = items[target_idx]
    
    context = "Here is a list of items:\n" + "\n".join(items)
    
    messages = [
        {"role": "user", "content": f"{context}\n\nWhat is item number {target_idx + 1} in the list? Reply with just the item name."},
    ]
    
    try:
        start = time.time()
        result = call_api(base_url, model, messages, max_tokens=32)
        elapsed = time.time() - start
        
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if answer is None:
            answer = ""
        success = target_item in answer
        
        return success, f"Expected: {target_item} | Found: {success} | Time: {elapsed:.1f}s | Answer: {answer}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Long context coherency tests")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Long Context Coherency Tests")
    print("=" * 70)
    print(f"API: {args.base_url}")
    print(f"Model: {args.model}")
    print()
    
    tests = []
    
    # Needle-in-haystack at various context lengths
    print("Test 1: Needle-in-Haystack (find hidden fact in context)")
    print("-" * 50)
    for ctx_len in [1000, 4000, 8000, 16000]:
        success, details = test_needle_in_haystack(args.base_url, args.model, ctx_len)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {ctx_len:>6} tokens: {status}")
        print(f"    {details}")
        tests.append(("needle", ctx_len, success))
    print()
    
    # Arithmetic chains
    print("Test 2: Arithmetic Chain (verify reasoning coherence)")
    print("-" * 50)
    for chain_len in [10, 25, 50]:
        success, details = test_arithmetic_chain(args.base_url, args.model, chain_len)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {chain_len:>3} steps: {status}")
        print(f"    {details}")
        tests.append(("arithmetic", chain_len, success))
    print()
    
    # List recall
    print("Test 3: List Recall (verify exact memory)")
    print("-" * 50)
    for num_items in [50, 100, 200]:
        success, details = test_repetition_coherence(args.base_url, args.model, num_items)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {num_items:>3} items: {status}")
        print(f"    {details}")
        tests.append(("list_recall", num_items, success))
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    passed = sum(1 for _, _, s in tests if s)
    total = len(tests)
    print(f"Passed: {passed}/{total} ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n✓ All coherency tests passed! TP=2 communication is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
