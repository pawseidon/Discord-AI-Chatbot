import re
import json

# Simplified version of the _clean_cache_response function from router_compatibility.py
def clean_cache_response(response_text):
    """Clean up a cached response to remove formatting artifacts"""
    # Handle tuple responses directly
    if isinstance(response_text, tuple):
        if len(response_text) >= 1:
            response_text = response_text[0]
        else:
            return "I couldn't format my response properly."
    
    # Ensure we're working with a string
    if not isinstance(response_text, str):
        try:
            response_text = str(response_text)
        except:
            return "I couldn't format my response properly."
    
    # Process string representation of tuples - only if it's a complete tuple
    # This helps avoid accidentally removing nested parentheses in the content
    if isinstance(response_text, str) and response_text.startswith("(") and response_text.endswith(")"):
        # Look for a full tuple pattern with metadata dict
        # Example: ("text", {'method': 'xyz'})
        full_tuple_pattern = r'^\s*\(\s*[\'"](.+?)[\'"]\s*,\s*\{.+\}\s*\)$'
        if re.search(full_tuple_pattern, response_text, re.DOTALL):
            try:
                # Extract just the text portion from the tuple
                match = re.search(r'^\s*\(\s*[\'"](.+?)[\'"]\s*,', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
                    return response_text  # Return early to avoid further processing
            except:
                pass
        
        # Simpler tuple pattern without metadata, still only if it's a full match
        simple_tuple_pattern = r'^\s*\(\s*[\'"](.+?)[\'"]\s*\)$'
        if re.search(simple_tuple_pattern, response_text, re.DOTALL):
            try:
                match = re.search(r'^\s*\(\s*[\'"](.+?)[\'"]', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
                    return response_text  # Return early to avoid further processing
            except:
                pass
    
    # Only clean up tuple patterns if they're actual tuple patterns, not just content with parentheses
    # This requires more specific pattern matching
    
    # Remove full tuple with metadata pattern if it exists as a full string
    tuple_with_metadata_pattern = r'^\s*\(\s*[\'"](.+?)[\'"]\s*,\s*\{.+\}\s*\)$'
    if re.search(tuple_with_metadata_pattern, response_text, re.DOTALL):
        try:
            match = re.search(r'^\s*\(\s*[\'"](.+?)[\'"]\s*,', response_text, re.DOTALL)
            if match:
                response_text = match.group(1)
        except:
            pass
    
    # Remove conclusion artifacts
    response_text = response_text.replace("Conclusion: ", "")
    
    # Clean up metadata patterns if present as separate text
    response_text = re.sub(r"\s*\{'method':.*?\}\s*", "", response_text).strip()
    
    # Remove any explicit tuple formatting for complete tuples only
    # This avoids removing content that happens to have parentheses
    if response_text.startswith("(") and response_text.endswith(")") and "'" in response_text:
        # Only strip outer quotes and parentheses, not internal ones
        response_text = response_text.strip()
        if response_text.startswith("(") and response_text.endswith(")"):
            response_text = response_text[1:-1].strip()
            if (response_text.startswith("'") and response_text.endswith("'")) or \
               (response_text.startswith('"') and response_text.endswith('"')):
                response_text = response_text[1:-1]
    
    return response_text

# Test the tuple response from our problem case
problem_response = """("I'm ready to assist you, Master Paws. What's on your mind today? Need help with something tech-related, or perhaps a query about the latest news or bitcoin?", {'method': 'default', 'method_name': 'Standard Processing', 'method_emoji': '💬', 'latency': 0.6827647686004639, 'total_latency': 0.6834619045257568, 'enhanced_context_used': True})"""

# Clean it up
cleaned_response = clean_cache_response(problem_response)

# Show the results
print(f"Original Response:\n{problem_response}\n")
print(f"Cleaned Response:\n{cleaned_response}\n")

# Additional test cases
test_cases = [
    # Tuple string with metadata
    ("(\"I'm ready to assist you, Master Paws. What's on your mind today?\", {'method': 'default', 'method_name': 'Standard Processing', 'method_emoji': '💬'})",
     "I'm ready to assist you, Master Paws. What's on your mind today?"),
    
    # Clean message
    ("Hello there!", "Hello there!"),
    
    # Actual tuple object
    ((("How can I help you?"), {"method": "standard"}), "How can I help you?"),
    
    # Complex nested response
    ("(\"I found some information: 1. First item, 2. Second item (with parens)\", {'method': 'search'})",
     "I found some information: 1. First item, 2. Second item (with parens)"),
     
    # Tuple with newlines
    ("(\"Here's a multi-line response.\nThis is line 2.\nThis is line 3.\", {'method': 'default'})",
     "Here's a multi-line response.\nThis is line 2.\nThis is line 3.")
]

# Run tests
for i, (input_text, expected) in enumerate(test_cases):
    result = clean_cache_response(input_text)
    success = result == expected
    print(f"Test {i+1}: {'✅ PASS' if success else '❌ FAIL'}")
    if not success:
        print(f"  Input: {input_text[:50]}{'...' if len(input_text) > 50 else ''}")
        print(f"  Expected: {expected[:50]}{'...' if len(expected) > 50 else ''}")
        print(f"  Got: {result[:50]}{'...' if len(result) > 50 else ''}")
        # Debug: print exact string representations for comparison
        print(f"  Expected (repr): {repr(expected)}")
        print(f"  Got (repr): {repr(result)}")
        # Print character by character comparison for the first 50 chars
        for j in range(min(50, len(result), len(expected))):
            if result[j] != expected[j]:
                print(f"  Diff at position {j}: expected '{expected[j]}' (ord {ord(expected[j])}), got '{result[j]}' (ord {ord(result[j])})")
                break 