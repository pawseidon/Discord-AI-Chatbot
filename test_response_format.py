import re
import ast

def clean_response(response_text):
    """Test function to clean response text for debugging"""
    
    # If input is already a tuple, extract text directly
    if isinstance(response_text, tuple):
        if len(response_text) >= 1:
            return response_text[0]
        else:
            return "Empty tuple"
    
    # Handle string representations of tuples
    if isinstance(response_text, str) and response_text.startswith("(") and response_text.endswith(")"):
        # Try to evaluate as a literal tuple first
        try:
            tuple_eval = ast.literal_eval(response_text)
            if isinstance(tuple_eval, tuple) and len(tuple_eval) > 0:
                return str(tuple_eval[0])
        except:
            # Fall back to regex if literal_eval fails
            pass
            
        # Check if this looks like a tuple with metadata dict
        if "{" in response_text and "}" in response_text:
            try:
                # Extract just the text portion
                match = re.search(r'^\s*\(\s*[\'"](.+?)[\'"]\s*,\s*\{', response_text, re.DOTALL)
                if match:
                    return match.group(1)
            except:
                pass
                
        # Simpler tuple pattern without metadata
        try:
            match = re.search(r'^\s*\(\s*[\'"](.+?)[\'"]', response_text, re.DOTALL)
            if match:
                return match.group(1)
        except:
            pass
    
    # Clean up metadata patterns if present
    response_text = re.sub(r"\{'method':.*?\}", "", response_text).strip()
    
    # Use safer pattern matching - capture the text in the tuple and replace the whole string with it
    tuple_pattern = r"\('(.*?)', \{'method':.*?\}\)"
    if re.search(tuple_pattern, response_text):
        response_text = re.sub(tuple_pattern, r"\1", response_text).strip()
    
    # Remove any remaining formatting artifacts
    response_text = response_text.strip('"\'() ')
    
    return response_text

# Test cases
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
     "Here's a multi-line response.\nThis is line 2.\nThis is line 3."),
     
    # Nested tuples
    ("(\"The answer is (x, y) coordinate pairs.\", {'method': 'default'})",
     "The answer is (x, y) coordinate pairs."),
     
    # Malformed tuple
    ("(I'm not properly formatted, {'method': 'default'})",
     "I'm not properly formatted"),
     
    # Tuple with code blocks
    ("(\"Here's some code: ```python\ndef hello():\n    print('world')\n```\", {'method': 'default'})",
     "Here's some code: ```python\ndef hello():\n    print('world')\n```")
]

# Run tests
for i, (input_text, expected) in enumerate(test_cases):
    result = clean_response(input_text)
    success = result == expected
    print(f"Test {i+1}: {'✅ PASS' if success else '❌ FAIL'}")
    if not success:
        print(f"  Input: {input_text[:50]}{'...' if len(input_text) > 50 else ''}")
        print(f"  Expected: {expected[:50]}{'...' if len(expected) > 50 else ''}")
        print(f"  Got: {result[:50]}{'...' if len(result) > 50 else ''}")

print("\nRunning the test...") 