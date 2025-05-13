import re
from typing import List, Dict, Any, Tuple

class TokenOptimizer:
    def __init__(self):
        # Common patterns to clean
        self.patterns = [
            (r'http[s]?://\S+', '[URL]'),  # Replace URLs
            (r'[\n\s]+', ' '),  # Compress whitespace
            (r'```[\s\S]*?```', '[CODE]'),  # Replace code blocks
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean text to reduce token usage"""
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)
        return text.strip()
    
    def truncate_text(self, text: str, max_tokens: int = 1000) -> str:
        """Roughly truncate text to stay within token limit"""
        # Very rough approximation: 1 token ≈ 4 characters for English
        # A more accurate solution would use tiktoken or a similar tokenizer
        if len(text) > max_tokens * 4:
            return text[:max_tokens * 4] + "..."
        return text
    
    def optimize_prompt(self, prompt: str, context: str, query: str, 
                       max_tokens: int = 4000) -> str:
        """Optimize a prompt by adjusting context to fit within token limits"""
        # Estimate token usage (very rough)
        prompt_tokens = len(prompt) // 4
        query_tokens = len(query) // 4
        
        # Calculate remaining tokens for context
        remaining_tokens = max_tokens - prompt_tokens - query_tokens - 100  # Buffer
        
        # Truncate context if needed
        optimized_context = self.truncate_text(context, remaining_tokens)
        
        # Replace context placeholder with optimized context
        full_prompt = prompt.replace("{context}", optimized_context)
        full_prompt = full_prompt.replace("{query}", query)
        
        return full_prompt
    
    def optimize_memory(self, memory: str, max_tokens: int = 800) -> str:
        """Optimize memory context to stay within token limits"""
        if not memory:
            return ""
        
        # Rough estimate of tokens
        if len(memory) > max_tokens * 4:
            # Split by lines (each conversation turn)
            lines = memory.split('\n')
            
            # Keep the first line (header) and the most recent conversations
            result = [lines[0]]  # Keep header
            
            # Count tokens used by header
            header_tokens = len(lines[0]) // 4
            available_tokens = max_tokens - header_tokens
            
            # Process conversation pairs (User: ... Assistant: ...)
            i = len(lines) - 1  # Start from the end
            conversation_pairs = []
            
            while i > 0 and available_tokens > 0:
                # Try to get a user-assistant pair
                if i >= 2 and i - 1 > 0:
                    assistant_line = lines[i]
                    user_line = lines[i-1]
                    
                    # Rough token estimate
                    pair_tokens = (len(assistant_line) + len(user_line)) // 4
                    
                    if pair_tokens <= available_tokens:
                        conversation_pairs.insert(0, user_line)
                        conversation_pairs.insert(1, assistant_line)
                        available_tokens -= pair_tokens
                        i -= 2
                    else:
                        break
                else:
                    break
            
            # Combine the result
            result.extend(conversation_pairs)
            return "\n".join(result)
        
        return memory

# Initialize global token optimizer
token_optimizer = TokenOptimizer() 