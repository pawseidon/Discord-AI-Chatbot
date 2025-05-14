import re
import string
import tiktoken
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger("token_utils")

# Constant for tiktoken encoding
ENCODING_NAME = "cl100k_base"  # For gpt-4, claude, etc.

class TokenOptimizer:
    """Implements advanced token optimization techniques for LLM interactions"""
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the token optimizer
        
        Args:
            model_name: The name of the model to optimize for
        """
        self.model_name = model_name
        try:
            self.encoding = tiktoken.get_encoding(ENCODING_NAME)
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}. Using fallback token counting.")
            self.encoding = None
        self.max_tokens = self._get_model_max_tokens(model_name)
        
    def _get_model_max_tokens(self, model_name: str) -> int:
        """Get the maximum context length for a given model"""
        model_max_tokens = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 16385,  # Updated limit
            "gpt-3.5-turbo-16k": 16385,
            "claude-instant-1": 100000,
            "claude-2": 100000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "anthropic.claude-3-opus:1": 200000,
            "anthropic.claude-3-sonnet:1": 200000,
            "anthropic.claude-3-haiku:1": 200000,
            "llama-3-70b-instruct": 8192,
            "llama-3-8b-instruct": 8192,
            "llama-3-1b-instruct": 8192,
            "mistral-7b-v0.2": 32768,
            "mistral-medium": 32768,
            "mistral-small": 32768,
            "mistral-tiny": 32768,
            "gemini-1.0-pro": 32768,
            "gemini-1.5-pro": 1000000,  # Supports up to 1M tokens
            "default": 8192  # Default fallback (increased)
        }
        return model_max_tokens.get(model_name.lower(), model_max_tokens["default"])
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        if not text:
            return 0
        
        if self.encoding:
            try:
                tokens = self.encoding.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Error counting tokens with tiktoken: {e}. Using fallback.")
                # Fallback to approximate count
                return self._fallback_token_count(text)
        else:
            return self._fallback_token_count(text)
    
    def _fallback_token_count(self, text: str) -> int:
        """Fallback method to approximately count tokens"""
        # Simple approximation: 1 token ≈ 4 chars for English text
        return len(text) // 4 + 1
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a message list format (role, content)"""
        if not messages:
            return 0
            
        token_count = 0
        for message in messages:
            # Add tokens for message format (varies by model but this is a safe estimate)
            token_count += 4  # ~4 tokens for message formatting
            
            # Add tokens for role
            role = message.get("role", "")
            token_count += self.count_tokens(role)
            
            # Add tokens for content
            content = message.get("content", "")
            token_count += self.count_tokens(content)
            
        # Add tokens for overall formatting
        token_count += 2  # ~2 tokens for overall formatting
        
        return token_count
    
    def truncate_text(self, text: str, max_tokens: int, preserve_suffix_tokens: int = 0) -> str:
        """
        Truncate text to fit within max_tokens
        
        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed
            preserve_suffix_tokens: Number of tokens to preserve from the end
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
            
        # Check if truncation is needed
        if self.encoding:
            tokens = self.encoding.encode(text)
            
            if len(tokens) <= max_tokens:
                return text
                
            # Simple case: no suffix preservation
            if preserve_suffix_tokens <= 0:
                truncated_tokens = tokens[:max_tokens]
                return self.encoding.decode(truncated_tokens)
                
            # Complex case: preserve suffix tokens
            if preserve_suffix_tokens >= max_tokens:
                preserve_suffix_tokens = max_tokens // 2  # Ensure we don't preserve too much
                
            prefix_tokens = max_tokens - preserve_suffix_tokens
            prefix = self.encoding.decode(tokens[:prefix_tokens])
            suffix = self.encoding.decode(tokens[-preserve_suffix_tokens:])
            
            # Add ellipsis between prefix and suffix
            return prefix + " [...] " + suffix
        else:
            # Fallback method when encoding is not available
            if len(text) // 4 <= max_tokens:
                return text
                
            if preserve_suffix_tokens <= 0:
                # Simple approximation based on characters
                char_limit = max_tokens * 4
                return text[:char_limit]
            else:
                # With suffix preservation
                suffix_chars = preserve_suffix_tokens * 4
                prefix_chars = (max_tokens - preserve_suffix_tokens) * 4
                
                if suffix_chars + prefix_chars >= len(text):
                    return text
                    
                prefix = text[:prefix_chars]
                suffix = text[-suffix_chars:]
                
                return prefix + " [...] " + suffix
    
    def smart_truncate_messages(self, messages: List[Dict[str, str]], max_tokens: int, 
                               system_message_priority: float = 1.0) -> List[Dict[str, str]]:
        """
        Intelligently truncate messages to fit within token limit
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum token count to target
            system_message_priority: Priority factor for preserving system messages
            
        Returns:
            Truncated message list
        """
        if not messages:
            return []
            
        current_count = self.count_message_tokens(messages)
        if current_count <= max_tokens:
            return messages
            
        result_messages = []
        
        # First, identify system messages for special handling
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # Calculate token budgets
        excess_tokens = current_count - max_tokens
        
        # Determine how much to reduce each category
        if system_message_priority > 0:
            system_token_count = self.count_message_tokens(system_messages)
            other_token_count = self.count_message_tokens(other_messages)
            
            # Allocate token reduction proportionally with priority weighting
            total_weighted = system_token_count/system_message_priority + other_token_count
            system_reduction = int(excess_tokens * (system_token_count/system_message_priority) / total_weighted)
            other_reduction = excess_tokens - system_reduction
        else:
            # Only reduce other messages if system has absolute priority
            system_reduction = 0
            other_reduction = excess_tokens
        
        # Process system messages (often just one)
        if system_messages:
            # Calculate how many tokens to allocate per system message
            tokens_per_system = max(1, (self.count_message_tokens(system_messages) - system_reduction) // len(system_messages))
            
            for msg in system_messages:
                content = msg.get("content", "")
                new_content = self.truncate_text(content, tokens_per_system - 4)  # Subtract 4 for message formatting
                result_messages.append({"role": "system", "content": new_content})
        
        # Process other messages (focus on most recent)
        if other_messages:
            # Sort by recency (assuming they're already in chronological order)
            recent_first = list(reversed(other_messages))
            
            # Prioritize user-assistant alternating pattern retention
            alternating_pairs = []
            i = 0
            while i < len(recent_first) - 1:
                if recent_first[i]["role"] == "user" and recent_first[i+1]["role"] == "assistant":
                    alternating_pairs.append((recent_first[i], recent_first[i+1]))
                    i += 2
                else:
                    i += 1
            
            # Determine tokens for alternating pairs and remaining messages
            pair_token_budget = int(0.7 * (self.count_message_tokens(other_messages) - other_reduction))
            remaining_budget = (self.count_message_tokens(other_messages) - other_reduction) - pair_token_budget
            
            # Add the important user-assistant pairs (most recent first)
            added_messages = []
            tokens_used = 0
            for user_msg, assistant_msg in alternating_pairs:
                pair_tokens = self.count_message_tokens([user_msg, assistant_msg])
                if tokens_used + pair_tokens <= pair_token_budget:
                    added_messages.extend([assistant_msg, user_msg])  # Reverse to restore chronological order later
                    tokens_used += pair_tokens
                else:
                    # Try to fit in partially
                    user_tokens = self.count_tokens(user_msg.get("content", ""))
                    assistant_tokens = self.count_tokens(assistant_msg.get("content", ""))
                    
                    # More aggressive truncation on assistant responses than user queries
                    if user_tokens + assistant_tokens > 0:
                        user_proportion = user_tokens / (user_tokens + assistant_tokens)
                        remaining_pair_budget = pair_token_budget - tokens_used
                        
                        # Allocate remaining budget
                        user_budget = int(remaining_pair_budget * user_proportion * 1.2)  # Favor user messages slightly
                        assistant_budget = remaining_pair_budget - user_budget
                        
                        # Truncate and add
                        truncated_user = dict(user_msg)
                        truncated_user["content"] = self.truncate_text(
                            user_msg.get("content", ""), 
                            user_budget
                        )
                        
                        truncated_assistant = dict(assistant_msg)
                        truncated_assistant["content"] = self.truncate_text(
                            assistant_msg.get("content", ""), 
                            assistant_budget
                        )
                        
                        added_messages.extend([truncated_assistant, truncated_user])
                        tokens_used = pair_token_budget  # Consider budget fully used
                    break  # Stop adding pairs
            
            # Add other relevant messages with remaining budget if available
            if remaining_budget > 20:  # Only if we have meaningful budget left
                singleton_messages = [msg for msg in recent_first if all(
                    msg is not pair_msg 
                    for pair in alternating_pairs 
                    for pair_msg in pair
                )]
                
                for msg in singleton_messages:
                    msg_tokens = self.count_message_tokens([msg])
                    if remaining_budget >= msg_tokens:
                        added_messages.append(msg)
                        remaining_budget -= msg_tokens
                    else:
                        # Truncate to fit
                        truncated_msg = dict(msg)
                        truncated_msg["content"] = self.truncate_text(
                            msg.get("content", ""), 
                            max(1, remaining_budget - 4)  # Account for message overhead
                        )
                        added_messages.append(truncated_msg)
                        break
            
            # Restore chronological order
            result_messages.extend(reversed(added_messages))
        
        # Final check to ensure we're within budget
        final_count = self.count_message_tokens(result_messages)
        if final_count > max_tokens:
            # Emergency truncation (shouldn't usually happen)
            logger.warning(f"Smart truncation failed to meet token budget: {final_count}/{max_tokens}")
            
            # Further truncate the longest message
            messages_by_length = sorted(
                enumerate(result_messages), 
                key=lambda x: self.count_tokens(x[1].get("content", "")),
                reverse=True
            )
            
            for idx, msg in messages_by_length:
                content = msg.get("content", "")
                tokens_to_remove = final_count - max_tokens + 5  # Add some buffer
                new_content = self.truncate_text(
                    content, 
                    max(1, self.count_tokens(content) - tokens_to_remove)
                )
                result_messages[idx]["content"] = new_content
                break
        
        return result_messages
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace, normalizing quotes, etc."""
        if not text:
            return ""
        
        # Replace multiple newlines with single
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        
        # Remove common advertising phrases
        ad_phrases = [
            r'Subscribe to read the full article',
            r'Sign up for our newsletter',
            r'Log in to continue reading',
            r'You have reached your.*limit',
            r'Create your free account',
            r'Exclusive offer',
            r'Limited time only',
            r'Click here',
            r'Join now',
            r'For more information, visit',
            r'All rights reserved'
        ]
        
        for phrase in ad_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
        
        # Clean up HTML artifacts
        text = re.sub(r'</?[a-z][^>]*>', '', text)  # Remove HTML tags
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        return text.strip()

# Create a default instance for easier imports
token_optimizer = TokenOptimizer() 