import re
import string
import tiktoken
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from collections import Counter

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
        self.encoding = tiktoken.get_encoding(ENCODING_NAME)
        self.max_tokens = self._get_model_max_tokens(model_name)
        
    def _get_model_max_tokens(self, model_name: str) -> int:
        """Get the maximum context length for a given model"""
        model_max_tokens = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "claude-instant": 100000,
            "claude-2": 100000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "llama-3-70b": 8192,
            "mistral-medium": 32768,
            "gemini-pro": 32768,
            "default": 4096  # Default fallback
        }
        return model_max_tokens.get(model_name.lower(), model_max_tokens["default"])
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        if not text:
            return 0
        tokens = self.encoding.encode(text)
        return len(tokens)
    
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
                        user_budget = int(remaining_pair_budget * user_proportion * 1.2)  # Favor preserving user message
                        assistant_budget = remaining_pair_budget - user_budget
                        
                        truncated_user = {"role": "user", 
                                          "content": self.truncate_text(user_msg.get("content", ""), user_budget)}
                        truncated_assistant = {"role": "assistant", 
                                               "content": self.truncate_text(assistant_msg.get("content", ""), assistant_budget)}
                        
                        added_messages.extend([truncated_assistant, truncated_user])
                    break
            
            # Add any leftover high-priority messages with remaining budget
            other_messages_set = set((m["role"], m.get("content", "")) for m in other_messages)
            added_set = set((m["role"], m.get("content", "")) for m in added_messages)
            remaining_messages = []
            
            for msg in other_messages:
                if (msg["role"], msg.get("content", "")) not in added_set:
                    remaining_messages.append(msg)
            
            # Sort remaining messages by priority (most recent first)
            remaining_messages.reverse()
            
            # Add as many as possible within budget
            remaining_tokens = remaining_budget
            for msg in remaining_messages:
                msg_tokens = self.count_message_tokens([msg])
                if remaining_tokens >= msg_tokens:
                    added_messages.append(msg)
                    remaining_tokens -= msg_tokens
                else:
                    # Try truncation for the last message
                    content_tokens = self.count_tokens(msg.get("content", ""))
                    if content_tokens > 0 and remaining_tokens > 10:  # Only if reasonable space remains
                        truncated_content = self.truncate_text(msg.get("content", ""), remaining_tokens - 5)
                        added_messages.append({"role": msg["role"], "content": truncated_content})
                    break
            
            # Restore chronological order
            added_messages.reverse()
            result_messages.extend(added_messages)
        
        # Add marker if we truncated messages
        if len(result_messages) < len(messages):
            trunc_note = {"role": "system", "content": "(Note: Some earlier messages were truncated to fit token limits)"}
            result_messages.insert(0, trunc_note)
        
        return result_messages
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute a simple measure of semantic similarity between two texts"""
        # Simple bag-of-words cosine similarity
        # Convert to lowercase and split into words
        words1 = re.findall(r'\b\w+\b', text1.lower())
        words2 = re.findall(r'\b\w+\b', text2.lower())
        
        # Create word frequency dictionaries
        counter1 = Counter(words1)
        counter2 = Counter(words2)
        
        # Get all unique words
        all_words = set(counter1.keys()).union(set(counter2.keys()))
        
        # Create vectors
        vec1 = [counter1.get(word, 0) for word in all_words]
        vec2 = [counter2.get(word, 0) for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compress_context(self, messages: List[Dict[str, str]], max_tokens: int, 
                         relevance_query: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Compress context by merging similar messages and removing redundancies
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to target
            relevance_query: Optional query to determine message relevance
            
        Returns:
            Compressed message list
        """
        if not messages or len(messages) <= 2:
            return messages
            
        current_count = self.count_message_tokens(messages)
        if current_count <= max_tokens:
            return messages
            
        # Extract system messages for separate treatment
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        # Identify redundant/similar messages for merging or removal
        # Start with assistant messages as they're typically longer
        compressed_assistant = self._compress_similar_messages(
            assistant_messages, 
            similarity_threshold=0.6,
            relevance_query=relevance_query
        )
        
        # Handle user messages similarly but with higher similarity threshold
        compressed_user = self._compress_similar_messages(
            user_messages, 
            similarity_threshold=0.75,
            relevance_query=relevance_query
        )
        
        # Combine results while preserving conversation flow
        result = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                result.append(msg)
            elif role == "user":
                # Find this message in compressed list
                for comp_msg in compressed_user:
                    if comp_msg.get("original_index", -1) == next(
                        (i for i, m in enumerate(user_messages) if m.get("content") == msg.get("content")), 
                        -1
                    ):
                        result.append({"role": "user", "content": comp_msg.get("content", "")})
                        break
            elif role == "assistant":
                # Find this message in compressed list
                for comp_msg in compressed_assistant:
                    if comp_msg.get("original_index", -1) == next(
                        (i for i, m in enumerate(assistant_messages) if m.get("content") == msg.get("content")), 
                        -1
                    ):
                        result.append({"role": "assistant", "content": comp_msg.get("content", "")})
                        break
        
        # If still over token limit, use smart truncation as fallback
        if self.count_message_tokens(result) > max_tokens:
            return self.smart_truncate_messages(result, max_tokens)
            
        return result
    
    def _compress_similar_messages(self, messages: List[Dict[str, str]], 
                                 similarity_threshold: float = 0.7,
                                 relevance_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compress similar messages by merging or filtering"""
        if not messages or len(messages) <= 1:
            return messages
            
        # Add original indices
        indexed_messages = []
        for i, msg in enumerate(messages):
            indexed_messages.append({
                "content": msg.get("content", ""),
                "original_index": i
            })
            
        # If we have a relevance query, compute relevance scores
        if relevance_query:
            for msg in indexed_messages:
                content = msg.get("content", "")
                msg["relevance"] = self.compute_semantic_similarity(content, relevance_query)
        else:
            # Without query, assign equal relevance prioritizing recent messages
            for i, msg in enumerate(indexed_messages):
                msg["relevance"] = 0.5 + (i / (2 * len(indexed_messages)))  # Score from 0.5 to 1.0 based on recency
        
        # Sort by relevance (highest first)
        indexed_messages.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Create clusters of similar messages
        clusters = []
        for msg in indexed_messages:
            content = msg.get("content", "")
            
            # Check if this message is similar to any existing cluster
            found_cluster = False
            for cluster in clusters:
                for cluster_msg in cluster:
                    cluster_content = cluster_msg.get("content", "")
                    similarity = self.compute_semantic_similarity(content, cluster_content)
                    
                    if similarity >= similarity_threshold:
                        cluster.append(msg)
                        found_cluster = True
                        break
                        
                if found_cluster:
                    break
                    
            # If not similar to any cluster, create a new one
            if not found_cluster:
                clusters.append([msg])
        
        # Process each cluster to produce a representative message
        compressed = []
        for cluster in clusters:
            if len(cluster) == 1:
                compressed.append(cluster[0])
            else:
                # Sort cluster by relevance and recency
                cluster.sort(key=lambda x: (x.get("relevance", 0), x.get("original_index", 0)), reverse=True)
                
                # Choose the most relevant message or merge if needed
                compressed.append(cluster[0])
        
        # Sort back by original order
        compressed.sort(key=lambda x: x.get("original_index", 0))
        
        return compressed
    
    def optimize_prompt(self, prompt: str, max_tokens: int, preserve_end: bool = True) -> str:
        """
        Optimize a prompt to fit within token limits while preserving its effectiveness
        
        Args:
            prompt: The prompt to optimize
            max_tokens: Maximum tokens allowed
            preserve_end: Whether to preserve the end of the prompt (usually instructions)
            
        Returns:
            Optimized prompt
        """
        current_count = self.count_tokens(prompt)
        if current_count <= max_tokens:
            return prompt
            
        # Split prompt into sections based on double newlines (paragraph breaks)
        sections = re.split(r'\n\s*\n', prompt)
        
        # Identify instruction sections which typically appear at the end
        instruction_markers = ["instruction", "task", "answer", "respond", "follow", "guidelines",
                               "step", "requirements", "format", "note", "important"]
        
        instruction_sections = []
        content_sections = []
        
        for section in sections:
            is_instruction = False
            for marker in instruction_markers:
                if marker.lower() in section.lower():
                    is_instruction = True
                    break
                    
            if is_instruction:
                instruction_sections.append(section)
            else:
                content_sections.append(section)
        
        # Always keep instructions if preserve_end is True
        must_keep = []
        may_compress = []
        
        if preserve_end and instruction_sections:
            must_keep = instruction_sections
            may_compress = content_sections
        else:
            # Keep the first and last section, compress the middle
            if sections:
                must_keep = [sections[0]]
                if len(sections) > 1:
                    must_keep.append(sections[-1])
                if len(sections) > 2:
                    may_compress = sections[1:-1]
        
        # Calculate token counts
        must_keep_text = '\n\n'.join(must_keep)
        must_keep_tokens = self.count_tokens(must_keep_text)
        
        # If must-keep sections already exceed limit, fall back to simple truncation
        if must_keep_tokens >= max_tokens:
            if preserve_end:
                # Preserve the end with higher priority
                ending_portion = int(max_tokens * 0.7)  # 70% for ending
                beginning_portion = max_tokens - ending_portion
                
                beginning = self.truncate_text(sections[0], beginning_portion)
                ending = self.truncate_text('\n\n'.join(instruction_sections), ending_portion)
                
                return beginning + "\n\n" + ending
            else:
                # Simple truncation as fallback
                return self.truncate_text(prompt, max_tokens)
        
        # Compress the middle sections to fit
        remaining_tokens = max_tokens - must_keep_tokens
        if not may_compress or remaining_tokens <= 0:
            return must_keep_text
            
        # Simple compression: keep as many sections as we can fit
        compressed_middle = ""
        tokens_used = 0
        for section in may_compress:
            section_tokens = self.count_tokens(section) + 2  # +2 for paragraph break
            if tokens_used + section_tokens <= remaining_tokens:
                if compressed_middle:
                    compressed_middle += "\n\n"
                compressed_middle += section
                tokens_used += section_tokens
            else:
                # Try to fit part of the last section
                if remaining_tokens - tokens_used > 20:  # Only if reasonable space remains
                    partial_section = self.truncate_text(section, remaining_tokens - tokens_used - 2)
                    if compressed_middle:
                        compressed_middle += "\n\n"
                    compressed_middle += partial_section
                break
        
        # Reassemble the prompt
        if preserve_end:
            return content_sections[0] + "\n\n" + compressed_middle + "\n\n" + '\n\n'.join(must_keep)
        else:
            return must_keep[0] + "\n\n" + compressed_middle + "\n\n" + must_keep[1]

# Create a singleton instance for global use
token_optimizer = TokenOptimizer() 