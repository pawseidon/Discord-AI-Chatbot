class AIProvider:
    # ... existing code ...
    
    async def generate_text(self, messages=None, prompt=None, **kwargs):
        """
        Generate text using the LLM provider
        
        This is an alias for async_call to maintain compatibility with different code patterns
        
        Args:
            messages: List of message objects
            prompt: String prompt (alternative to messages)
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Generated text
        """
        if messages:
            return await self.async_call(messages=messages, **kwargs)
        elif prompt:
            return await self.async_call(prompt=prompt, **kwargs)
        else:
            raise ValueError("Either messages or prompt must be provided") 