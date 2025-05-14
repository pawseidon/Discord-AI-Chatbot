"""
Message Service

This module provides utilities for message formatting, sending, and processing
to ensure consistent handling of Discord messages.
"""

import logging
import asyncio
import re
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import discord
from discord.ext import commands

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('message_service')

# Regular expressions for detecting different types of content
CODE_BLOCK_REGEX = r"```(?:([\w+]+)\n)?([\s\S]*?)```"
INLINE_CODE_REGEX = r"`([^`]+)`"
URL_REGEX = r"https?://[^\s)>]+"
BULLET_LIST_REGEX = r"(?:^|\n)[\*\-\+•] .+"

class MessageService:
    """Service for handling Discord messages and responses"""
    
    @staticmethod
    async def smart_mention(content: str, message: discord.Message, bot) -> str:
        """
        Clean up message content by removing bot mentions
        
        Args:
            content: The message content
            message: The Discord message
            bot: The Discord bot instance
            
        Returns:
            Cleaned message content
        """
        # Remove bot mention from message
        if bot.user.mentioned_in(message):
            # Replace all forms of bot mentions
            content = re.sub(f'<@!?{bot.user.id}>', '', content).strip()
            content = re.sub(f'@{bot.user.name}', '', content).strip()
        
        return content

    @staticmethod
    def process_code_block(match):
        """Process a code block match and format it properly"""
        # Different patterns have different group structures
        if len(match.groups()) > 1:
            lang, code = match.groups()
            
            # If no language is specified, try to detect common languages
            if not lang:
                # Detect language based on common patterns
                if re.search(r"function|const|let|var|=>", code):
                    lang = "javascript"
                elif re.search(r"def |class |import |from |if __name__", code):
                    lang = "python"
                elif re.search(r"<html|<div|<body|<script", code):
                    lang = "html"
                elif re.search(r"\{.*\:.*\}", code) and not re.search(r"[\w]+\s*\(.*\)\s*\{", code):
                    # Likely JSON but not a JS/C/Java function
                    try:
                        json.loads(code.strip())
                        lang = "json"
                    except:
                        pass
            
            # Return formatted code block
            if lang:
                return f"```{lang}\n{code}```"
            return f"```\n{code}```"
        else:
            # Single backtick inline code
            code = match.group(1)
            return f"`{code}`"

    @staticmethod
    async def format_code_blocks(text):
        """
        Format code blocks with proper syntax highlighting
        
        Args:
            text: The text containing code blocks
            
        Returns:
            Formatted text with improved code blocks
        """
        # Patterns to match various code block formats
        patterns = [
            r"```([a-zA-Z0-9]+)\n([\s\S]*?)```",  # ```language\ncode```
            r"```([\s\S]*?)```",                  # ```code```
            r"`([^`]+)`"                          # `code`
        ]
        
        # Process each pattern
        for pattern in patterns:
            text = re.sub(pattern, lambda m: MessageService.process_code_block(m), text)
        
        return text

    @staticmethod
    async def detect_content_type(response):
        """
        Detect the primary content type in a response
        
        Args:
            response: The response text
            
        Returns:
            str: One of 'code', 'list', 'url', 'table', 'normal'
        """
        # Check for code blocks
        code_blocks = re.findall(CODE_BLOCK_REGEX, response)
        if code_blocks:
            return 'code'
        
        # Check for bulleted lists
        bullet_lists = re.findall(BULLET_LIST_REGEX, response)
        if bullet_lists and len(bullet_lists) > 2:  # At least 3 bullet points
            return 'list'
        
        # Check for URLs
        urls = re.findall(URL_REGEX, response)
        if urls and len(urls) > 2:  # Multiple URLs
            return 'url'
        
        # Check for table-like content
        if '|' in response and '-+-' in response:
            return 'table'
        
        # Default to normal text
        return 'normal'

    @staticmethod
    async def create_embed_for_response(response, author_name=None, avatar_url=None):
        """
        Create an appropriate embed based on the response content
        
        Args:
            response: The response text
            author_name: Optional author name for the embed
            avatar_url: Optional avatar URL for the embed
            
        Returns:
            discord.Embed or None if embedding is not appropriate
        """
        content_type = await MessageService.detect_content_type(response)
        
        # Create different embed types based on content
        embed = discord.Embed()
        
        if author_name:
            embed.set_author(name=author_name, icon_url=avatar_url if avatar_url else discord.Embed.Empty)
        
        if content_type == 'code':
            embed.title = "Code Snippet"
            embed.color = discord.Color.blue()
            # Code is better displayed in the regular message with formatting
            return None
        
        elif content_type == 'list':
            embed.title = "Information"
            embed.description = response[:4000]  # Discord embed description limit
            embed.color = discord.Color.green()
            
        elif content_type == 'url':
            embed.title = "Resources"
            embed.description = response[:4000]
            embed.color = discord.Color.gold()
            
            # Extract and add the first URL as an embed URL if present
            urls = re.findall(URL_REGEX, response)
            if urls:
                embed.url = urls[0]
        
        elif content_type == 'table':
            embed.title = "Data Table"
            # Tables might look better in regular text with code formatting
            return None
            
        else:  # Normal text
            # For short responses, no need for an embed
            if len(response) < 200:
                return None
                
            embed.description = response[:4000]
            embed.color = discord.Color.light_grey()
        
        return embed
    
    @staticmethod
    async def update_message(message: discord.Message, content: str, **kwargs) -> List[discord.Message]:
        """
        Update an existing message with new content, handling long messages
        
        Args:
            message: The Discord message to update
            content: The new content
            **kwargs: Additional arguments to pass to edit()
            
        Returns:
            List of all messages (original and any additional ones)
        """
        try:
            all_messages = [message]
            
            # Split long messages if needed
            if len(content) > 2000:
                chunks = await MessageService.split_message(content)
                
                # Update the original message with the first chunk
                await message.edit(content=chunks[0], **kwargs)
                
                # Send additional chunks as new messages
                channel = message.channel
                for chunk in chunks[1:]:
                    new_message = await channel.send(chunk)
                    all_messages.append(new_message)
                    
                return all_messages
            else:
                # Update as a single message
                await message.edit(content=content, **kwargs)
                return all_messages
                    
        except discord.HTTPException as e:
            logger.error(f"Error updating message: {e}")
            # Try a simpler message if the original failed
            error_msg = "I encountered an error updating my response. Please try again."
            await message.edit(content=error_msg)
            return [message]
    
    @staticmethod
    async def send_response(ctx_or_message: Union[commands.Context, discord.Message], content: str, **kwargs) -> discord.Message:
        """
        Send a response message with proper error handling
        
        Args:
            ctx_or_message: The message context or Discord message
            content: The content to send
            **kwargs: Additional arguments to pass to send()
            
        Returns:
            The sent Discord message
        """
        try:
            # Split long messages if needed
            if len(content) > 2000:
                chunks = await MessageService.split_message(content)
                messages = []
                
                for i, chunk in enumerate(chunks):
                    # Send only the first chunk with reference and other kwargs
                    if i == 0:
                        if isinstance(ctx_or_message, commands.Context):
                            msg = await ctx_or_message.send(chunk, **kwargs)
                        else:
                            msg = await ctx_or_message.channel.send(chunk, reference=ctx_or_message, **kwargs)
                        messages.append(msg)
                    else:
                        # Send subsequent chunks as follow-ups
                        if isinstance(ctx_or_message, commands.Context):
                            msg = await ctx_or_message.send(chunk)
                        else:
                            msg = await ctx_or_message.channel.send(chunk)
                        messages.append(msg)
                
                return messages[0]  # Return the first message for reference
            else:
                # Send as a single message
                if isinstance(ctx_or_message, commands.Context):
                    return await ctx_or_message.send(content, **kwargs)
                else:
                    return await ctx_or_message.channel.send(content, reference=ctx_or_message, **kwargs)
                    
        except discord.HTTPException as e:
            logger.error(f"Error sending message: {e}")
            # Try a simpler message if the original failed
            error_msg = "I encountered an error sending my response. Please try again."
            
            if isinstance(ctx_or_message, commands.Context):
                return await ctx_or_message.send(error_msg)
            else:
                return await ctx_or_message.channel.send(error_msg, reference=ctx_or_message)
    
    @staticmethod
    async def send_typing_indicator(ctx_or_message: Union[commands.Context, discord.Message], duration: float = 1.0) -> None:
        """
        Send a typing indicator for a specified duration
        
        Args:
            ctx_or_message: The message context or Discord message
            duration: Duration to show typing indicator in seconds
        """
        try:
            if isinstance(ctx_or_message, commands.Context):
                channel = ctx_or_message.channel
            else:
                channel = ctx_or_message.channel
                
            async with channel.typing():
                await asyncio.sleep(duration)
                
        except Exception as e:
            # Just log the error without disrupting the flow
            logger.error(f"Error sending typing indicator: {e}")
    
    @staticmethod
    async def split_message(content: str, max_length: int = 2000) -> List[str]:
        """
        Split a long message into chunks that fit within Discord's message limit
        
        Args:
            content: The message content to split
            max_length: Maximum length of each chunk
            
        Returns:
            List of message chunks
        """
        # Check if content is already within limits
        if len(content) <= max_length:
            return [content]
            
        chunks = []
        current_chunk = ""
        
        # Split by lines first to maintain formatting
        lines = content.split('\n')
        
        for line in lines:
            # If the line itself is too long, split it
            if len(line) > max_length:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split the long line
                line_parts = [line[i:i + max_length] for i in range(0, len(line), max_length)]
                chunks.extend(line_parts[:-1])  # Add all but the last part
                current_chunk = line_parts[-1]  # Start a new chunk with the last part
            
            # If adding this line would make the chunk too long
            elif len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            
            # Otherwise, add the line to the current chunk
            else:
                if current_chunk:
                    current_chunk += '\n'
                current_chunk += line
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    @staticmethod
    async def format_with_agent_emoji(response: str, agent_type: str) -> Tuple[str, Optional[str]]:
        """
        Format a response with an agent emoji if available
        
        Args:
            response: The message content
            agent_type: The type of agent that generated the response
            
        Returns:
            Tuple of (formatted_response, emoji)
        """
        # Use lazy import to avoid circular reference
        emoji = None
        
        # Define common emoji mapping for basic agent types
        emoji_map = {
            "conversational": "💬",
            "rag": "📚",
            "sequential": "🔄",
            "knowledge": "🧠",
            "verification": "✅",
            "creative": "🎨",
            "calculation": "🧮",
            "planning": "📝",
            "graph": "📊",
            "multi_agent": "👥",
            "react": "⚡",
            "cot": "🔍",
            "step_back": "🔙"
        }
        
        # Try to get the emoji from mapping first
        if agent_type in emoji_map:
            emoji = emoji_map[agent_type]
        else:
            # Fall back to agent_service if the type isn't in our map
            try:
                from ..services.agent_service import agent_service
                emoji = await agent_service.get_agent_emoji(agent_type)
            except (ImportError, AttributeError):
                # If agent_service is not available, use default emoji
                emoji = "🤖"
        
        if emoji:
            return f"{emoji} {response}", emoji
        else:
            return response, None

# Create a singleton instance for global access
message_service = MessageService() 