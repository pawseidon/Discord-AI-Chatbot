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
    """
    Service for formatting, processing, and managing Discord message responses
    """
    
    @staticmethod
    async def smart_mention(content: str, message: discord.Message, bot) -> str:
        """
        Clean up message content by removing bot mentions and persona prefixes
        
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
        
        # Get the bot's possible persona names from the configuration
        from bot_utilities.ai_utils import get_bot_names_and_triggers
        bot_info = get_bot_names_and_triggers()
        bot_names = bot_info["names"]
        trigger_words = bot_info["triggers"]
        
        # Get bot's current name and nickname in the server
        bot_username = bot.user.name.lower() if bot.user else "assistant"
        bot_nickname = None
        if message.guild and message.guild.me:
            bot_nickname = message.guild.me.display_name.lower()
        
        # Add bot's actual name/nickname to the names list
        if bot_username and bot_username not in bot_names:
            bot_names.append(bot_username)
        if bot_nickname and bot_nickname not in bot_names and bot_nickname != bot_username:
            bot_names.append(bot_nickname)
        
        # Process dynamic triggers
        actual_triggers = []
        for trigger in trigger_words:
            if trigger == "%BOT_NAME%":
                actual_triggers.append(bot_username)
            elif trigger == "%BOT_NICKNAME%" and bot_nickname:
                actual_triggers.append(bot_nickname)
            elif trigger == "%BOT_USERNAME%":
                actual_triggers.append(bot_username)
            else:
                actual_triggers.append(trigger)
        
        # Add triggers to names list
        for trigger in actual_triggers:
            if trigger and trigger not in bot_names:
                bot_names.append(trigger)
        
        # Remove persona prefixes (like "Hand" or other persona names)
        content_lower = content.lower()
        # First check for exact matches at the beginning of the content
        for name in bot_names:
            name_lower = name.lower()
            # If the content starts with the name exactly (case insensitive)
            if content_lower.startswith(name_lower):
                # Remove it from the content (preserve case in the rest of the content)
                content = content[len(name_lower):].strip()
                break
            
        # Remove common prefixes that might be used to address the bot
        prefixes = ["hey", "hi", "hello", "ok", "okay", "help", "tell", "ask"]
        for prefix in prefixes:
            prefix_pattern = f"^{prefix}\\s+(?:{bot.user.name}\\s+)?"
            content = re.sub(prefix_pattern, '', content, flags=re.IGNORECASE).strip()
        
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
    async def safe_edit_message(message, content=None, embed=None):
        """
        Safely edit a message, handling the case where the message may have been deleted
        
        Args:
            message: The Discord message to edit
            content: The new content
            embed: An optional embed to include
            
        Returns:
            bool: Whether the edit was successful
        """
        try:
            await message.edit(content=content, embed=embed)
            return True
        except discord.NotFound:
            logging.warning(f"Attempted to edit a deleted message (ID: {message.id})")
            return False
        except discord.HTTPException as e:
            logging.error(f"Error editing message: {str(e)}")
            return False
            
    @staticmethod
    async def send_response(message, content, update_existing=False):
        """
        Send a response to a message, handling scenarios like long messages
        
        Args:
            message: The Discord message to respond to or update
            content: The content to send
            update_existing: Whether to update an existing message
            
        Returns:
            The sent or updated message
        """
        try:
            if update_existing:
                return await MessageService.safe_edit_message(message, content=content)
            else:
                return await message.channel.send(content)
        except Exception as e:
            logging.error(f"Error sending response: {str(e)}")
            return None
    
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
        Split a long message into chunks while preserving formatting
        
        Args:
            content: The message content to split
            max_length: Maximum length of each chunk
            
        Returns:
            List of message chunks
        """
        if len(content) <= max_length:
            return [content]
            
        chunks = []
        current_chunk = ""
        
        # Track code block and formatting states
        in_code_block = False
        code_block_lang = ""
        code_block_content = ""
        
        lines = content.split('\n')
        
        for line in lines:
            # Check for code block delimiters
            if line.startswith('```'):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                    # Add closing backticks
                    if len(current_chunk) + len(line) <= max_length:
                        current_chunk += line + '\n'
                    else:
                        # If chunk would exceed max length, start a new chunk
                        # but make sure to close the code block in the current chunk
                        # and open it in the new chunk
                        chunks.append(current_chunk + "```\n")
                        current_chunk = f"```{code_block_lang}\n{line[3:]}\n"
                else:
                    # Start of code block
                    in_code_block = True
                    # Extract language if specified
                    if len(line) > 3:
                        code_block_lang = line[3:].strip()
                    else:
                        code_block_lang = ""
                    
                    # Check if adding this line would exceed the max length
                    if len(current_chunk) + len(line) <= max_length:
                        current_chunk += line + '\n'
                    else:
                        chunks.append(current_chunk)
                        current_chunk = line + '\n'
            else:
                # Regular line, check if it fits in the current chunk
                if len(current_chunk) + len(line) + 1 <= max_length:  # +1 for newline
                    current_chunk += line + '\n'
                else:
                    # Line won't fit, check if we're in a code block
                    if in_code_block:
                        # Need to close code block in current chunk and open in next
                        chunks.append(current_chunk + "```\n")
                        current_chunk = f"```{code_block_lang}\n{line}\n"
                    else:
                        # Normal text, just create a new chunk
                        chunks.append(current_chunk)
                        current_chunk = line + '\n'
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        # Final pass to ensure no chunk exceeds the max length
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Chunk is still too long, split at max_length
                # This is a safety fallback and should rarely happen
                for i in range(0, len(chunk), max_length):
                    final_chunks.append(chunk[i:i+max_length])
        
        return final_chunks
    
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

    async def process_response(self, 
                           message_content: str, 
                           update_data: Dict[str, Any] = None,
                           is_thinking: bool = False,
                           is_error: bool = False,
                           is_final: bool = False,
                           reasoning_type: str = None,
                           is_combined: bool = False,
                           clarification_data: Dict[str, Any] = None) -> str:
        """
        Process a response message with optional styling and formatting
        
        Args:
            message_content: The content of the message
            update_data: Additional data for updating the message
            is_thinking: Whether this is a thinking message
            is_error: Whether this is an error message
            is_final: Whether this is the final response
            reasoning_type: The reasoning type being used
            is_combined: Whether multiple reasoning types are being used
            clarification_data: Data related to clarification request
            
        Returns:
            str: The formatted message
        """
        # If this is a clarification request, format it differently
        if clarification_data:
            original_query = clarification_data.get("original_query", "")
            reason = clarification_data.get("reason", "")
            
            clarification_message = "**🔍 I need some clarification:**\n\n"
            clarification_message += f"{message_content}\n\n"
            
            if reason:
                clarification_message += f"*Reason: {reason}*\n\n"
            
            return clarification_message
        
        # Process thinking message
        if is_thinking:
            thinking_content = message_content
            if update_data and "thinking" in update_data:
                thinking_content = update_data["thinking"]
            
            thinking_message = f"*{thinking_content}*"
            return thinking_message
        
        # Process error message
        if is_error:
            error_message = f"⚠️ **Error**: {message_content}"
            return error_message
        
        # Add reasoning indicators if available
        if reasoning_type:
            reasoning_emoji = self._get_reasoning_emoji(reasoning_type)
            if is_combined:
                return f"{reasoning_emoji} **Combined Reasoning**: {message_content}"
            else:
                return f"{reasoning_emoji} {message_content}"
        
        # Default response
        return message_content
    
    def _get_reasoning_emoji(self, reasoning_type: str) -> str:
        """
        Get the emoji for a specific reasoning type
        
        Args:
            reasoning_type: The type of reasoning
            
        Returns:
            str: The corresponding emoji
        """
        reasoning_emojis = {
            "rag": "📚",
            "sequential": "🔄",
            "verification": "✅",
            "multi_agent": "👥",
            "graph": "📊",
            "symbolic": "🧮",
            "creative": "🎨",
            "step_back": "🔍",
            "chain_of_thought": "⛓️",
            "contextual": "👤",
            "detail_analysis": "🔎",
            "component_breakdown": "🧩"
        }
        
        return reasoning_emojis.get(reasoning_type, "🤖")
    
    async def handle_update_callback(self, 
                                  update_type: str, 
                                  data: Dict[str, Any],
                                  message_obj: discord.Message = None) -> str:
        """
        Handle updates from the workflow service
        
        Args:
            update_type: The type of update
            data: The update data
            message_obj: The Discord message object to update
            
        Returns:
            str: The formatted update message
        """
        if update_type == "thinking":
            thinking_message = await self.process_response("", data, is_thinking=True)
            if message_obj:
                await self.safe_edit_message(message_obj, content=thinking_message)
            return thinking_message
        
        elif update_type == "reasoning_switch":
            reasoning_types = data.get("reasoning_types", [])
            is_combined = data.get("is_combined", False)
            workflow = data.get("workflow", None)
            
            # Format based on the reasoning type
            if reasoning_types:
                if is_combined and len(reasoning_types) > 1:
                    # For combined reasoning, show the sequence
                    emoji_sequence = " → ".join([self._get_reasoning_emoji(r) for r in reasoning_types])
                    switch_message = f"*Using {emoji_sequence} reasoning workflow*"
                    
                    if workflow:
                        workflow_name = workflow.replace("_", " ").title()
                        switch_message = f"*Using {emoji_sequence} reasoning workflow ({workflow_name})*"
                else:
                    # For single reasoning type
                    reasoning_type = reasoning_types[0]
                    emoji = self._get_reasoning_emoji(reasoning_type)
                    reasoning_name = reasoning_type.replace("_", " ").title()
                    switch_message = f"*Using {emoji} {reasoning_name} reasoning*"
                
                if message_obj:
                    await self.safe_edit_message(message_obj, content=switch_message)
                return switch_message
            
            # If no reasoning types are specified, return empty string
            return ""
        
        elif update_type == "clarification_needed":
            message = data.get("message", "Could you please clarify your question?")
            original_query = data.get("original_query", "")
            reason = data.get("reason", "")
            
            clarification_message = await self.process_response(
                message, 
                clarification_data={
                    "original_query": original_query,
                    "reason": reason
                }
            )
            
            if message_obj:
                await self.safe_edit_message(message_obj, content=clarification_message)
            return clarification_message
        
        # Default empty response
        return ""

# Create a singleton instance for global access
message_service = MessageService() 