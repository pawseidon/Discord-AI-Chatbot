import discord
from discord.ext import commands
import asyncio
import json
import os
import re
import time

from bot_utilities.response_utils import split_response
from bot_utilities.ai_utils import generate_response, get_crypto_price, text_to_speech 
from bot_utilities.memory_utils import UserPreferences, process_conversation_history, get_enhanced_instructions
from bot_utilities.news_utils import get_news_context
from bot_utilities.formatting_utils import format_response_for_discord, find_and_format_user_mentions, chunk_message
from bot_utilities.config_loader import config, load_active_channels
from ..common import allow_dm, trigger_words, replied_messages, smart_mention, message_history, MAX_HISTORY, instructions


class OnMessage(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_channels = load_active_channels
        self.instructions = instructions
        self.thread_contexts = {}  # Store parent message context for threads
        self.STREAM_UPDATE_INTERVAL = 0.5  # Update streaming messages every 0.5 seconds

    async def process_message(self, message):
        active_channels = self.active_channels()
        string_channel_id = f"{message.channel.id}"
        
        # Improve reply detection - check if message is a reply to any message from the bot
        is_replied = False
        if message.reference and message.reference.resolved:
            # Check if the message being replied to is from the bot
            if message.reference.resolved.author.id == self.bot.user.id:
                is_replied = True
                
        is_dm_channel = isinstance(message.channel, discord.DMChannel)
        is_active_channel = string_channel_id in active_channels
        is_allowed_dm = allow_dm and is_dm_channel
        contains_trigger_word = any(word in message.content for word in trigger_words)
        is_bot_mentioned = self.bot.user.mentioned_in(message) and smart_mention and not message.mention_everyone
        bot_name_in_message = self.bot.user.name.lower() in message.content.lower() and smart_mention

        if not (is_active_channel or is_allowed_dm or contains_trigger_word or is_bot_mentioned or is_replied or bot_name_in_message):
            return

        instruc_config = active_channels.get(string_channel_id, config['DEFAULT_INSTRUCTION'])
        
        # Build enhanced instructions with channel context
        channel_context = ""
        
        # Check if we're in a thread
        parent_message = None
        if isinstance(message.channel, discord.Thread):
            try:
                # Try to get thread parent message for context
                parent_message = await message.channel.parent.fetch_message(message.channel.id)
                thread_key = f"thread-{message.channel.id}"
                
                if thread_key in self.thread_contexts:
                    # Use stored thread context
                    thread_context = self.thread_contexts[thread_key]
                    channel_context = f" This conversation is happening in a thread titled '{message.channel.name}'. The thread was started with this context: {thread_context}"
                elif parent_message and parent_message.content:
                    # Store new thread context
                    self.thread_contexts[thread_key] = parent_message.content
                    channel_context = f" This conversation is happening in a thread titled '{message.channel.name}'. The thread was started with this message: {parent_message.content}"
            except:
                # If we can't get the parent message, just use thread name
                channel_context = f" This conversation is happening in a thread titled '{message.channel.name}'."
        elif message.channel.name:
            channel_context = f" The current channel name is '{message.channel.name}', but remember to focus on directly answering the user's question regardless of the channel name."
        
        base_instruction_text = f"{self.instructions[instruc_config]}{channel_context}"
        
        # Enhance instructions with user preferences and personalization
        instruction_text = get_enhanced_instructions(base_instruction_text, message.author.id)
        
        channel_id = message.channel.id
        key = f"{message.author.id}-{channel_id}"
        
        # Initialize or get existing history
        message_history[key] = message_history.get(key, [])
        message_history[key] = message_history[key][-MAX_HISTORY:]
        message_history[key].append({"role": "user", "content": message.content})

        # Process and potentially summarize conversation history
        await process_conversation_history(message_history, message.author.id, channel_id)

        # Check user preferences for voice response
        user_prefs = await UserPreferences.get_user_preferences(message.author.id)
        use_voice = user_prefs.get("use_voice", False)
        
        # Get user preference for streaming
        use_streaming = user_prefs.get("use_streaming", True)  # Default to True for better user experience

        # Check if user specifically asked to create a thread
        thread_request_phrases = [
            "create a thread", "make a thread", "start a thread", 
            "open a thread", "new thread", "create thread", 
            "in a thread", "as a thread"
        ]
        
        should_create_thread = False
        
        # Only create a thread if the user specifically requests one
        if not isinstance(message.channel, discord.Thread) and not isinstance(message.channel, discord.DMChannel):
            message_lower = message.content.lower()
            should_create_thread = any(phrase in message_lower for phrase in thread_request_phrases)

        # Add typing indicator
        async with message.channel.typing():
            # Traditional non-streaming response by default
            use_streaming_for_this_message = use_streaming
            
            try:
                if use_streaming_for_this_message:
                    # Send an initial message that will be updated with streamed content
                    initial_message = await message.channel.send("_Thinking..._")
                    
                    # Get streaming response
                    response_stream = await self.generate_response(instruction_text, message_history[key], stream=True)
                    
                    # Process the stream and update the message
                    full_response = await self.process_streaming_response(initial_message, response_stream)
                    
                    # Add the response to message history
                    message_history[key].append({"role": "assistant", "content": full_response})
                    
                    # Create thread if requested
                    if should_create_thread and initial_message and hasattr(message.channel, 'create_thread'):
                        try:
                            thread_name = message.content[:95] + "..." if len(message.content) > 95 else message.content
                            thread = await initial_message.create_thread(name=thread_name)
                            await thread.send(f"I've created this thread as requested. Feel free to continue our conversation here!")
                            
                            # Store thread context
                            thread_key = f"thread-{thread.id}"
                            self.thread_contexts[thread_key] = message.content
                        except Exception as e:
                            print(f"Error creating thread: {e}")
                else:
                    # Traditional non-streaming response
                    response = await self.generate_response(instruction_text, message_history[key])
                    message_history[key].append({"role": "assistant", "content": response})
                    
                    # Send response and potentially create thread if requested
                    first_message = await self.send_response(message, response, use_voice=use_voice, create_thread=should_create_thread)
                    
                    # If we created a thread, store the context
                    if should_create_thread and first_message and hasattr(first_message, 'thread'):
                        thread_key = f"thread-{first_message.thread.id}"
                        self.thread_contexts[thread_key] = message.content
            except Exception as e:
                # If any error occurs, fall back to non-streaming response
                print(f"Error in message processing: {e}")
                try:
                    # Try again with non-streaming
                    response = await self.generate_response(instruction_text, message_history[key], stream=False)
                    message_history[key].append({"role": "assistant", "content": response})
                    
                    # Send response and potentially create thread if requested
                    first_message = await self.send_response(message, response, use_voice=use_voice, create_thread=should_create_thread)
                    
                    # If we created a thread, store the context
                    if should_create_thread and first_message and hasattr(first_message, 'thread'):
                        thread_key = f"thread-{first_message.thread.id}"
                        self.thread_contexts[thread_key] = message.content
                except Exception as e2:
                    print(f"Critical error in fallback response: {e2}")
                    await message.channel.send("I'm experiencing technical difficulties at the moment. Please try again later.")

    async def generate_response(self, instructions, history, stream=False):
        return await generate_response(instructions=instructions, history=history, stream=stream)
    
    async def process_streaming_response(self, initial_message, response_stream):
        """Process a streaming response, updating the message periodically"""
        
        accumulated_response = ""
        buffer = ""
        last_update_time = time.time()
        
        # Collect response chunks
        try:
            async for chunk in response_stream:
                buffer += chunk
                accumulated_response += chunk
                
                # Update message periodically to avoid rate limits
                current_time = time.time()
                if current_time - last_update_time >= self.STREAM_UPDATE_INTERVAL:
                    try:
                        # Format the response for display
                        if len(buffer) > 0:
                            # Keep it under Discord's character limit
                            content = buffer[:1990] if len(buffer) > 1990 else buffer
                            await initial_message.edit(content=content)
                            last_update_time = current_time
                    except Exception as e:
                        print(f"Error updating message: {e}")
        except Exception as e:
            print(f"Error processing stream: {e}")
            # If we have no accumulated response at all, use a fallback message
            if not accumulated_response:
                accumulated_response = "I encountered an error while generating a response. Please try again."
                try:
                    await initial_message.edit(content=accumulated_response)
                except:
                    pass
        
        # Ensure we do a final update with the complete response
        try:
            # Check if we have any accumulated response
            if not accumulated_response:
                accumulated_response = "I'm experiencing technical difficulties. Please try again later."
                await initial_message.edit(content=accumulated_response)
                return accumulated_response
                
            # Need to format final response properly
            user_id = str(initial_message.reference.resolved.author.id) if initial_message.reference and initial_message.reference.resolved else None
            user_prefs = await UserPreferences.get_user_preferences(user_id) if user_id else {}
            use_embeds = user_prefs.get('use_embeds', False)
            
            # Process mentions in the response before formatting
            if initial_message.reference and initial_message.reference.resolved:
                formatted_response = find_and_format_user_mentions(initial_message.reference.resolved, accumulated_response, self.bot)
            else:
                formatted_response = accumulated_response
                
            # Format the response for Discord
            content, embed, chunks = format_response_for_discord(
                formatted_response,
                use_embeds,
                author_name=initial_message.reference.resolved.author.display_name if initial_message.reference and initial_message.reference.resolved else "User",
                avatar_url=initial_message.reference.resolved.author.avatar.url if initial_message.reference and initial_message.reference.resolved and initial_message.reference.resolved.author.avatar else None
            )
            
            # Update the message with the formatted response
            if embed:
                await initial_message.edit(content=content, embed=embed)
            else:
                # If the response is in chunks, handle them appropriately
                if chunks:
                    channel = initial_message.channel
                    
                    # Update the initial message with the first chunk
                    await initial_message.edit(content=chunks[0])
                    
                    # Send additional chunks as separate messages
                    for chunk in chunks[1:]:
                        await channel.send(content=chunk)
                else:
                    # Single response within limits
                    await initial_message.edit(content=content)
            
            return accumulated_response
        
        except Exception as e:
            print(f"Error finalizing streamed response: {e}")
            
            # Fallback to simple chunking if anything fails
            try:
                # If response is still too long, chunk it
                if len(accumulated_response) > 2000:
                    chunks = chunk_message(accumulated_response)
                    
                    # Update initial message with first chunk
                    await initial_message.edit(content=chunks[0])
                    
                    # Send additional chunks
                    for chunk in chunks[1:]:
                        await initial_message.channel.send(content=chunk)
                else:
                    # Just update with what we have (truncated if needed)
                    await initial_message.edit(content=accumulated_response[:2000])
            except:
                # Last resort fallback
                try:
                    await initial_message.edit(content="I was unable to complete my response. Please try again.")
                except:
                    pass
            
            return accumulated_response

    async def send_response(self, message, response, use_voice=False, create_thread=False):
        """
        Send the AI response to the user
        
        Args:
            message (discord.Message): The original message
            response (str): The AI response
            use_voice (bool): Whether to send a voice response
            create_thread (bool): Whether to create a thread
            
        Returns:
            discord.Message: The first message sent in response
        """
        first_message = None
        
        try:
            # Get user preferences
            user_prefs = await UserPreferences.get_user_preferences(message.author.id)
            use_embeds = user_prefs.get('use_embeds', False)
            
            # Format response for Discord
            response = find_and_format_user_mentions(message, response, self.bot)
            content, embed, chunks = format_response_for_discord(response, use_embeds, 
                                                        author_name=message.author.display_name, 
                                                        avatar_url=message.author.avatar.url if message.author.avatar else None)
            
            # If the response is in chunks, send each chunk sequentially
            if chunks:
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        # First message
                        if use_voice:
                            # Create voice response in the background
                            voice_task = asyncio.create_task(self.create_voice_response(response))
                            
                        # Send the first message
                        if embed:
                            first_message = await message.channel.send(embed=embed)
                        else:
                            first_message = await message.channel.send(content=chunk)
                            
                        # Create thread if requested and possible
                        if create_thread and first_message and hasattr(message.channel, 'create_thread'):
                            try:
                                thread_name = message.content[:95] + "..." if len(message.content) > 95 else message.content
                                await first_message.create_thread(name=thread_name)
                            except Exception as e:
                                print(f"Error creating thread: {e}")
                        
                        # Send voice file if available
                        if use_voice:
                            try:
                                voice_file = await voice_task
                                if voice_file:
                                    await message.channel.send(file=discord.File(voice_file, filename="response.mp3"))
                            except Exception as e:
                                print(f"Error sending voice response: {e}")
                    else:
                        # Additional chunks
                        await message.channel.send(content=chunk)
            else:
                # Single message response
                if use_voice:
                    # Create voice response in the background
                    voice_task = asyncio.create_task(self.create_voice_response(response))
                    
                # Send the message
                if embed:
                    first_message = await message.channel.send(embed=embed)
                else:
                    first_message = await message.channel.send(content=content)
                    
                # Create thread if requested and possible
                if create_thread and first_message and hasattr(message.channel, 'create_thread'):
                    try:
                        thread_name = message.content[:95] + "..." if len(message.content) > 95 else message.content
                        await first_message.create_thread(name=thread_name)
                    except Exception as e:
                        print(f"Error creating thread: {e}")
                
                # Send voice file if available
                if use_voice:
                    try:
                        voice_file = await voice_task
                        if voice_file:
                            await message.channel.send(file=discord.File(voice_file, filename="response.mp3"))
                    except Exception as e:
                        print(f"Error sending voice response: {e}")
            
            # Store the message ID for future reference
            replied_messages[message.id] = first_message.id if first_message else None
            
            return first_message
            
        except Exception as e:
            print(f"Error sending response: {e}")
            # Fallback to plain text if anything fails
            try:
                if len(response) > 2000:
                    chunks = chunk_message(response)
                    first_message = await message.channel.send(content=chunks[0])
                    for chunk in chunks[1:]:
                        await message.channel.send(content=chunk)
                else:
                    first_message = await message.channel.send(content=response[:2000])
                
                # Store the message ID for future reference
                replied_messages[message.id] = first_message.id if first_message else None
                
                return first_message
            except:
                print("Critical error sending any response")
                return None

    async def create_voice_response(self, text):
        """Create a voice file from the response text"""
        try:
            # Use a simpler first sentence for the voice response
            first_sentence = re.split(r'(?<=[.!?])\s+', text)[0]
            voice_text = first_sentence[:500]  # Limit to 500 chars
            return await text_to_speech(voice_text)
        except Exception as e:
            print(f"Error creating voice response: {e}")
            return None

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user and message.reference:
            replied_messages[message.reference.message_id] = message
            if len(replied_messages) > 5:
                oldest_message_id = min(replied_messages.keys())
                del replied_messages[oldest_message_id]

        if message.mentions:
            for mention in message.mentions:
                message.content = message.content.replace(f'<@{mention.id}>', f'{mention.display_name}')

        if message.stickers or message.author.bot or (message.reference and (message.reference.resolved.author != self.bot.user or message.reference.resolved.embeds)):
            return

        await self.process_message(message)

async def setup(bot):
    await bot.add_cog(OnMessage(bot))