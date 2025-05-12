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
from bot_utilities.formatting_utils import format_response_for_discord, find_and_format_user_mentions
from bot_utilities.config_loader import config, load_active_channels
from ..common import allow_dm, trigger_words, replied_messages, smart_mention, message_history, MAX_HISTORY, instructions


class OnMessage(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_channels = load_active_channels
        self.instructions = instructions
        self.thread_contexts = {}  # Store parent message context for threads

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

        # Add typing indicator
        async with message.channel.typing():
            response = await self.generate_response(instruction_text, message_history[key])

        message_history[key].append({"role": "assistant", "content": response})

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
            
        # Send response and potentially create thread if requested
        first_message = await self.send_response(message, response, use_voice=use_voice, create_thread=should_create_thread)
        
        # If we created a thread, store the context
        if should_create_thread and first_message and hasattr(first_message, 'thread'):
            thread_key = f"thread-{first_message.thread.id}"
            self.thread_contexts[thread_key] = message.content

    async def generate_response(self, instructions, history):
        return await generate_response(instructions=instructions, history=history)

    async def send_response(self, message, response, use_voice=False, create_thread=False):
        """Send a response, optionally creating a thread for complex responses"""
        bytes_obj = None
        
        # Only process text-to-speech if voice is enabled
        if use_voice:
            bytes_obj = await text_to_speech(response)
            
        author_voice_channel = None
        author_member = None
        
        # Check if the user is in a voice channel and voice is enabled
        if use_voice and message.guild:
            author_member = message.guild.get_member(message.author.id)
            if author_member and author_member.voice:
                author_voice_channel = author_member.voice.channel

        # Join voice channel and play synthesized speech if conditions are met
        if use_voice and bytes_obj and author_voice_channel:
            try:
                voice_client = await author_voice_channel.connect()
                audio_source = discord.FFmpegPCMAudio(bytes_obj)
                voice_client.play(audio_source)
                
                # Disconnect once audio is done playing
                while voice_client.is_playing():
                    await asyncio.sleep(1)
                await voice_client.disconnect()
            except Exception as e:
                print(f"Error during voice playback: {e}")
                
        # Get user preferences for formatting
        user_id = str(message.author.id)
        user_prefs = await UserPreferences.get_user_preferences(user_id)
        use_embeds = user_prefs.get('use_embeds', False)
        
        # Process mentions in the response before formatting
        formatted_response = find_and_format_user_mentions(message, response, self.bot)
        
        # Format the response for Discord
        content, embed = format_response_for_discord(
            formatted_response,
            use_embeds,
            author_name=message.author.display_name,
            avatar_url=message.author.avatar.url if message.author.avatar else None
        )
        
        try:
            if embed:
                # Send the embed
                first_message = await message.channel.send(content=content, embed=embed)
                
                # Create a thread if requested and we're in a guild channel
                if create_thread and first_message and hasattr(message.channel, 'create_thread') and not isinstance(message.channel, discord.Thread):
                    try:
                        thread_name = message.content[:95] + "..." if len(message.content) > 95 else message.content
                        thread = await first_message.create_thread(name=thread_name)
                        await thread.send(f"I've created this thread as requested. Feel free to continue our conversation here!")
                    except Exception as e:
                        print(f"Error creating thread: {e}")
                
                return
            else:
                # Handle content that is too long (should be split into chunks)
                if isinstance(content, list):
                    first_message = None
                    for i, chunk in enumerate(content):
                        if i == 0:
                            first_message = await message.channel.send(content=chunk)
                        else:
                            await message.channel.send(content=chunk)
                    
                    # Create a thread if requested and appropriate
                    if create_thread and first_message and hasattr(message.channel, 'create_thread') and not isinstance(message.channel, discord.Thread):
                        try:
                            thread_name = message.content[:95] + "..." if len(message.content) > 95 else message.content
                            thread = await first_message.create_thread(name=thread_name)
                            await thread.send(f"I've created this thread as requested. Feel free to continue our conversation here!")
                        except Exception as e:
                            print(f"Error creating thread: {e}")
                else:
                    # Send a regular message
                    await message.channel.send(content=content)
        except Exception as e:
            print(f"Error sending message: {e}")
            # Try to send a simplified message if the above fails
            await message.channel.send(f"I had trouble formatting my response. Here it is in plain text: {response[:1900]}...")

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