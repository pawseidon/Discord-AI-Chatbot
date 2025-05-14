import discord
from discord.ext import commands
import asyncio
import json
import os
import re
import time
import traceback
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# Import services - these are the preferred interfaces
from bot_utilities.services.intent_detection import intent_service
from bot_utilities.services.message_service import message_service  
from bot_utilities.services.memory_service import memory_service
from bot_utilities.services.agent_service import agent_service
from bot_utilities.services.workflow_service import workflow_service
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service

# Import remaining utility modules that don't have service replacements yet
from bot_utilities.ai_utils import get_crypto_price, text_to_speech, get_bot_names_and_triggers
from bot_utilities.config_loader import config, load_active_channels

# Common imports
from ..common import allow_dm, MAX_HISTORY, instructions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('on_message')

# Directory for storing bot data
BOT_DATA_DIR = os.path.join("bot_data", "user_tracking")
os.makedirs(BOT_DATA_DIR, exist_ok=True)
USER_DATA_FILE = os.path.join(BOT_DATA_DIR, "user_data.json")

class OnMessage(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_channels = load_active_channels
        self.instructions = instructions
        self.thread_contexts = {}  # Store parent message context for threads
        self.STREAM_UPDATE_INTERVAL = 0.5  # Update streaming messages every 0.5 seconds
        self.replied_messages = {}  # Store message IDs for reference
        
        # Initialize services
        self.bot.loop.create_task(self.initialize_services())
    
    async def initialize_services(self):
        """Initialize services with the LLM provider"""
        from bot_utilities.ai_utils import get_ai_provider
        llm_provider = await get_ai_provider()
        await agent_service.initialize(llm_provider)
        await workflow_service.initialize(llm_provider)
        # Other services don't need explicit initialization

    async def process_message(self, message):
        """Process an incoming message and generate a response"""
        # Get clean content without bot mentions
        content = message.clean_content
        message_content = await message_service.smart_mention(content, message, self.bot)
        
        # Check for privacy command first
        privacy_commands = ["clear my data", "forget me", "delete my data", "reset my history", "forget our conversation", "clear my history"]
        if any(cmd in message_content.lower() for cmd in privacy_commands):
            return await self.handle_privacy_command(message)
        
        # Check for specific intents
        intent_data = await intent_service.detect_intent(message_content, message)
        if intent_data:
            intent_type = intent_data.get("intent")
            
            if intent_type == "web_search":
                return await self.handle_web_search(message, intent_data.get("query"))
                
            elif intent_type == "sequential_thinking":
                return await self.handle_sequential_thinking(message, intent_data.get("problem"))
                
            elif intent_type == "multi_agent":
                return await self.handle_multi_agent(message, intent_data.get("query"))
                
            elif intent_type == "symbolic_reasoning":
                return await self.handle_symbolic_reasoning(message, intent_data.get("expression"))
                
            elif intent_type == "privacy":
                return await self.handle_privacy_command(message)
                
            elif intent_type == "chat":
                # Standard AI chat interaction
                try:
                    # Check if we should stream the response
                    stream = config.get('STREAM_RESPONSES', False)
                    
                    # Show typing indicator while generating response
                    async with message.channel.typing():
                        # Detect reasoning types
                        reasoning_types = await agent_service.detect_multiple_reasoning_types(
                            query=message_content,
                            conversation_id=f"{message.guild.id if message.guild else 'DM'}:{message.channel.id}"
                        )
                        
                        # Check if we should combine reasoning types
                        should_combine = await agent_service.should_combine_reasoning(
                            query=message_content,
                            conversation_id=f"{message.guild.id if message.guild else 'DM'}:{message.channel.id}"
                        )
                        
                        # Send initial reply with appropriate reasoning type emoji
                        initial_message = await message.reply("🧠 Processing your request...")
                        
                        # Add initial emoji reactions for reasoning types
                        emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
                        if emoji_reaction_cog:
                            if should_combine:
                                await emoji_reaction_cog.add_reasoning_reactions(initial_message, reasoning_types[:2])
                            else:
                                await emoji_reaction_cog.add_reasoning_reactions(initial_message, [reasoning_types[0]])
                        
                        # Call the agent service to generate a response
                        # Set up a callback for streaming updates
                        async def update_callback(status: str, metadata: Dict[str, Any]):
                            if status == "thinking":
                                thinking = metadata.get("thinking", "")
                                if thinking:
                                    await initial_message.edit(content=f"🧠 **Processing your request...**\n\n{thinking[:1500]}...")
                            elif status == "agent_switch":
                                agent_type = metadata.get("agent_type", "")
                                emoji, _ = await agent_service.get_agent_emoji_and_description(agent_type)
                                await initial_message.edit(content=f"{emoji} **Using {agent_type.capitalize()} Agent**\n\nWorking on your request...")
                                
                                # Update emoji reactions when the agent type changes
                                if emoji_reaction_cog:
                                    await emoji_reaction_cog.update_reasoning_reactions(initial_message, [agent_type])
                            elif status == "tool_use":
                                tool_name = metadata.get("tool_name", "")
                                await initial_message.edit(content=f"🔧 **Using tool: {tool_name}**\n\nGathering information...")
                            elif status == "reasoning_switch":
                                reasoning_types = metadata.get("reasoning_types", [])
                                is_combined = metadata.get("is_combined", False)
                                if reasoning_types and emoji_reaction_cog:
                                    await emoji_reaction_cog.update_reasoning_reactions(
                                        initial_message, 
                                        reasoning_types[:2] if is_combined else [reasoning_types[0]]
                                    )
                        
                        # Process the query with the detected reasoning
                        response = await agent_service.process_query(
                            query=message_content,
                            user_id=str(message.author.id),
                            conversation_id=f"{message.guild.id if message.guild else 'DM'}:{message.channel.id}",
                            reasoning_type=reasoning_types[0] if reasoning_types else "conversational",
                            update_callback=update_callback
                        )
                        
                        # Format response with emoji
                        formatted_response, _ = await agent_service.format_with_agent_emoji(
                            response, 
                            reasoning_types[0] if reasoning_types else "conversational"
                        )
                        
                        # Edit the initial message with the final response
                        await initial_message.edit(content=formatted_response)
                        
                        # Update emoji reactions for the final message
                        if emoji_reaction_cog:
                            if should_combine:
                                await emoji_reaction_cog.update_reasoning_reactions(initial_message, reasoning_types[:2])
                            else:
                                await emoji_reaction_cog.update_reasoning_reactions(initial_message, [reasoning_types[0]])
                        
                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logger.error(f"Error in process_message standard AI chat: {error_traceback}")
                    await message.channel.send(f"I encountered an error while processing your message: {str(e)[:1500]}")
        
        # No specific intent detected or intent was not handled
        return None

    async def handle_web_search(self, message, query):
        """Handle web search intent"""
        if not query:
            await message.reply("Please provide a search query.")
            return
            
        async with message.channel.typing():
            # Send a processing message
            processing_message = await message.reply(f"Searching for: **{query}**...")
            
            # Add search emoji reaction
            emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
            if emoji_reaction_cog:
                await emoji_reaction_cog.add_reasoning_reactions(processing_message, ["rag"])
            
            try:
                # Use agent service to search the web
                search_result = await agent_service.search_web(query)
                
                if search_result:
                    # Create an embed for the response
                    embed = discord.Embed(
                        title=f"Search Results: {query}",
                        description=search_result[:4000],  # Limit to fit in embed
                        color=discord.Color.blue()
                    )
                    
                    # Send the search results
                    await processing_message.edit(content=None, embed=embed)
                else:
                    await processing_message.edit(content=f"❌ No search results found for: {query}")
                
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error in handle_web_search: {error_traceback}")
                await processing_message.edit(content=f"❌ Error searching the web: {str(e)[:1500]}")

    async def handle_sequential_thinking(self, message, problem):
        """Handle sequential thinking for complex problem solving"""
        if not problem:
            await message.reply("Please provide a problem to solve with sequential thinking.")
            return
            
        # Create tracking message
        initial_message = await message.reply("🧠 **Processing with sequential thinking...**")
        
        # Add sequential thinking emoji reaction
        emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
        if emoji_reaction_cog:
            await emoji_reaction_cog.add_reasoning_reactions(initial_message, ["sequential"])
        
        try:
            # Set up a callback for streaming updates
            async def update_callback(status: str, metadata: Dict[str, Any]):
                if status == "thinking":
                    thinking = metadata.get("thinking", "")
                    if thinking:
                        await initial_message.edit(content=f"🧠 **Sequential Analysis**\n\n{thinking[:1500]}...")
                elif status == "agent_switch":
                    agent_type = metadata.get("agent_type", "")
                    emoji, _ = await agent_service.get_agent_emoji_and_description(agent_type)
                    await initial_message.edit(content=f"{emoji} **Using {agent_type.capitalize()} Agent**\n\nWorking on your request...")
                    
                    # Update emoji reactions when the agent type changes
                    if emoji_reaction_cog:
                        await emoji_reaction_cog.update_reasoning_reactions(initial_message, [agent_type])
                elif status == "tool_use":
                    tool_name = metadata.get("tool_name", "")
                    await initial_message.edit(content=f"🔧 **Using tool: {tool_name}**\n\nGathering information...")
                elif status == "reasoning_switch":
                    reasoning_types = metadata.get("reasoning_types", [])
                    is_combined = metadata.get("is_combined", False)
                    if reasoning_types and emoji_reaction_cog:
                        await emoji_reaction_cog.update_reasoning_reactions(
                            initial_message, 
                            reasoning_types[:2] if is_combined else [reasoning_types[0]]
                        )
            
            # Process the query with the sequential agent
            user_id = str(message.author.id)
            guild_id = str(message.guild.id) if message.guild else "DM"
            conversation_id = f"{guild_id}:{message.channel.id}"
            
            response = await agent_service.process_query(
                query=problem,
                user_id=user_id,
                conversation_id=conversation_id,
                reasoning_type="sequential",
                update_callback=update_callback
            )
            
            # Format the response with emoji
            formatted_response, _ = await agent_service.format_with_agent_emoji(response, "sequential")
            
            # Update the message with the final response
            await initial_message.edit(content=formatted_response)
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in handle_sequential_thinking: {error_traceback}")
            await initial_message.edit(content=f"❌ Error in sequential thinking: {str(e)[:1500]}")

    async def handle_symbolic_reasoning(self, message, expression):
        """Handle symbolic reasoning for math and logic expressions"""
        if not expression:
            await message.reply("Please provide an expression to evaluate.")
            return
            
        # Create tracking message
        initial_message = await message.reply("🧮 **Processing symbolic reasoning...**")
        
        # Add symbolic reasoning emoji reaction
        emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
        if emoji_reaction_cog:
            await emoji_reaction_cog.add_reasoning_reactions(initial_message, ["calculation"])
        
        try:
            # Process with symbolic reasoning service
            result = await symbolic_reasoning_service.evaluate_expression(expression)
            
            # Format the response
            if "error" in result:
                formatted_response = f"❌ **Error in symbolic reasoning**\n\n{result['error']}"
            else:
                formatted_response = f"🧮 **Symbolic Reasoning Result**\n\nExpression: `{expression}`\nResult: `{result['result']}`"
                
                # Add steps if available
                if "steps" in result and result["steps"]:
                    steps_text = "\n".join([f"• {step}" for step in result["steps"]])
                    formatted_response += f"\n\n**Steps:**\n{steps_text}"
            
            # Update the message with the final response
            await initial_message.edit(content=formatted_response)
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in handle_symbolic_reasoning: {error_traceback}")
            await initial_message.edit(content=f"❌ Error in symbolic reasoning: {str(e)[:1500]}")

    async def handle_multi_agent(self, message, query):
        """Handle request with multiple agent perspectives"""
        if not query:
            await message.reply("Please provide a query for multi-agent processing.")
            return
            
        # Create tracking message
        initial_message = await message.reply("👥 **Processing with multiple agent perspectives...**")
        
        # Add multi-agent emoji reaction
        emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
        if emoji_reaction_cog:
            await emoji_reaction_cog.add_reasoning_reactions(initial_message, ["multi_agent"])
        
        try:
            # Set up a callback for streaming updates
            async def update_callback(status, metadata):
                if status == "thinking":
                    thinking = metadata.get("thinking", "")
                    if thinking:
                        await initial_message.edit(content=f"👥 **Multi-Agent Processing**\n\n{thinking[:1500]}...")
                elif status == "agent_switch":
                    agent_type = metadata.get("agent_type", "")
                    emoji, _ = await agent_service.get_agent_emoji_and_description(agent_type)
                    await initial_message.edit(content=f"{emoji} **Using {agent_type.capitalize()} Agent**\n\nWorking on your request...")
                    
                    # Update emoji reactions when the agent type changes
                    if emoji_reaction_cog:
                        await emoji_reaction_cog.update_reasoning_reactions(initial_message, ["multi_agent", agent_type])
                elif status == "tool_use":
                    tool_name = metadata.get("tool_name", "")
                    await initial_message.edit(content=f"🔧 **Using tool: {tool_name}**\n\nGathering information...")
                elif status == "reasoning_switch":
                    reasoning_types = metadata.get("reasoning_types", [])
                    is_combined = metadata.get("is_combined", False)
                    if reasoning_types and emoji_reaction_cog:
                        await emoji_reaction_cog.update_reasoning_reactions(
                            initial_message, 
                            reasoning_types
                        )
            
            # Process with the multi-agent system
            user_id = str(message.author.id)
            guild_id = str(message.guild.id) if message.guild else "DM"
            conversation_id = f"{guild_id}:{message.channel.id}"
            
            if workflow_service.is_workflow_available():
                # Use workflow-based processing for more advanced orchestration
                response = await workflow_service.process_with_workflow(
                    query=query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workflow_type="multi_agent",
                    update_callback=update_callback
                )
                
                # Format response with multi-agent emoji
                formatted_response, _ = await agent_service.format_with_agent_emoji(
                    response, 
                    "multi_agent"
                )
            else:
                # Use standard multi-agent processing
                response = await agent_service.process_query(
                    query=query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    reasoning_type="multi_agent",
                    update_callback=update_callback
                )
                
                # Format response with emoji
                formatted_response, _ = await agent_service.format_with_agent_emoji(
                    response, 
                    "multi_agent"
                )
            
            # Update message with final response
            await initial_message.edit(content=formatted_response)
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in handle_multi_agent: {error_traceback}")
            await initial_message.edit(content=f"❌ Error in multi-agent processing: {str(e)[:1500]}")

    async def process_streaming_response(self, initial_message, response_stream):
        """Process a streaming response from the AI"""
        try:
            accumulated_response = ""
            last_update_time = time.time()
            
            async for chunk in response_stream:
                accumulated_response += chunk
                
                # Update the message periodically to avoid rate limits
                current_time = time.time()
                if current_time - last_update_time >= self.STREAM_UPDATE_INTERVAL:
                    try:
                        await message_service.send_response(
                            initial_message, 
                            accumulated_response, 
                            update_existing=True
                        )
                        last_update_time = current_time
                    except discord.HTTPException as e:
                        if e.code == 50035:  # Invalid Form Body
                            # Message might be too long, just continue accumulating
                            pass
                        else:
                            raise
            
            # Final update with complete response
            if accumulated_response:
                await message_service.send_response(
                    initial_message, 
                    accumulated_response,
                    update_existing=True
                )
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in process_streaming_response: {error_traceback}")
            await initial_message.channel.send(f"❌ Error while streaming response: {str(e)[:1500]}")

    async def create_voice_response(self, text):
        """Create a voice response from text"""
        try:
            # Limit text length for voice
            text = text[:2000]
            
            # Call the text to speech function
            audio_data = await text_to_speech(text)
            
            if audio_data:
                return discord.File(audio_data, filename="response.mp3")
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating voice response: {e}")
            return None

    async def handle_privacy_command(self, message):
        """Handle user privacy commands"""
        user_id = str(message.author.id)
        
        # Check for specific privacy commands
        content = message.content.lower()
        
        try:
            if any(cmd in content for cmd in ["clear my data", "forget me", "delete my data"]):
                # Clear all user data
                await memory_service.clear_user_data(user_id)
                await agent_service.clear_user_data(user_id)
                
                await message.reply("✅ I've cleared all your data from my memory. Your previous conversations and preferences have been removed.")
                return True
                
            elif any(cmd in content for cmd in ["reset our conversation", "forget this conversation", "start over", "clear my history"]):
                # Reset just this conversation
                guild_id = str(message.guild.id) if message.guild else "DM"
                conversation_id = f"{guild_id}:{message.channel.id}"
                
                await memory_service.reset_conversation(conversation_id)
                await agent_service.reset_conversation(conversation_id)
                
                await message.reply("✅ I've reset our conversation. Let's start fresh!")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error handling privacy command: {e}")
            await message.reply(f"❌ I encountered an error processing your privacy request: {str(e)[:1500]}")
            return True

    async def handle_ai_response(self, message, user_id, content):
        try:
            # Get the ReasoningCog and EmojiReactionCog for tracking responses
            reasoning_cog = self.bot.get_cog('ReasoningCog')
            emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
            
            # Determine conversation ID
            conversation_id = f"{message.guild.id if message.guild else 'DM'}:{message.channel.id}"
            
            # Initial response message
            initial_response = await message.reply("🧠 Processing...")
            
            # Detect reasoning types for this query
            reasoning_types = await agent_service.detect_multiple_reasoning_types(
                query=content,
                conversation_id=conversation_id
            )
            
            # Determine if we should combine reasoning types
            should_combine = await agent_service.should_combine_reasoning(
                query=content,
                conversation_id=conversation_id
            )
            
            # Add initial reasoning emoji reactions
            if emoji_reaction_cog:
                if should_combine:
                    await emoji_reaction_cog.add_reasoning_reactions(initial_response, reasoning_types[:2])
                else:
                    await emoji_reaction_cog.add_reasoning_reactions(initial_response, [reasoning_types[0]])
            
            # Define update callback for status updates
            async def update_callback(status: str, metadata: Dict[str, Any]):
                if status == "thinking":
                    thinking = metadata.get("thinking", "")
                    if thinking:
                        await initial_response.edit(content=f"🧠 **Processing your request...**\n\n{thinking[:1500]}...")
                elif status == "agent_switch":
                    agent_type = metadata.get("agent_type", "")
                    emoji, _ = await agent_service.get_agent_emoji_and_description(agent_type)
                    await initial_response.edit(content=f"{emoji} **Using {agent_type.capitalize()} Agent**\n\nWorking on your request...")
                    
                    # Update emoji reactions when the agent type changes
                    if emoji_reaction_cog:
                        if should_combine:
                            await emoji_reaction_cog.update_reasoning_reactions(initial_response, [reasoning_types[0], agent_type])
                        else:
                            await emoji_reaction_cog.update_reasoning_reactions(initial_response, [agent_type])
                elif status == "tool_use":
                    tool_name = metadata.get("tool_name", "")
                    await initial_response.edit(content=f"🔧 **Using tool: {tool_name}**\n\nGathering information...")
                elif status == "reasoning_switch":
                    reasoning_types = metadata.get("reasoning_types", [])
                    is_combined = metadata.get("is_combined", False)
                    if reasoning_types and emoji_reaction_cog:
                        await emoji_reaction_cog.update_reasoning_reactions(
                            initial_response, 
                            reasoning_types[:2] if is_combined else [reasoning_types[0]]
                        )
            
            # Process the query
            response = await agent_service.process_query(
                query=content,
                user_id=user_id,
                conversation_id=conversation_id,
                reasoning_type=reasoning_types[0] if reasoning_types else "conversational",
                update_callback=update_callback
            )
            
            # Format with appropriate emoji
            formatted_response, emoji = await agent_service.format_with_agent_emoji(
                response, 
                reasoning_types[0] if reasoning_types else "conversational"
            )
            
            # Send the response
            await initial_response.edit(content=formatted_response)
            
            # Update emoji reactions for the final message
            if emoji_reaction_cog:
                if should_combine:
                    await emoji_reaction_cog.update_reasoning_reactions(initial_response, reasoning_types[:2])
                else:
                    await emoji_reaction_cog.update_reasoning_reactions(initial_response, [reasoning_types[0]])
                
            return True
            
        except Exception as e:
            error_message = f"Error in AI response: {str(e)}"
            print(f"{error_message}\n{traceback.format_exc()}")
            
            try:
                await message.reply(f"❌ I encountered an error: {str(e)[:1500]}")
            except:
                print(f"Failed to send error message")
                
            return False

    @commands.Cog.listener()
    async def on_message(self, message):
        """Handle incoming messages"""
        if message.author.bot:
            return
            
        # Check if the bot is mentioned or the message is a reply to the bot
        is_mentioned = self.bot.user in message.mentions
        is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author.id == self.bot.user.id
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Process DMs, mentions, and replies
        if is_mentioned or is_reply_to_bot or is_dm:
            try:
                # Track message processing
                logger.info(f"Processing message from {message.author.name} ({message.author.id}): {message.content[:50]}...")
                
                # Process the message
                await self.process_message(message)
                
                # Update user's last seen time
                await memory_service.update_user_last_seen(str(message.author.id))
                
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error in on_message: {error_traceback}")
                
                try:
                    await message.channel.send(f"❌ I encountered an error: {str(e)[:1500]}")
                except:
                    pass

async def setup(bot):
    await bot.add_cog(OnMessage(bot))