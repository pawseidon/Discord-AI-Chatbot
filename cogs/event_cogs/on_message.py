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
        self.processed_messages = set()  # Track messages that have been processed
        self.currently_processing = set()  # Track messages currently being processing
        self.last_reasoning_type = {}  # Store last reasoning type per conversation
        
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
        try:
            # Get clean content without bot mentions
            content = message.clean_content
            message_content = await message_service.smart_mention(content, message, self.bot)
            
            # Check for empty content
            if not message_content.strip():
                await message.reply("How can I help you?")
                return
            
            # Check for privacy command first
            privacy_commands = ["clear my data", "forget me", "delete my data", "reset my history", "forget our conversation", "clear my history"]
            if any(cmd in message_content.lower() for cmd in privacy_commands):
                return await self.handle_privacy_command(message)
            
            # Check for reasoning-related commands
            if await self.handle_reasoning_commands(message, message_content):
                return True
            
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
            
            # Standard AI chat interaction
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
                
                # Set up a callback for streaming updates
                async def update_callback(status: str, metadata: Dict[str, Any]):
                    if status == "thinking":
                        # Don't show detailed thinking process to users
                        await initial_message.edit(content=f"🧠 **Processing your request...**")
                    elif status == "agent_switch":
                        agent_type = metadata.get("agent_type", "")
                        emoji, _ = await agent_service.get_agent_emoji_and_description(agent_type)
                        try:
                            await initial_message.edit(content=f"{emoji} **Working on your request...**")
                            
                            # Update emoji reactions when the agent type changes
                            if emoji_reaction_cog:
                                await emoji_reaction_cog.update_reasoning_reactions(initial_message, [agent_type])
                        except discord.HTTPException as e:
                            logger.warning(f"Error updating message during agent switch: {str(e)}")
                    elif status == "tool_use":
                        tool_name = metadata.get("tool_name", "")
                        try:
                            await initial_message.edit(content=f"🔧 **Gathering information...**")
                        except discord.HTTPException as e:
                            logger.warning(f"Error updating message during tool use: {str(e)}")
                    elif status == "reasoning_switch":
                        reasoning_types = metadata.get("reasoning_types", [])
                        is_combined = metadata.get("is_combined", False)
                        if reasoning_types and emoji_reaction_cog:
                            try:
                                await emoji_reaction_cog.update_reasoning_reactions(
                                    initial_message, 
                                    reasoning_types[:2] if is_combined else [reasoning_types[0]]
                                )
                                
                                # Show simple operational status
                                await initial_message.edit(content=f"🧠 **Processing your request...**")
                            except discord.HTTPException as e:
                                logger.warning(f"Error updating reasoning reactions: {str(e)}")
                    elif status == "update":
                        # Only show updates for final content, not intermediate steps
                        pass
                
                try:
                    # Process the query with the detected reasoning
                    response = await agent_service.process_query(
                        query=message_content,
                        user_id=str(message.author.id),
                        conversation_id=f"{message.guild.id if message.guild else 'DM'}:{message.channel.id}",
                        reasoning_type=reasoning_types[0] if reasoning_types else "conversational",
                        update_callback=update_callback
                    )
                    
                    # Validate response
                    if not response or response.strip() == "":
                        logger.warning("Empty response received from agent")
                        response = "I encountered an issue generating a response. Please try asking your question again."
                    
                    # Format response with emoji
                    formatted_response, _ = await agent_service.format_with_agent_emoji(
                        response, 
                        reasoning_types[0] if reasoning_types else "conversational"
                    )
                    
                    # Ensure the response is properly formatted and not too long
                    if len(formatted_response) > 2000:
                        # Get message chunks using the service
                        chunks = await message_service.split_message(formatted_response)
                        
                        # Update the initial message with the first chunk
                        await initial_message.edit(content=chunks[0])
                        
                        # Send additional chunks as new messages
                        for chunk in chunks[1:]:
                            await message.channel.send(chunk)
                    else:
                        # Send as a single message
                        await initial_message.edit(content=formatted_response)
                
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
                    await initial_message.edit(content=f"⚠️ I encountered an error while processing your request: {str(e)[:100]}...\n\nPlease try again or contact the bot administrator if the issue persists.")
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in process_message: {error_traceback}")
            try:
                await message.channel.send(f"❌ I encountered an error while processing your message: {str(e)[:1500]}")
            except:
                pass
        
        # No specific intent detected or intent was not handled
        return None
        
    async def handle_reasoning_commands(self, message, content):
        """Handle natural language commands related to reasoning modes"""
        guild_id = str(message.guild.id) if message.guild else "DM"
        user_id = str(message.author.id)
        conversation_id = f"{guild_id}:{message.channel.id}"
        
        # Check for reasoning preference commands
        preference_keywords = [
            "set my reasoning mode", "change reasoning mode", "use reasoning mode",
            "prefer reasoning", "set reasoning preference", "change my reasoning"
        ]
        
        # Check for reasoning info requests
        info_keywords = [
            "explain reasoning modes", "reasoning modes info", "how do reasoning modes work",
            "tell me about reasoning", "what reasoning modes", "reasoning types"
        ]
        
        # Check for workflow mode toggle
        workflow_keywords = [
            "use workflow mode", "enable langgraph", "use graph workflow", 
            "disable workflow mode", "toggle workflow"
        ]
        
        # Check for reset conversation requests
        reset_conv_keywords = [
            "reset conversation", "reset our chat", "start fresh", "new conversation",
            "restart chat", "clear conversation", "forget this conversation", 
            "start over", "reset our conversation", "begin again", "clean slate"
        ]
        
        # Process reset conversation request
        for keyword in reset_conv_keywords:
            if keyword.lower() in content.lower():
                await self.reset_current_conversation(message, conversation_id)
                return True
        
        # Process workflow mode toggle
        for keyword in workflow_keywords:
            if keyword.lower() in content.lower():
                if "disable" in keyword.lower() or "off" in content.lower():
                    await memory_service.set_user_preference(user_id, "use_workflow_mode", False)
                    await message.reply("📊 Workflow mode is now disabled. I'll use the standard orchestrator for reasoning.")
                    return True
                else:
                    try:
                        # Check if LangGraph is available
                        from langgraph.graph import StateGraph
                        await memory_service.set_user_preference(user_id, "use_workflow_mode", True)
                        await message.reply("📊 Workflow mode is now enabled. I'll use LangGraph for more advanced reasoning flows.")
                        return True
                    except ImportError:
                        await message.reply("📊 Workflow mode requires LangGraph to be installed. Please install it with `pip install langgraph`.")
                        return True
        
        # Process reasoning preference request
        for keyword in preference_keywords:
            if keyword.lower() in content.lower():
                # Get reasoning type from content
                reasoning_type = None
                reasoning_types = [
                    "sequential", "rag", "conversational", "knowledge", "verification", 
                    "creative", "calculation", "planning", "graph", "multi_agent",
                    "step_back", "cot", "react"
                ]
                
                for rtype in reasoning_types:
                    if rtype.lower() in content.lower():
                            reasoning_type = rtype
                            break
                
                if reasoning_type:
                    # Save the preference
                    await memory_service.set_user_preference(user_id, "default_reasoning", reasoning_type)
                    
                    # Get the emoji for the reasoning type
                    emoji, _ = await agent_service.get_agent_emoji_and_description(reasoning_type)
                    
                    # Confirm to the user
                    await message.reply(f"{emoji} I've set your default reasoning mode to **{reasoning_type}**. I'll use this mode unless you specify otherwise.")
                    return True
                else:
                    # No specific reasoning type found
                    await message.reply("Please specify a valid reasoning mode (sequential, rag, conversational, etc.)")
                    return True
        
        # Process reasoning info request
        for keyword in info_keywords:
            if keyword.lower() in content.lower():
                await self.send_reasoning_info(message)
                return True
        
        return False
    
    async def send_reasoning_info(self, message):
        """Send information about available reasoning modes"""
        reasoning_types = [
            "sequential", "rag", "conversational", "knowledge", "verification", 
            "creative", "calculation", "planning", "graph", "multi_agent",
            "step_back", "cot", "react"
        ]
        
        # Create an embed with information about reasoning modes
        embed = discord.Embed(
            title="AI Reasoning Modes",
            description="I can use different reasoning modes to approach different types of questions. Here are the available modes:",
            color=discord.Color.blue()
        )
        
        # Add fields for each reasoning type
        for reasoning_type in reasoning_types:
            emoji, description = await agent_service.get_agent_emoji_and_description(reasoning_type)
            embed.add_field(
                name=f"{emoji} {reasoning_type.capitalize()}",
                value=description,
                inline=False
            )
        
        # Add a footer with instructions
        embed.set_footer(text="You can set your default reasoning mode with 'set my reasoning mode to [mode]'")
        
        # Send the embed
        await message.reply(embed=embed)
    
    async def reset_current_conversation(self, message, conversation_id: str):
        """Reset the current conversation"""
        try:
            # Reset conversation in memory service
            await memory_service.reset_conversation(conversation_id)
            
            # Reset conversation in agent service
            await agent_service.reset_conversation(conversation_id)
            
            # Reset local tracking
            if conversation_id in self.last_reasoning_type:
                del self.last_reasoning_type[conversation_id]
            
            # Let the user know it's done
            await message.reply("✅ I've reset our conversation. Let's start fresh!")
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error resetting conversation: {error_traceback}")
            await message.reply(f"❌ I encountered an error while resetting our conversation: {str(e)[:1500]}")

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
                    # Get the user and conversation IDs
                    user_id = str(message.author.id)
                    guild_id = str(message.guild.id) if message.guild else "DM"
                    conversation_id = f"{guild_id}:{message.channel.id}"
                    
                    # Check if the query seems to request analysis, synthesis, or step-by-step processing
                    needs_processing = any(term in query.lower() for term in [
                        "analyze", "breakdown", "explain", "research", "review", "compare", 
                        "step by step", "in detail", "pros and cons", "advantages", "disadvantages",
                        "summarize", "evaluate", "assess", "study", "examine"
                    ])
                    
                    if needs_processing:
                        # Update the message to indicate we're analyzing the data
                        await processing_message.edit(content="🔎 Found relevant information. Now analyzing and synthesizing into a comprehensive response...")
                        
                        # Add sequential thinking emoji reaction
                        if emoji_reaction_cog:
                            await emoji_reaction_cog.add_reasoning_reactions(processing_message, ["sequential"])
                        
                        # Use workflow service to process with sequential_rag workflow
                        from bot_utilities.services.workflow_service import workflow_service
                        
                        # Set up context with search results
                        context = {
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "retrieved_information": search_result
                        }
                        
                        # Define an update callback for streaming process updates
                        async def update_callback(status: str, metadata: Dict[str, Any]):
                            if status == "thinking":
                                thinking = metadata.get("thinking", "")
                                await processing_message.edit(content=f"🧠 **Processing**: {thinking}")
                            elif status == "reasoning_switch":
                                reasoning_types = metadata.get("reasoning_types", [])
                                is_combined = metadata.get("is_combined", False)
                                emoji_list = []
                                for rtype in reasoning_types:
                                    emoji, _ = await agent_service.get_agent_emoji_and_description(rtype)
                                    emoji_list.append(emoji)
                                
                                emojis = " + ".join(emoji_list)
                                await processing_message.edit(content=f"{emojis} **Using {' + '.join(reasoning_types)} reasoning**\n\nAnalyzing web search results...")
                                
                                # Update emoji reactions
                                if emoji_reaction_cog:
                                    await emoji_reaction_cog.add_reasoning_reactions(processing_message, reasoning_types)
                        
                        # Process with sequential_rag workflow
                        analysis_result = await workflow_service.process_with_workflow(
                            query=query,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            workflow_type="sequential_rag",
                            update_callback=update_callback
                        )
                        
                        # Update the message with the final analysis
                        if len(analysis_result) > 2000:
                            # Split long message into chunks of 2000 characters
                            chunks = [analysis_result[i:i+2000] for i in range(0, len(analysis_result), 2000)]
                            
                            # Send first chunk by editing the original message
                            await processing_message.edit(content=chunks[0])
                            
                            # Send remaining chunks as new messages
                            for chunk in chunks[1:]:
                                await message.channel.send(content=chunk)
                        else:
                            await processing_message.edit(content=analysis_result)
                    else:
                        # Just display search results without further processing
                        # Create an embed for the response
                        embed = discord.Embed(
                            title=f"Search Results: {query}",
                            description=search_result[:4000],  # Limit to fit in embed (4000 chars max)
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
            # Import sequential thinking service
            from bot_utilities.services.sequential_thinking_service import sequential_thinking_service
            
            # Get user ID and context
            user_id = str(message.author.id)
            guild_id = str(message.guild.id) if message.guild else "DM"
            conversation_id = f"{guild_id}:{message.channel.id}"
            
            # Prepare context with user info
            context = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "channel_id": str(message.channel.id),
                "guild_id": guild_id
            }
            
            # Set up a callback for streaming updates
            async def update_callback(status: str, metadata: Dict[str, Any]):
                if status == "thinking":
                    # Don't show detailed thinking process to users
                    await initial_message.edit(content=f"🧠 **Processing with sequential thinking...**")
                elif status == "agent_switch":
                    agent_type = metadata.get("agent_type", "")
                    emoji, _ = await agent_service.get_agent_emoji_and_description(agent_type)
                    await initial_message.edit(content=f"{emoji} **Using {agent_type.capitalize()} Agent**\n\nWorking on your request...")
                    
                    # Update emoji reactions when the agent type changes
                    if emoji_reaction_cog:
                        await emoji_reaction_cog.update_reasoning_reactions(initial_message, ["sequential", agent_type])
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
            
            # Process with sequential thinking service
            success, response = await sequential_thinking_service.process_sequential_thinking(
                problem=problem,
                context=context,
                prompt_style="sequential",
                num_thoughts=5,
                temperature=0.3,
                enable_revision=True,
                enable_reflection=False,
                session_id=f"session_{user_id}_{conversation_id}"
            )
            
            # Format the response and update it with agent emoji
            formatted_response, _ = await agent_service.format_with_agent_emoji(response, "sequential")
            
            # Update the message with the final response
            if len(formatted_response) > 2000:
                # Split the message into chunks
                chunks = await message_service.split_message(formatted_response)
                
                # Send the first chunk by updating the initial message
                await initial_message.edit(content=chunks[0])
                
                # Send remaining chunks as new messages
                for chunk in chunks[1:]:
                    await message.channel.send(content=chunk)
            else:
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
            if len(formatted_response) > 2000:
                # Split long message into chunks
                chunks = await message_service.split_message(formatted_response)
                
                # Update initial message with first chunk
                await initial_message.edit(content=chunks[0])
                
                # Send remaining chunks as new messages
                for chunk in chunks[1:]:
                    await message.channel.send(content=chunk)
            else:
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
                    # Don't show detailed thinking process to users
                    await initial_message.edit(content=f"👥 **Processing with multiple agent perspectives...**")
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
            
            # Send the response
            if len(formatted_response) > 2000:
                # Split long message into chunks
                chunks = await message_service.split_message(formatted_response)
                
                # Update the initial message with the first chunk
                await initial_message.edit(content=chunks[0])
                
                # Send remaining chunks as new messages
                for chunk in chunks[1:]:
                    await message.channel.send(content=chunk)
            else:
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
                    # Don't show detailed thinking process to users
                    await initial_response.edit(content=f"🧠 **Processing your request...**")
                elif status == "agent_switch":
                    agent_type = metadata.get("agent_type", "")
                    emoji, _ = await agent_service.get_agent_emoji_and_description(agent_type)
                    await initial_response.edit(content=f"{emoji} **Working on your request...**")
                    
                    # Update emoji reactions when the agent type changes
                    if emoji_reaction_cog:
                        if should_combine:
                            await emoji_reaction_cog.update_reasoning_reactions(initial_response, [reasoning_types[0], agent_type])
                        else:
                            await emoji_reaction_cog.update_reasoning_reactions(initial_response, [agent_type])
                elif status == "tool_use":
                    tool_name = metadata.get("tool_name", "")
                    await initial_response.edit(content=f"🔧 **Gathering information...**")
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
            if len(formatted_response) > 2000:
                # Split long message into chunks
                chunks = await message_service.split_message(formatted_response)
                
                # Update the initial message with the first chunk
                await initial_response.edit(content=chunks[0])
                
                # Send remaining chunks as new messages
                for chunk in chunks[1:]:
                    await message.channel.send(content=chunk)
            else:
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
        # Skip bot messages
        if message.author.bot:
            return
            
        # Check if we've already processed this message
        if message.id in self.processed_messages:
            return
            
        # Check if this message is currently being processed
        if message.id in self.currently_processing:
            return
            
        # Check if the bot is mentioned or the message is a reply to the bot
        is_mentioned = self.bot.user in message.mentions
        is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author.id == self.bot.user.id
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Process DMs, mentions, and replies
        if is_mentioned or is_reply_to_bot or is_dm:
            try:
                # Mark this message as currently being processed
                self.currently_processing.add(message.id)
                
                # Track message processing
                logger.info(f"Processing message from {message.author.name} ({message.author.id}): {message.content[:50]}...")
                
                # Process the message
                await self.process_message(message)
                
                # Mark message as processed after completion
                self.processed_messages.add(message.id)
                self.currently_processing.remove(message.id)
                
                # Update user's last seen time
                try:
                    await memory_service.update_user_last_seen(str(message.author.id))
                except Exception as user_seen_err:
                    logger.error(f"Error updating user last seen: {user_seen_err}")
                    # Continue execution even if this fails
                
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error in on_message: {error_traceback}")
                
                # Clean up processing status
                if message.id in self.currently_processing:
                    self.currently_processing.remove(message.id)
                
                try:
                    await message.channel.send(f"❌ I encountered an error: {str(e)[:1500]}")
                except:
                    pass

async def setup(bot):
    await bot.add_cog(OnMessage(bot))