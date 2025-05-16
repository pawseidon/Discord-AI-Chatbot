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
        # Register services after initialization to avoid circular imports
        workflow_service.register_services(agent_service=agent_service, memory_service=memory_service)
        # Other services don't need explicit initialization

    async def process_message(self, message):
        """Process an incoming message and generate a response"""
        try:
            # Get clean content without bot mentions and persona prefixes
            content = message.clean_content
            message_content = await message_service.smart_mention(content, message, self.bot)
            
            # Check for empty content
            if not message_content.strip():
                await message.reply("How can I help you?")
                return
            
            # Check if the message is a persona-specific command (e.g., "Hand break down...")
            # Get persona name from config
            default_persona = config.get('DEFAULT_INSTRUCTION', 'hand')
            persona_prefix = False
            
            # Check if message starts with a persona name
            if message_content.split()[0].lower() == default_persona.lower():
                # Remove the persona name from the content
                message_content = ' '.join(message_content.split()[1:])
                persona_prefix = True
            
            # Check for privacy command first
            privacy_commands = ["clear my data", "forget me", "delete my data", "reset my history", "forget our conversation", "clear my history"]
            if any(cmd in message_content.lower() for cmd in privacy_commands):
                return await self.handle_privacy_command(message)
            
            # Check if this is a clarification response to a previous question
            if await self.check_and_handle_clarification(message, message_content):
                return True
            
            # Check for reasoning-related commands
            if await self.handle_reasoning_commands(message, message_content):
                return True
            
            # Check for specific intents
            intent_data = await intent_service.detect_intent(message_content, message)
            if intent_data:
                intent_type = intent_data.get("intent")
                
                if intent_type == "web_search":
                    return await self.handle_web_search(message, intent_data.get("query"))
                
                elif intent_type == "social_relationship":
                    # Get the query from the intent data
                    social_query = intent_data.get("query")
                    
                    # Create initial response message
                    initial_message = await message.reply("👥 Processing your relationship query...")
                    
                    # Add appropriate emoji reactions
                    emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
                    if emoji_reaction_cog:
                        await emoji_reaction_cog.add_reasoning_reactions(initial_message, ["sequential"])
                    
                    try:
                        # Use the workflow service to handle social relationship queries
                        from bot_utilities.services.workflows.sequential_rag_workflow import handle_social_relationship_query, get_ai_provider
                        
                        # Get AI provider for generating response
                        llm_provider = await get_ai_provider()
                        
                        # Use specialized handler for social relationship queries
                        relationship_response = await handle_social_relationship_query(social_query, llm_provider)
                        
                        # Send the response
                        if len(relationship_response) > 2000:
                            # Split into chunks
                            chunks = await message_service.split_message(relationship_response)
                            await initial_message.edit(content=chunks[0])
                            
                            # Send remaining chunks
                            for chunk in chunks[1:]:
                                await message.channel.send(chunk)
                        else:
                            await initial_message.edit(content=relationship_response)
                        
                        return True
                    except Exception as e:
                        logger.error(f"Error handling social relationship query: {str(e)}\n{traceback.format_exc()}")
                        await initial_message.edit(content=f"❌ I encountered an error processing your request: {str(e)[:1500]}")
                        return True
                    
                elif intent_type == "sequential_thinking":
                    return await self.handle_sequential_thinking(message, intent_data.get("problem"))
                    
                elif intent_type == "multi_agent":
                    return await self.handle_multi_agent(message, intent_data.get("query"))
                    
                elif intent_type == "symbolic_reasoning":
                    return await self.handle_symbolic_reasoning(message, intent_data.get("expression"))
                    
                elif intent_type == "privacy":
                    return await self.handle_privacy_command(message)
                
                elif intent_type == "crypto_price":
                    return await self.handle_crypto_price(message, intent_data.get("crypto"))
            
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
            
        print(f"\n🔍 WEB SEARCH REQUEST: '{query}'")
        print(f"├── User: {message.author} ({message.author.id})")
        print(f"└── Channel: {message.channel}")
        
        async with message.channel.typing():
            # Send a processing message
            processing_message = await message.reply(f"🔍 **Searching for**: {query}...")
            
            # Add search emoji reaction
            emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
            if emoji_reaction_cog:
                await emoji_reaction_cog.add_reasoning_reactions(processing_message, ["rag"])
            
            try:
                # Search the web
                search_result = await agent_service.search_web(query)
                
                if search_result:
                    # Always process search results with agent_service's sequential reasoning
                    
                    # Get user and conversation IDs
                    user_id = str(message.author.id)
                    guild_id = str(message.guild.id) if message.guild else "DM"
                    conversation_id = f"{guild_id}:{message.channel.id}"
                    
                    # Setup callback for streaming updates
                    async def update_callback(status: str, metadata: Dict[str, Any]):
                        if status == "thinking":
                            thinking = metadata.get("thinking", "")
                            if thinking:
                                # Truncate long thinking messages to avoid Discord's 2000 character limit
                                if len(f"🧠 **Processing**: {thinking}") > 1900:
                                    thinking = thinking[:1500] + "..."
                                await processing_message.edit(content=f"🧠 **Processing**: {thinking}")
                    
                    print(f"\n📊 WORKFLOW: Sequential RAG for query '{query}'")
                    print(f"├── Starting: RAG → Sequential → Verification pipeline")
                    print(f"├── Search Results Length: {len(search_result.split())} words")
                    
                    # Process using workflow service for better orchestration
                    analysis_result = await workflow_service.process_with_workflow(
                        query=query,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        workflow_type="sequential_rag",
                        update_callback=update_callback,
                        search_results=search_result,
                        display_raw_results=False  # Always process results
                    )
                    
                    # Handle potential long responses
                    if len(analysis_result) > 2000:
                        # Split into chunks of 2000 characters max
                        chunks = [analysis_result[i:i+2000] for i in range(0, len(analysis_result), 2000)]
                        
                        # Update original message with first chunk
                        await processing_message.edit(content=chunks[0])
                        
                        # Send additional chunks as follow-up messages
                        for chunk in chunks[1:]:
                            await message.channel.send(content=chunk)
                    else:
                        # Send as single message
                        await processing_message.edit(content=analysis_result)
                    
                    # Log workflow completion
                    print(f"└── Completed: Sequential RAG workflow")
                else:
                    await processing_message.edit(content=f"❌ No results found for your search on '{query}'.")
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error in handle_web_search: {error_traceback}")
                await processing_message.edit(content=f"❌ Error during web search: {str(e)[:1500]}")

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
        
        # Start animated thinking indicator in background
        thinking_task = self.bot.loop.create_task(
            self.animate_thinking_indicator(initial_message, "🧠 **Processing with sequential thinking")
        )
        
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
                if status == "agent_switch":
                    # Cancel the thinking animation
                    thinking_task.cancel()
                    
                    agent_type = metadata.get("agent_type", "")
                    emoji, _ = await agent_service.get_agent_emoji_and_description(agent_type)
                    await initial_message.edit(content=f"{emoji} **Using {agent_type.capitalize()} Agent**\n\nWorking on your request...")
                    
                    # Update emoji reactions when the agent type changes
                    if emoji_reaction_cog:
                        await emoji_reaction_cog.update_reasoning_reactions(initial_message, ["sequential", agent_type])
                elif status == "tool_use":
                    # Cancel the thinking animation
                    thinking_task.cancel()
                    
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
                enable_revision=True,
                session_id=f"session_{user_id}_{conversation_id}"
            )
            
            # Cancel the thinking animation when done
            thinking_task.cancel()
            
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
            # Cancel the thinking animation on error
            thinking_task.cancel()
            
            error_traceback = traceback.format_exc()
            logger.error(f"Error in handle_sequential_thinking: {error_traceback}")
            await initial_message.edit(content=f"❌ Error in sequential thinking: {str(e)[:1500]}")

    async def animate_thinking_indicator(self, message, base_text):
        """Animate a thinking indicator with sequential dots for better user feedback"""
        dots = 0
        max_dots = 3
        try:
            while True:
                dots = (dots % max_dots) + 1
                dot_text = "." * dots + " " * (max_dots - dots)
                await message.edit(content=f"{base_text}{dot_text}**")
                await asyncio.sleep(0.7)  # Adjust the animation speed
        except asyncio.CancelledError:
            # Task was cancelled, which is expected when processing completes
            pass
        except Exception as e:
            logger.error(f"Error in animate_thinking_indicator: {str(e)}")

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

    async def handle_crypto_price(self, message, crypto_name):
        """Handle cryptocurrency price queries"""
        if not crypto_name:
            await message.reply("Please specify which cryptocurrency you'd like the price for.")
            return
            
        # Create tracking message
        initial_message = await message.reply("💰 **Fetching current cryptocurrency prices...**")
        
        # Add RAG emoji reaction for web data retrieval
        emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
        if emoji_reaction_cog:
            await emoji_reaction_cog.add_reasoning_reactions(initial_message, ["rag"])
        
        try:
            # Standardize crypto name
            crypto_name = crypto_name.lower()
            crypto_map = {
                "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
                "doge": "dogecoin", "ada": "cardano", "dot": "polkadot",
                "avax": "avalanche", "matic": "polygon"
            }
            
            # Map shorthand to full name if needed
            if crypto_name in crypto_map:
                crypto_name = crypto_map[crypto_name]
                
            # Get price from crypto price service
            crypto_data = await get_crypto_price(crypto_name)
            
            if crypto_data and "price" in crypto_data:
                # Format the response with price and metadata
                price = crypto_data["price"]
                change_24h = crypto_data.get("change_24h", 0)
                market_cap = crypto_data.get("market_cap", 0)
                timestamp = crypto_data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
                coin_id = crypto_data.get("coin_id", crypto_name)
                
                # Determine trend emoji
                trend_emoji = "🟢" if change_24h >= 0 else "🔴"
                
                # Create formatted response
                # Map of coin IDs to their ticker symbols
                ticker_symbols = {
                    "bitcoin": "BTC",
                    "ethereum": "ETH",
                    "dogecoin": "DOGE",
                    "ripple": "XRP",
                    "cardano": "ADA",
                    "solana": "SOL",
                    "binancecoin": "BNB",
                    "polkadot": "DOT",
                    "litecoin": "LTC",
                    "chainlink": "LINK",
                    "matic-network": "MATIC",
                    "avalanche-2": "AVAX"
                }
                
                # Get the ticker symbol
                ticker = ticker_symbols.get(coin_id, coin_id.upper() if len(coin_id) <= 4 else "")
                
                formatted_response = f"""# 💰 {coin_id.title()} ({ticker}) Price Information

## Current Price
**${price:,.2f}** {trend_emoji} {change_24h:.2f}% (24h)

## 24-Hour Range
- **High:** ${crypto_data.get("high_24h", price * 1.02):,.2f}
- **Low:** ${crypto_data.get("low_24h", price * 0.98):,.2f}

## Market Data
- **Market Cap:** ${market_cap:,.0f}

*Data sourced from CoinGecko as of {timestamp}*

> This is real-time market data and prices may change rapidly. This information should not be considered financial advice."""
                
                # Update the message with the price information
                await initial_message.edit(content=formatted_response)
                return True
            else:
                # Use web search directly without showing error message
                await initial_message.edit(content=f"🔍 Searching for current {crypto_name} price information...")
                
                # Use the web search method directly - passing the initial message to update it
                await self.handle_web_search_for_crypto(message, initial_message, crypto_name)
                return True
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in handle_crypto_price: {error_traceback}")
            
            # Use web search as fallback without showing error message
            await initial_message.edit(content=f"🔍 Searching for current {crypto_name} price information...")
            
            # Use the web search method directly - passing the initial message to update it
            await self.handle_web_search_for_crypto(message, initial_message, crypto_name)
            return True
            
    async def handle_web_search_for_crypto(self, message, initial_message, crypto_name):
        """Handle web search specifically for cryptocurrency prices"""
        query = f"current {crypto_name} price"
        
        try:
            # Use agent service to search the web
            print(f"🔍 Executing web search with query: '{query}'")
            search_result = await agent_service.search_web(query)
            
            if search_result:
                # Get the user and conversation IDs
                user_id = str(message.author.id)
                guild_id = str(message.guild.id) if message.guild else "DM"
                conversation_id = f"{guild_id}:{message.channel.id}"
                
                # Use workflow service to process with sequential_rag workflow
                
                # Define an update callback for streaming process updates
                async def update_callback(status: str, metadata: Dict[str, Any]):
                    if status == "thinking":
                        thinking = metadata.get("thinking", "")
                        # Truncate long thinking messages to avoid Discord's 2000 character limit
                        if thinking and len(f"🧠 **Processing**: {thinking}") > 1900:
                            thinking = thinking[:1500] + "..."
                        await initial_message.edit(content=f"🧠 **Processing**: {thinking}")
                
                # Process with workflow service
                analysis_result = await workflow_service.process_with_workflow(
                    query=query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workflow_type="sequential_rag",
                    update_callback=update_callback,
                    search_results=search_result
                )
                
                # Update the message with the final analysis
                if len(analysis_result) > 2000:
                    # Split long message into chunks of 2000 characters
                    chunks = [analysis_result[i:i+2000] for i in range(0, len(analysis_result), 2000)]
                    
                    # Send first chunk by editing the original message
                    await initial_message.edit(content=chunks[0])
                    
                    # Send remaining chunks as new messages
                    for chunk in chunks[1:]:
                        await message.channel.send(content=chunk)
                else:
                    await initial_message.edit(content=analysis_result)
            else:
                await initial_message.edit(content=f"❌ No search results found for {crypto_name} price information.")
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in handle_web_search_for_crypto: {error_traceback}")
            await initial_message.edit(content=f"❌ Error searching for {crypto_name} price information.")

    async def check_and_handle_clarification(self, message, content):
        """
        Check if this message is a clarification response to a previous question
        
        Args:
            message: The Discord message
            content: The message content
            
        Returns:
            bool: True if this was a clarification response, False otherwise
        """
        # Get conversation ID
        guild_id = str(message.guild.id) if message.guild else "DM"
        channel_id = str(message.channel.id)
        user_id = str(message.author.id)
        conversation_id = f"{guild_id}:{channel_id}"
        
        # Check if the reference message exists (this is a reply)
        if message.reference and message.reference.resolved:
            referenced_message = message.reference.resolved
            
            # Check if the referenced message is from the bot
            if referenced_message.author.id == self.bot.user.id:
                # Check if reference message contains clarification request indicators
                reference_content = referenced_message.content.lower()
                
                clarification_indicators = [
                    "i need some clarification",
                    "could you clarify",
                    "need more information",
                    "please provide more details",
                    "🔍 i need some clarification"
                ]
                
                if any(indicator in reference_content for indicator in clarification_indicators):
                    # This is a response to a clarification request
                    try:
                        # Create initial response message
                        initial_response = await message.reply("🧠 Processing your clarification...")
                        
                        # Extract the original query from the clarification message
                        original_query_match = re.search(r"Original query:(.+?)(?:\n|$)", reference_content)
                        original_query = original_query_match.group(1).strip() if original_query_match else ""
                        
                        if not original_query:
                            # Try to find the original query in the conversation history
                            conversation_history = await memory_service.get_conversation_history(user_id, conversation_id)
                            if conversation_history and len(conversation_history) >= 4:
                                # The original query should be the second-to-last user message
                                messages = conversation_history[-4:]
                                for i in range(len(messages) - 1, -1, -1):
                                    if messages[i]["role"] == "user" and i < len(messages) - 1:
                                        original_query = messages[i]["content"]
                                        break
                        
                        # Set up a callback for updates
                        async def update_callback(status: str, metadata: Dict[str, Any]):
                            if status == "thinking":
                                await initial_response.edit(content="🧠 **Processing your clarification...**")
                            elif status == "reasoning_switch":
                                reasoning_types = metadata.get("reasoning_types", [])
                                is_combined = metadata.get("is_combined", False)
                                
                                # Update emoji reactions if needed
                                emoji_reaction_cog = self.bot.get_cog('EmojiReactionCog')
                                if reasoning_types and emoji_reaction_cog:
                                    await emoji_reaction_cog.update_reasoning_reactions(
                                        initial_response, 
                                        reasoning_types[:2] if is_combined else [reasoning_types[0]]
                                    )
                        
                        # Process the clarification response using workflow service
                        response = await workflow_service.handle_clarification_response(
                            clarification_response=content,
                            original_query=original_query,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            update_callback=update_callback
                        )
                        
                        # Send the response
                        if len(response) > 2000:
                            chunks = await message_service.split_message(response)
                            await initial_response.edit(content=chunks[0])
                            
                            for chunk in chunks[1:]:
                                await message.channel.send(chunk)
                        else:
                            await initial_response.edit(content=response)
                        
                        return True
                    except Exception as e:
                        logger.error(f"Error handling clarification: {str(e)}\n{traceback.format_exc()}")
                        await message.reply(f"⚠️ I had trouble processing your clarification: {str(e)[:1500]}")
                        return True
        
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
            
        # Track user activity in the system
        try:
            # Get user details
            user_id = str(message.author.id)
            username = message.author.display_name
            joined_at = message.author.joined_at if hasattr(message.author, 'joined_at') else None
            
            # Track user
            await memory_service.track_user(user_id, username, joined_at)
        except Exception as e:
            logger.error(f"Error tracking user: {e}")
            # Continue processing even if tracking fails
        
        # Check if the message is in an active channel
        channel_id = str(message.channel.id)
        active_channels = load_active_channels()
        is_active_channel = channel_id in active_channels
        
        # Check if the bot is mentioned, replied to, or in a DM
        is_mentioned = self.bot.user in message.mentions
        is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author.id == self.bot.user.id
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Check for role mentions
        role_mentions = message.role_mentions
        is_role_mentioned = any(role.id == self.bot.user.id for role in role_mentions) if role_mentions else False
        
        # Check if bot's name is mentioned in the message content
        bot_info = get_bot_names_and_triggers()
        bot_names = bot_info["names"]
        bot_name_mentioned = False
        message_lower = message.content.lower()
        
        # Check for bot's name in message content
        for name in bot_names:
            if name.lower() in message_lower:
                bot_name_mentioned = True
                break
                
        # Process messages if appropriate trigger is detected or in active channel
        if is_mentioned or is_role_mentioned or is_reply_to_bot or is_dm or bot_name_mentioned or is_active_channel:
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