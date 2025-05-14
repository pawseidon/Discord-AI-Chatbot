import discord
from discord.ext import commands
import traceback
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import time

# Import services - these are the preferred interfaces
from bot_utilities.services.agent_service import agent_service
from bot_utilities.services.memory_service import memory_service
from bot_utilities.services.message_service import message_service
from bot_utilities.services.workflow_service import workflow_service
from bot_utilities.services.intent_detection import intent_service

# Only import utility modules that don't have service replacements yet
from bot_utilities.monitoring import AgentMonitor
from bot_utilities.ai_utils import get_ai_provider

class ReasoningCog(commands.Cog):
    """A cog for context-aware reasoning with multi-agent orchestration"""
    
    def __init__(self, bot):
        self.bot = bot
        self.monitor = AgentMonitor()
        self.active_conversations = {}
        self.last_reasoning_type = {}
        
        # Track last activity time for cache cleanup
        self.conversation_last_activity = {}
        
        # Schedule periodic cache cleanup
        self.bot.loop.create_task(self._periodic_cache_cleanup())
        
        # Schedule periodic memory persistence
        self.bot.loop.create_task(self._periodic_memory_persistence())
        
        # Initialize the agent service with the LLM provider
        self.bot.loop.create_task(self.initialize_services())
        
        # Add initialization for the emoji reaction cog
        self.emoji_reaction_cog = None
        
        # Schedule periodic cache cleanup
        self.bot.loop.create_task(self._periodic_cache_cleanup())
        
        # Schedule periodic memory persistence
        self.bot.loop.create_task(self._periodic_memory_persistence())
        
        # Initialize the agent service with the LLM provider
        self.bot.loop.create_task(self.initialize_services())
    
    async def initialize_services(self):
        """Initialize services with the LLM provider"""
        llm_provider = await get_ai_provider()
        await agent_service.initialize(llm_provider)
        await workflow_service.initialize(llm_provider)
        
        # Get a reference to the EmojiReactionCog
        for cog in self.bot.cogs.values():
            if isinstance(cog, discord.ext.commands.Cog) and cog.__class__.__name__ == "EmojiReactionCog":
                self.emoji_reaction_cog = cog
                break
        
    async def _periodic_memory_persistence(self):
        """Periodically save agent memory to disk"""
        await self.bot.wait_until_ready()
        while not self.bot.is_closed():
            try:
                # Wait for 1 hour between saves
                await asyncio.sleep(3600)
                
                # Save memory
                await memory_service.save_to_disk()
                print("Saved agent memory to disk")
                
            except Exception as e:
                print(f"Error saving memory to disk: {e}")
    
    async def _periodic_cache_cleanup(self):
        """Periodically clean up inactive conversations from cache"""
        await self.bot.wait_until_ready()
        while not self.bot.is_closed():
            try:
                # Wait for 1 hour between cleanups
                await asyncio.sleep(3600)
                
                current_time = asyncio.get_event_loop().time()
                inactive_conversations = []
                
                # Find conversations inactive for more than 24 hours
                for conv_id, last_time in self.conversation_last_activity.items():
                    if current_time - last_time > 86400:  # 24 hours in seconds
                        inactive_conversations.append(conv_id)
                
                # Clean up inactive conversations
                for conv_id in inactive_conversations:
                    if conv_id in self.active_conversations:
                        del self.active_conversations[conv_id]
                    if conv_id in self.last_reasoning_type:
                        del self.last_reasoning_type[conv_id]
                    del self.conversation_last_activity[conv_id]
                    
                    # Reset conversation in services
                    await agent_service.reset_conversation(conv_id)
                
                print(f"Cleaned up {len(inactive_conversations)} inactive conversations")
                
            except Exception as e:
                print(f"Error in cache cleanup: {e}")
    
    async def handle_natural_language_command(self, message, content):
        """Handle natural language commands for reasoning preferences and info"""
        guild_id = str(message.guild.id) if message.guild else "DM"
        user_id = str(message.author.id)
        conversation_id = f"{guild_id}:{message.channel.id}"
        
        # Update last activity time
        self.conversation_last_activity[conversation_id] = asyncio.get_event_loop().time()
        
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
        
        # Check for clear data requests
        clear_data_keywords = [
            "clear my data", "delete my data", "remove my information", "forget me",
            "clear my history", "erase my data", "delete my history", "forget my data",
            "remove my data", "clear my cache", "reset my profile"
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
        
        # Process clear data request
        for keyword in clear_data_keywords:
            if keyword.lower() in content.lower():
                await self.clear_user_data(message, user_id)
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
    
    async def process_thinking_request(self, message, content):
        """Process a request that requires specialized thinking"""
        guild_id = str(message.guild.id) if message.guild else "DM"
        user_id = str(message.author.id)
        conversation_id = f"{guild_id}:{message.channel.id}"
        
        # Update last activity time
        self.conversation_last_activity[conversation_id] = asyncio.get_event_loop().time()
        
        # Create tracking message
        initial_message = await message.reply("🧠 **Processing your request...**")
        
        try:
            # Detect reasoning type from content
            reasoning_type = await agent_service.detect_reasoning_type(content)
            
            # Check if user has workflow mode enabled
            user_prefs = await memory_service.get_user_preferences(user_id)
            use_workflow = user_prefs.get("use_workflow_mode", False)
            
            if use_workflow:
                try:
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
                        elif status == "tool_use":
                            tool_name = metadata.get("tool_name", "")
                            await initial_message.edit(content=f"🔧 **Using tool: {tool_name}**\n\nGathering information...")
                    
                    # Use workflow service for processing
                    response = await workflow_service.create_and_run_default_workflow(
                        user_query=content,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        update_callback=update_callback
                    )
                    
                    # Store the reasoning type used for this conversation
                    self.last_reasoning_type[conversation_id] = "workflow"
                    
                    # Format response with appropriate emoji
                    if isinstance(response, dict) and "agent_type" in response and "content" in response:
                        formatted_response, _ = await agent_service.format_with_agent_emoji(
                            response["content"], 
                            response["agent_type"]
                        )
                        
                        # Update emoji reactions for workflow mode with agent type
                        if self.emoji_reaction_cog:
                            await self.emoji_reaction_cog.update_reasoning_reactions(
                                initial_message, 
                                ["workflow", response["agent_type"]]
                            )
                    else:
                        formatted_response, _ = await agent_service.format_with_agent_emoji(
                            response, 
                            reasoning_type
                        )
                        
                        # Update emoji reactions for workflow mode
                        if self.emoji_reaction_cog:
                            await self.emoji_reaction_cog.update_reasoning_reactions(
                                initial_message, 
                                ["workflow"]
                            )
                    
                    # Edit the initial message with the final response
                    await initial_message.edit(content=formatted_response)
                    
                except Exception as e:
                    # Fallback to standard processing
                    print(f"Error in workflow mode, falling back to standard processing: {e}")
                    await initial_message.edit(content="🧠 **Processing with standard mode...**")
                    await self._process_with_standard_mode(message, content, initial_message, reasoning_type, user_id, conversation_id)
            else:
                # Use standard processing
                await self._process_with_standard_mode(message, content, initial_message, reasoning_type, user_id, conversation_id)
            
            return True
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in process_thinking_request: {error_traceback}")
            
            try:
                await initial_message.edit(content=f"❌ I encountered an error while processing your request: {str(e)[:1500]}")
            except:
                await message.channel.send(f"❌ I encountered an error while processing your request: {str(e)[:1500]}")
            
            return True
    
    async def _process_with_standard_mode(self, message, content, initial_message, reasoning_type, user_id, conversation_id):
        """Process a request using the standard agent mode"""
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
                if self.emoji_reaction_cog:
                    await self.emoji_reaction_cog.update_reasoning_reactions(initial_message, [agent_type])
            elif status == "tool_use":
                tool_name = metadata.get("tool_name", "")
                await initial_message.edit(content=f"🔧 **Using tool: {tool_name}**\n\nGathering information...")
        
        # Use agent service to process the query
        response = await agent_service.process_query(
            query=content,
            user_id=user_id,
            conversation_id=conversation_id,
            reasoning_type=reasoning_type,
            update_callback=update_callback
        )
        
        # Store the reasoning type used for this conversation
        self.last_reasoning_type[conversation_id] = reasoning_type
        
        # Format response with appropriate emoji
        formatted_response, _ = await agent_service.format_with_agent_emoji(response, reasoning_type)
        
        # Edit the initial message with the final response
        await initial_message.edit(content=formatted_response)
        
        # Update emoji reactions on the final message
        if self.emoji_reaction_cog:
            await self.emoji_reaction_cog.update_reasoning_reactions(initial_message, [reasoning_type])
    
    def _is_follow_up_query(self, query: str) -> bool:
        """Check if a query appears to be a follow-up question"""
        follow_up_patterns = [
            r'^(what|why|how|when|where|which|who|are|is|can|could|should|would|shall|will|do|does|tell|explain|elaborate)',
            r'^(and|but|so|or|as|if|then)',
            r'^(go on|continue|tell me more|more|more info|more about|elaborate|next|anything else|additional)',
            r'^(in addition|furthermore|also|besides|moreover)',
            r'^[^a-zA-Z0-9]?\s*[a-zA-Z]+\s*\?'  # Ends with a question mark
        ]
        
        # Check if any pattern matches
        return any(re.match(pattern, query.lower()) for pattern in follow_up_patterns)
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Listen for messages to process with reasoning"""
        # Skip messages from bots
        if message.author.bot:
            return
            
        # Check if the bot is mentioned or the message is a reply to the bot
        is_mentioned = self.bot.user in message.mentions
        is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author.id == self.bot.user.id
        
        if not (is_mentioned or is_reply_to_bot):
            # Not directed at the bot
            return
            
        # Get content without bot mention
        content = await message_service.smart_mention(message.clean_content, message, self.bot)
        
        # Handle empty content
        if not content.strip():
            await message.reply("How can I help you?")
            return
            
        # Try to handle as a natural language command first
        if await self.handle_natural_language_command(message, content):
            return
            
        # Process with appropriate reasoning
        try:
            await self.process_thinking_request(message, content)
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in on_message reasoning: {error_traceback}")
            await message.reply(f"❌ I encountered an error: {str(e)[:1500]}")
    
    @commands.hybrid_command(name="clear", description="Clear your conversation history and data")
    async def clear_data_command(self, ctx):
        """Clear your data from the bot's memory"""
        await self.clear_user_data(ctx, str(ctx.author.id))
    
    @commands.hybrid_command(name="reset", description="Reset the current conversation but keep your preferences")
    async def reset_conversation_command(self, ctx):
        """Reset the current conversation"""
        guild_id = str(ctx.guild.id) if ctx.guild else "DM"
        conversation_id = f"{guild_id}:{ctx.channel.id}"
        await self.reset_current_conversation(ctx, conversation_id)
    
    async def clear_user_data(self, message, user_id: str):
        """Clear a user's data from the system"""
        try:
            # Clear user data from all services
            await memory_service.clear_user_data(user_id)
            await agent_service.clear_user_data(user_id)
            
            # Remove from local tracking
            for conv_id in list(self.conversation_last_activity.keys()):
                if conv_id.endswith(f":{user_id}"):
                    del self.conversation_last_activity[conv_id]
                    if conv_id in self.active_conversations:
                        del self.active_conversations[conv_id]
                    if conv_id in self.last_reasoning_type:
                        del self.last_reasoning_type[conv_id]
            
            # Let the user know it's done
            await message.reply("✅ I've cleared all your data from my memory. Your previous conversations and preferences have been removed.")
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error clearing user data: {error_traceback}")
            await message.reply(f"❌ I encountered an error while clearing your data: {str(e)[:1500]}")
    
    async def reset_current_conversation(self, message, conversation_id: str):
        """Reset the current conversation"""
        try:
            # Reset conversation in memory service
            await memory_service.reset_conversation(conversation_id)
            
            # Reset conversation in agent service
            await agent_service.reset_conversation(conversation_id)
            
            # Reset local tracking
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]
            if conversation_id in self.last_reasoning_type:
                del self.last_reasoning_type[conversation_id]
            if conversation_id in self.conversation_last_activity:
                # Keep activity time but reset the timestamp
                self.conversation_last_activity[conversation_id] = asyncio.get_event_loop().time()
            
            # Let the user know it's done
            await message.reply("✅ I've reset our conversation. Let's start fresh!")
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error resetting conversation: {error_traceback}")
            await message.reply(f"❌ I encountered an error while resetting our conversation: {str(e)[:1500]}")

async def setup(bot):
    await bot.add_cog(ReasoningCog(bot)) 