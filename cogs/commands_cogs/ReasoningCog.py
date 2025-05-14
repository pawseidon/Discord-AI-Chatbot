import discord
from discord.ext import commands
import traceback
import asyncio
from typing import Dict, Any, Optional, List, Tuple

from bot_utilities.reasoning_utils import reasoning_manager
from bot_utilities.sequential_thinking import create_sequential_thinking
from bot_utilities.reflective_rag import SelfReflectiveRAG
from bot_utilities.ai_utils import get_ai_provider
from bot_utilities.monitoring import AgentMonitor
from bot_utilities.formatting_utils import chunk_message
from bot_utilities.mcp_utils import MCPToolsManager
from bot_utilities.agent_utils import AgentTools

class ReasoningCog(commands.Cog):
    """A cog for context-aware reasoning with automatic detection of reasoning types"""
    
    def __init__(self, bot):
        self.bot = bot
        self.sequential_thinking = create_sequential_thinking(llm_provider=None)
        self.mcp_manager = MCPToolsManager()
        self.server_rag_systems = {}
        self.monitor = AgentMonitor()
        self.agent_tools = AgentTools()
        self.active_conversations = {}
        self.last_reasoning_type = {}
        
        # Track last activity time for cache cleanup
        self.conversation_last_activity = {}
        
        # Schedule periodic cache cleanup
        self.bot.loop.create_task(self._periodic_cache_cleanup())
    
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
                    
                    # Reset conversation in reasoning manager
                    await reasoning_manager.reset_conversation(conv_id)
                
                print(f"Cleaned up {len(inactive_conversations)} inactive conversations")
                
            except Exception as e:
                print(f"Error in cache cleanup: {e}")
    
    def get_server_rag(self, guild_id: str) -> SelfReflectiveRAG:
        """Get or create a reflective RAG system for the server"""
        if guild_id not in self.server_rag_systems:
            self.server_rag_systems[guild_id] = SelfReflectiveRAG(guild_id)
        return self.server_rag_systems[guild_id]
    
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
        
        # Process reasoning preference request
        for keyword in preference_keywords:
            if keyword.lower() in content.lower():
                # Get reasoning type from content
                reasoning_type = None
                for rtype, keywords in reasoning_manager.detector.REASONING_KEYWORDS.items():
                    for kw in keywords:
                        if kw in content.lower():
                            reasoning_type = rtype
                            break
                    if reasoning_type:
                        break
                
                # Also check for emoji indicators
                if not reasoning_type:
                    for rtype, emoji in reasoning_manager.detector.REASONING_EMOJIS.items():
                        if emoji in content:
                            reasoning_type = rtype
                            break
                
                if reasoning_type:
                    emoji = reasoning_manager.detector.get_emoji_for_type(reasoning_type)
                    success = reasoning_manager.detector.set_user_preference(user_id, reasoning_type)
                    
                    if success:
                        response = f"{emoji} I've set your preferred reasoning mode to **{reasoning_type}**. I'll try to use this mode when appropriate for your queries."
                        await message.reply(response)
                        return True
        
        # Process reasoning info request
        for keyword in info_keywords:
            if keyword.lower() in content.lower():
                # Create a text-based explanation of reasoning modes
                info_text = "**AI Reasoning Modes**\n\nHere are the different reasoning modes available and how to use them:\n\n"
                
                reasoning_modes = [
                    ("Sequential Thinking 🧠", "Breaks down complex problems into step-by-step analysis. Best for detailed problem-solving.\n*Trigger with: \"step by step\", \"analyze\"*"),
                    ("Information Retrieval 🔍", "Searches for specific information and facts. Best for factual queries.\n*Trigger with: \"search\", \"find\", \"what is\"*"),
                    ("Knowledge Base 📚", "Provides detailed explanations of concepts. Best for educational content.\n*Trigger with: \"explain\", \"define\", \"describe\"*"),
                    ("Verification ✅", "Fact-checks information and assesses accuracy. Best for validating claims.\n*Trigger with: \"verify\", \"fact-check\", \"confirm\"*"),
                    ("Graph-of-Thought 🕸️", "Maps interconnected concepts for complex topics. Best for exploring relationships.\n*Trigger with: \"connections\", \"relationships\", \"map\"*"),
                    ("Chain-of-Thought ⛓️", "Shows logical progression of reasoning. Best for following logical arguments.\n*Trigger with: \"logical sequence\", \"chain of thought\"*"),
                    ("ReAct Reasoning 🔄", "Combines reasoning with actions. Best for tasks requiring both thought and execution.\n*Trigger with: \"implement\", \"execute\", \"take action\"*"),
                    ("Creative Thinking 🎨", "Emphasizes imagination and originality. Best for creative content.\n*Trigger with: \"create\", \"imagine\", \"design\"*"),
                    ("Calculation Mode 🔢", "Focuses on mathematical operations. Best for math problems.\n*Trigger with: \"calculate\", \"compute\", \"solve\"*"),
                    ("Planning Mode 📋", "Develops organized strategies. Best for project planning.\n*Trigger with: \"plan\", \"organize\", \"strategy\"*"),
                    ("Multi-Agent Thinking 👥", "Considers multiple perspectives. Best for balanced analysis.\n*Trigger with: \"different perspectives\", \"pros and cons\"*"),
                    ("Step-Back Thinking 🔎", "Takes a broad, holistic view. Best for big-picture analysis.\n*Trigger with: \"broader perspective\", \"big picture\"*"),
                ]
                
                for name, description in reasoning_modes:
                    info_text += f"**{name}**\n{description}\n\n"
                
                info_text += "**How to Use Reasoning Modes**\n" + \
                             "You can trigger different reasoning modes in three ways:\n" + \
                             "1. Use keywords in your message (listed above)\n" + \
                             "2. Include the emoji at the beginning of your message\n" + \
                             "3. Set a preferred mode with 'set my reasoning mode to [mode]'\n\n" + \
                             "The bot will automatically detect the most appropriate reasoning mode based on your message and conversation context."
                
                # Split into chunks if needed
                chunks = chunk_message(info_text)
                for chunk in chunks:
                    await message.channel.send(chunk)
                return True
        
        return False
    
    async def process_thinking_request(self, message, content):
        """Process a thinking request with automatic reasoning selection"""
        # Show typing indicator
        async with message.channel.typing():
            # Get server and user info
            guild_id = str(message.guild.id) if message.guild else "DM"
            user_id = str(message.author.id)
            conversation_id = f"{guild_id}:{message.channel.id}"
            
            # Update last activity time
            self.conversation_last_activity[conversation_id] = asyncio.get_event_loop().time()
            
            # Initial message showing we're processing
            processing_message = await message.reply("🤔 Processing your request...")
            
            # Track start time for monitoring
            start_time = asyncio.get_event_loop().time()
            
            # Get conversation history if available
            conversation_history = self.active_conversations.get(conversation_id, [])
            
            # Get current reasoning type if any
            current_reasoning = self.last_reasoning_type.get(conversation_id)
            
            # Detect most appropriate reasoning type
            reasoning_result = await reasoning_manager.process_query(
                query=content,
                conversation_id=conversation_id,
                user_id=user_id,
                conversation_history=conversation_history
            )
            
            reasoning_type = reasoning_result['reasoning_type']
            reasoning_emoji = reasoning_result['emoji']
            is_transition = reasoning_result['transition']
            
            # Store last reasoning type used for this conversation
            self.last_reasoning_type[conversation_id] = reasoning_type
            
            # Prepare message about reasoning mode
            reasoning_msg = f"{reasoning_emoji} Using {reasoning_type} reasoning"
            if is_transition and current_reasoning:
                reasoning_msg += f" (switched from {current_reasoning})"
                
            # Update message to show detected reasoning type
            await processing_message.edit(content=f"{reasoning_msg} to process: `{content[:50]}{'...' if len(content) > 50 else ''}`")
            
            # Generate response using the appropriate reasoning method
            response = await self._generate_response_with_reasoning_type(
                query=content,
                reasoning_type=reasoning_type,
                user_id=user_id,
                guild_id=guild_id,
                conversation_id=conversation_id
            )
            
            # Calculate execution time for monitoring
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Format the response with reasoning indicators
            formatted_response = await reasoning_manager.format_response(
                response=response,
                reasoning_type=reasoning_type,
                include_reasoning_details=(reasoning_type in ['sequential', 'graph', 'verification', 'cot', 'step_back'])
            )
            
            # Log the interaction for monitoring
            asyncio.create_task(self.monitor.log_interaction(
                command_name="natural_language_reasoning",
                user_id=user_id,
                server_id=guild_id,
                execution_time=execution_time,
                success=True,
                metadata={
                    "reasoning_type": reasoning_type,
                    "confidence": reasoning_result['confidence'],
                    "is_transition": is_transition
                }
            ))
            
            # Update the conversation history
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = []
            # Add to history (max 10 entries to avoid context overflow)
            self.active_conversations[conversation_id] = (
                self.active_conversations[conversation_id] + [content])[-10:]
            
            # Split into chunks if needed
            chunks = chunk_message(formatted_response)
            if len(chunks) > 1:
                await processing_message.edit(content=chunks[0])
                
                for chunk in chunks[1:]:
                    await message.channel.send(chunk)
            else:
                await processing_message.edit(content=formatted_response)
            
            return True
    
    async def _generate_response_with_reasoning_type(self, 
                                                query: str, 
                                                reasoning_type: str,
                                                user_id: str,
                                                guild_id: str,
                                                conversation_id: str) -> str:
        """Generate a response using the specified reasoning type"""
        try:
            # Get AI provider for use with different reasoning methods
            llm_provider = await get_ai_provider()
            
            if reasoning_type == 'sequential':
                # Use sequential thinking with enhanced revision capabilities
                await self.sequential_thinking.set_llm_provider(llm_provider)
                success, response = await self.sequential_thinking.run(
                    problem=query,
                    context={"user_id": user_id, "server_id": guild_id},
                    prompt_style="sequential",
                    num_thoughts=5,
                    temperature=0.2,
                    enable_revision=True  # Enable thought revision capabilities
                )
                if not success:
                    # Fallback to standard response
                    response = f"I had trouble with sequential thinking for this query. Here's my best response:\n\n{response}"
                    
            elif reasoning_type == 'rag':
                # Use RAG (Retrieval-Augmented Generation)
                rag_system = self.get_server_rag(guild_id)
                
                # Get relevant context
                context = await rag_system.format_reflective_results(query)
                
                # If no context was found, try a web search
                if not context or len(context.strip()) < 50:
                    try:
                        # Try to get web information
                        web_results = await self.agent_tools.search_web(query, max_results=3)
                        if web_results:
                            context = f"Web search results:\n\n{web_results}"
                    except Exception as e:
                        print(f"Error during web search fallback: {e}")
                
                # Generate a response using the retrieved information
                prompt = f"""You are providing information based on retrieved content.
                
                Retrieved Information:
                {context if context else "No specific information was found in the knowledge base."}
                
                Query: {query}
                
                Provide a comprehensive answer to the query based on the retrieved information.
                If the retrieved information doesn't contain relevant details, acknowledge what you know
                and what information is missing. Cite sources when available.
                """
                
                # Get response from the LLM
                if context:
                    response = await llm_provider.async_call(prompt)
                else:
                    response = "I don't have specific information about this in my knowledge base. " + \
                              await llm_provider.async_call(f"Please answer this query as best you can: {query}")
                              
            elif reasoning_type == 'graph':
                # Use Graph-of-Thought reasoning
                await self.sequential_thinking.set_llm_provider(llm_provider)
                success, response = await self.sequential_thinking.run(
                    problem=query,
                    context={"user_id": user_id, "server_id": guild_id},
                    prompt_style="got",  # Graph of Thought
                    num_thoughts=5,
                    temperature=0.3
                )
                if not success:
                    # Fallback to standard response
                    response = f"I had trouble with graph-of-thought reasoning for this query. Here's my best response:\n\n{response}"
                    
            elif reasoning_type == 'verification':
                # Use Chain-of-Verification reasoning
                await self.sequential_thinking.set_llm_provider(llm_provider)
                success, response = await self.sequential_thinking.run(
                    problem=query,
                    context={"user_id": user_id, "server_id": guild_id},
                    prompt_style="cov",  # Chain of Verification
                    num_thoughts=4,
                    temperature=0.1  # Lower temperature for factual accuracy
                )
                if not success:
                    # Fallback to standard response
                    response = f"I had trouble with verification reasoning for this query. Here's my best response:\n\n{response}"
                    
            elif reasoning_type == 'cot':
                # Use Chain-of-Thought reasoning
                await self.sequential_thinking.set_llm_provider(llm_provider)
                success, response = await self.sequential_thinking.run(
                    problem=query,
                    context={"user_id": user_id, "server_id": guild_id},
                    prompt_style="cot",  # Chain of Thought
                    num_thoughts=3,
                    temperature=0.2
                )
                if not success:
                    # Fallback to standard response
                    response = f"I had trouble with chain-of-thought reasoning for this query. Here's my best response:\n\n{response}"
                    
            elif reasoning_type == 'react':
                # Use ReAct reasoning with MCP tools
                response = await self.mcp_manager.run_simple_mcp_agent(
                    query=query,
                    system_message=f"""You are a helpful assistant using ReAct (Reasoning+Acting) to solve problems.
                    Follow this process:
                    1. Reason about the problem step by step
                    2. Consider what actions might be helpful
                    3. Take actions using available tools when appropriate
                    4. Observe results and continue reasoning
                    
                    User ID: {user_id}
                    Server ID: {guild_id}
                    
                    Solve the user's problem by combining careful reasoning with appropriate actions.
                    """
                )
                
            elif reasoning_type == 'creative':
                # Use creative mode
                prompt = f"""You are in creative mode, focusing on generating imaginative, original content.
                
                Request: {query}
                
                Unleash your creativity to fulfill this request. Feel free to think outside conventional
                boundaries and explore unique possibilities. Generate content that is original, engaging,
                and tailored to the request.
                """
                
                # Use slightly higher temperature for creative responses
                response = await llm_provider.async_call(prompt, temperature=0.7)
                
            elif reasoning_type == 'knowledge':
                # Use knowledge base mode
                prompt = f"""You are in knowledge base mode, focusing on providing accurate, educational
                information in a structured, clear manner.
                
                Query: {query}
                
                Please provide a comprehensive, well-structured explanation that:
                - Covers the core concepts thoroughly
                - Organizes information logically
                - Includes relevant examples where helpful
                - Defines any specialized terminology
                - Takes an educational approach
                """
                
                response = await llm_provider.async_call(prompt, temperature=0.1)
                
            elif reasoning_type == 'step_back':
                # Use step-back prompting
                await self.sequential_thinking.set_llm_provider(llm_provider)
                success, response = await self.sequential_thinking.run(
                    problem=query,
                    context={"user_id": user_id, "server_id": guild_id},
                    prompt_style="step_back",
                    num_thoughts=4,
                    temperature=0.3
                )
                if not success:
                    # Fallback to standard response
                    response = f"I had trouble with step-back reasoning for this query. Here's my best response:\n\n{response}"
            
            elif reasoning_type == 'multi_agent':
                # Use multi-agent debate
                prompt = f"""You are conducting a multi-agent debate to analyze this query from different perspectives.
                
                Query: {query}
                
                Use these different perspectives to examine the topic:
                1. Expert perspective - Consider the technical/academic viewpoint
                2. Critical perspective - Identify potential issues or limitations
                3. Creative perspective - Explore innovative angles or solutions
                4. Practical perspective - Focus on real-world applications
                
                First analyze from each perspective, then synthesize a comprehensive response.
                """
                
                response = await llm_provider.async_call(prompt, temperature=0.4)
                
            else:
                # Default to conversational (simple response)
                prompt = f"""You are having a helpful conversation. Please respond to this message:
                
                {query}
                """
                
                response = await llm_provider.async_call(prompt, temperature=0.5)
                
            return response
            
        except Exception as e:
            print(f"Error generating response with reasoning type '{reasoning_type}': {e}")
            traceback.print_exc()
            return f"I encountered an error while processing your request: {str(e)}"
    
    # Listen for all messages to provide fully context-aware responses
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Listen for all messages and process them with context-aware reasoning"""
        # Skip if message is from a bot
        if message.author.bot:
            return
            
        # Check if the bot is mentioned or this is a DM
        is_mentioned = self.bot.user in message.mentions
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_reply = message.reference and message.reference.resolved and message.reference.resolved.author == self.bot.user
        
        # Extract the content without the mention
        content = message.content
        if is_mentioned:
            # Remove the mention from the content
            content = content.replace(f'<@{self.bot.user.id}>', '').strip()
            content = content.replace(f'<@!{self.bot.user.id}>', '').strip()
            
        if not content and not is_dm and not is_reply:
            return  # Skip empty messages unless in DM
            
        # Check if we should process this message
        should_process = is_mentioned or is_dm or is_reply
        
        # Also check for direct addressing (e.g., "hey bot", "bot")
        bot_name_triggers = [
            self.bot.user.name.lower(), 
            "bot", 
            "assistant", 
            "ai", 
            "help me", 
            "can you", 
            "could you",
            "thinking mode",
            "reasoning mode"
        ]
        
        for trigger in bot_name_triggers:
            if trigger in content.lower() and len(content) > len(trigger) + 3:  # +3 to avoid false positives
                should_process = True
                break
        
        if should_process:
            # First check for natural language commands
            command_handled = await self.handle_natural_language_command(message, content)
            
            if not command_handled and len(content) >= 2:  # Only process reasonable length requests
                # Process as a thinking request with automatic reasoning selection
                await self.process_thinking_request(message, content)
                
            # Record that we processed this message so we don't process it again
            guild_id = str(message.guild.id) if message.guild else "DM"
            conversation_id = f"{guild_id}:{message.channel.id}"
            
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = []
                
            # Add to conversation history for context
            self.active_conversations[conversation_id] = (
                self.active_conversations[conversation_id] + [content]
            )[-10:]  # Keep last 10 messages

    @commands.command(name="bonk")
    async def reset_conversation_command(self, ctx):
        """Reset the conversation history and reasoning cache"""
        guild_id = str(ctx.guild.id) if ctx.guild else "DM"
        conversation_id = f"{guild_id}:{ctx.channel.id}"
        
        # Clear conversation history
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        
        # Clear reasoning type
        if conversation_id in self.last_reasoning_type:
            del self.last_reasoning_type[conversation_id]
            
        # Clear activity tracker
        if conversation_id in self.conversation_last_activity:
            del self.conversation_last_activity[conversation_id]
        
        # Reset conversation in reasoning manager
        await reasoning_manager.reset_conversation(conversation_id)
        
        await ctx.send("🔄 Conversation history has been reset. Starting fresh!")

async def setup(bot):
    await bot.add_cog(ReasoningCog(bot)) 