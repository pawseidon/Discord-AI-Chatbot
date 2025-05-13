import discord
from discord.ext import commands
from discord import app_commands
import traceback
import asyncio
from typing import Dict, Any, Optional, List

from bot_utilities.mcp_utils import MCPToolsManager
from bot_utilities.monitoring import AgentMonitor
from bot_utilities.formatting_utils import chunk_message
from bot_utilities.sequential_thinking import create_sequential_thinking

class MCPAgentCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.mcp_manager = MCPToolsManager()
        self.sequential_thinking = create_sequential_thinking(llm_provider=None)
        self.monitor = AgentMonitor()
    
    async def discord_read_func(self, interaction: discord.Interaction = None) -> Dict[str, Any]:
        """Read function for MCP that includes Discord context"""
        context = {}
        
        if interaction:
            # Add basic user and server info
            context["user"] = {
                "id": str(interaction.user.id),
                "name": interaction.user.display_name,
                "roles": [str(role.id) for role in interaction.user.roles] if hasattr(interaction.user, "roles") else [],
            }
            
            if interaction.guild:
                context["server"] = {
                    "id": str(interaction.guild.id),
                    "name": interaction.guild.name,
                    "member_count": interaction.guild.member_count,
                }
            
            # Add channel info
            context["channel"] = {
                "id": str(interaction.channel.id),
                "name": interaction.channel.name if hasattr(interaction.channel, "name") else "DM",
                "type": str(interaction.channel.type),
            }
        
        return context
    
    async def discord_write_func(self, data: Dict[str, Any]) -> None:
        """Write function for MCP that handles Discord-specific outputs"""
        # This would be used to handle custom interactions with Discord
        # For now, just print the data
        print(f"MCP Tool Output for Discord: {data}")
    
    @app_commands.command(name="mcp-agent", description="Use an agent with MCP tools")
    @app_commands.describe(query="What would you like the MCP agent to help you with?")
    async def mcp_agent_command(self, interaction: discord.Interaction, query: str):
        """
        Interact with an AI agent equipped with specialized MCP tools
        
        Parameters:
        query (str): The task or question to handle
        """
        # Defer reply to give time for processing
        await interaction.response.defer()
        
        try:
            # Show typing indicator to indicate processing
            async with interaction.channel.typing():
                # Start a spinner indicator for long-running tasks
                message = await interaction.followup.send(f"🔄 Working on your request: `{query}`")
                
                # Track start time for monitoring
                start_time = asyncio.get_event_loop().time()
                
                # Run the MCP agent with the query
                response = await self.mcp_manager.run_simple_mcp_agent(query)
                
                # Calculate execution time for monitoring
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Log the interaction for monitoring
                user_id = str(interaction.user.id)
                server_id = str(interaction.guild.id) if interaction.guild else "DM"
                asyncio.create_task(self.monitor.log_interaction(
                    command_name="mcp_agent_command",
                    user_id=user_id,
                    server_id=server_id,
                    execution_time=execution_time,
                    success=True
                ))
                
                # Format the response for Discord
                # Split into chunks if needed
                chunks = chunk_message(response)
                if len(chunks) > 1:
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in MCP agent command: {error_traceback}")
            
            # Log error for monitoring
            if interaction and interaction.guild:
                asyncio.create_task(self.monitor.log_interaction(
                    command_name="mcp_agent_command",
                    user_id=str(interaction.user.id),
                    server_id=str(interaction.guild.id),
                    execution_time=0,
                    success=False,
                    error=str(e)
                ))
            
            await interaction.followup.send(f"💥 Error: I encountered a problem with the MCP agent. Please try again later.\n```{str(e)[:1500]}```")
    
    @app_commands.command(name="sequential-thinking", description="Use an agent with sequential thinking capabilities")
    @app_commands.describe(problem="What problem would you like to solve with sequential thinking?")
    async def sequential_thinking_command(self, interaction: discord.Interaction, problem: str):
        """
        Solve complex problems using an agent that applies sequential thinking
        
        Parameters:
        problem (str): The problem to solve
        """
        # Defer reply to give time for processing
        await interaction.response.defer()
        
        try:
            # Show typing indicator to indicate processing
            async with interaction.channel.typing():
                # Start a spinner indicator for long-running tasks
                message = await interaction.followup.send(f"🔄 Solving: `{problem}`")
                message_reference = message.id if message else None  # Store message ID for reference
                
                # Track start time for monitoring
                start_time = asyncio.get_event_loop().time()
                
                # Build context for enhanced reasoning
                context = {
                    "user": {
                        "id": str(interaction.user.id),
                        "name": interaction.user.display_name,
                        "roles": [str(role.id) for role in interaction.user.roles] if hasattr(interaction.user, "roles") else [],
                    },
                    "channel": {
                        "name": interaction.channel.name if hasattr(interaction.channel, "name") else "DM",
                        "type": str(interaction.channel.type),
                    },
                    "guild": {
                        "name": interaction.guild.name if interaction.guild else "DM",
                        "id": str(interaction.guild.id) if interaction.guild else None,
                    }
                }
                
                # Detect if the user wants to use knowledge bases or tools
                knowledge_base = None
                use_tools = False
                
                # Check for knowledge base references
                if any(term in problem.lower() for term in ["knowledge base", "kb", "rag", "kag", "crag"]):
                    knowledge_base = "default_kb"
                
                # Check for tool integration keywords
                for tool_keyword in ["search", "web", "browse", "information", "data", "lookup", "db", "database", 
                                 "fetch", "retrieve", "knowledge", "graph", "query", "store", "vector"]:
                    if tool_keyword in problem.lower():
                        use_tools = True
                        break
                
                # Update progress message if possible
                try:
                    # Safely check if message exists and is accessible
                    if message:
                        try:
                            if use_tools:
                                await message.edit(content=f"🔄 Solving with advanced tools: `{problem}`")
                        except discord.errors.NotFound:
                            print(f"Warning: Message not found (ID: {message_reference}), creating a new one")
                            try:
                                message = await interaction.followup.send(f"🔄 Continuing to solve: `{problem}`")
                                message_reference = message.id if message else None
                            except Exception as e:
                                print(f"Error creating replacement message: {e}")
                except Exception as e:
                    print(f"Warning: Could not edit message, continuing with processing: {e}")
                
                # Get LLM provider for sequential thinking
                llm_provider = None
                try:
                    from bot_utilities.ai_utils import get_ai_provider
                    llm_provider = await get_ai_provider()
                    # Set the LLM provider to our sequential thinking instance
                    await self.sequential_thinking.set_llm_provider(llm_provider)
                except Exception as e:
                    print(f"Error getting AI provider: {e}")
                    # Will use fallback mechanisms in sequential_thinking.py
                
                # Run sequential thinking using our new implementation
                success, response = await self.sequential_thinking.run(
                    problem=problem,
                    context=context, 
                    # Try different styles if needed
                    prompt_style="sequential",
                    # If tools are requested, we'll use more thinking steps
                    num_thoughts=7 if use_tools else 5,
                    temperature=0.2,
                    # Adjust max tokens based on complexity
                    max_tokens=2500 if use_tools else 2000,
                    # Set a reasonable timeout
                    timeout=90
                )
                
                # If our sequential thinking failed, try MCP fallback
                if not success:
                    try:
                        # Create a system message that encourages sequential thinking
                        system_message = """You are an AI assistant that solves problems using sequential thinking.
                        
                        Approach each problem by:
                        
                        1. Breaking it down into smaller, manageable parts
                        2. Addressing each part in a logical sequence
                        3. Building on previous steps to reach the final solution
                        4. Checking your work at each stage
                        5. Summarizing your approach and final answer
                        
                        Think step-by-step and show your reasoning process clearly. Explicitly state when you're moving from one step to the next.
                        """
                        
                        # Try with timeout to prevent hanging
                        response = await asyncio.wait_for(
                            self.mcp_manager.run_simple_mcp_agent(
                                query=problem,
                                system_message=system_message
                            ),
                            timeout=60
                        )
                        success = True
                    except asyncio.TimeoutError:
                        # If MCP times out
                        print("MCP sequential thinking timed out.")
                        response = "I encountered a timeout while performing sequential thinking. Please try again with a simpler query."
                    except Exception as e:
                        print(f"MCP agent error in sequential_thinking_command: {e}")
                        # Simply provide a clean error message to the user
                        response = "Error: I encountered a problem with sequential thinking. Please try again later."
                
                # Calculate execution time for monitoring
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Log the interaction for monitoring
                user_id = str(interaction.user.id)
                server_id = str(interaction.guild.id) if interaction.guild else "DM"
                asyncio.create_task(self.monitor.log_interaction(
                    command_name="sequential_thinking_command",
                    user_id=user_id,
                    server_id=server_id,
                    execution_time=execution_time,
                    success=True
                ))
                
                # Format the response for Discord
                # Split into chunks if needed
                chunks = chunk_message(response)
                
                # Robust message handling
                try:
                    if len(chunks) > 1:
                        # Update initial message with first chunk
                        first_chunk_sent = False
                        try:
                            # Check if message is still valid
                            if message:
                                try:
                                    await message.edit(content=chunks[0])
                                    first_chunk_sent = True
                                except discord.errors.NotFound:
                                    print(f"Message with ID {message_reference} not found, sending new one")
                                    raise discord.errors.NotFound("Message not found", None)
                            else:
                                # Try fetch by ID if needed
                                try:
                                    if message_reference:
                                        message = await interaction.channel.fetch_message(message_reference)
                                        await message.edit(content=chunks[0])
                                        first_chunk_sent = True
                                except discord.errors.NotFound:
                                    # Message truly gone, send new one
                                    message = await interaction.followup.send(chunks[0])
                                    first_chunk_sent = True
                        except discord.errors.NotFound:
                            print(f"Message with ID {message_reference} not found, sending new one")
                            message = await interaction.followup.send(chunks[0])
                            first_chunk_sent = True
                        except Exception as e:
                            print(f"Error updating first chunk: {e}")
                            try:
                                message = await interaction.followup.send(chunks[0])
                                first_chunk_sent = True
                            except Exception as e2:
                                print(f"Critical error sending first chunk: {e2}")
                        
                        # Final attempt if needed
                        if not first_chunk_sent:
                            try:
                                await interaction.followup.send(chunks[0])
                            except Exception as e:
                                print(f"Final attempt to send first chunk failed: {e}")
                        
                        # Send additional chunks as separate messages
                        for chunk in chunks[1:]:
                            try:
                                await interaction.channel.send(chunk)
                            except Exception as e:
                                print(f"Error sending chunk: {e}")
                                # Try again after a brief delay
                                try:
                                    await asyncio.sleep(1)
                                    await interaction.channel.send(chunk)
                                except:
                                    pass
                    else:
                        # Single update if content fits
                        sent_successfully = False
                        try:
                            # Try editing original message
                            if message:
                                try:
                                    await message.edit(content=chunks[0])
                                    sent_successfully = True
                                except discord.errors.NotFound:
                                    print("Message not accessible, trying alternate methods")
                                    raise discord.errors.NotFound("Message not found", None)
                            else:
                                # Try to fetch by ID if we have it
                                try:
                                    if message_reference:
                                        message = await interaction.channel.fetch_message(message_reference)
                                        await message.edit(content=chunks[0])
                                        sent_successfully = True
                                except:
                                    # Message truly gone, send new one
                                    await interaction.followup.send(chunks[0])
                                    sent_successfully = True
                        except discord.errors.NotFound:
                            # Message not found, send a new one
                            await interaction.followup.send(chunks[0])
                            sent_successfully = True
                        except Exception as e:
                            print(f"Error updating message: {e}")
                            try:
                                await interaction.followup.send(chunks[0])
                                sent_successfully = True
                            except Exception as e2:
                                print(f"Critical error sending message: {e2}")
                        
                        # Final attempt if needed
                        if not sent_successfully:
                            try:
                                await interaction.followup.send(chunks[0])
                            except Exception as e:
                                print(f"Final attempt to send message failed: {e}")
                except Exception as e:
                    print(f"Unexpected error in message handling: {e}")
                    # Last resort attempt
                    try:
                        await interaction.followup.send("I completed my analysis but encountered an error sending the results. Please try again.")
                    except:
                        pass
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in sequential thinking command: {error_traceback}")
            
            # Log error for monitoring
            if interaction and interaction.guild:
                asyncio.create_task(self.monitor.log_interaction(
                    command_name="sequential_thinking_command",
                    user_id=str(interaction.user.id),
                    server_id=str(interaction.guild.id),
                    execution_time=0,
                    success=False,
                    error=str(e)
                ))
            
            # Try to send error message with robust handling
            try:
                await interaction.followup.send("Error: I encountered a problem with sequential thinking. Please try again later.")
            except:
                # If that fails, try channel send
                try:
                    await interaction.channel.send("Error: I encountered a problem with sequential thinking. Please try again later.")
                except:
                    pass

    @app_commands.command(name="connect-info", description="Connect information from multiple sources to solve complex problems")
    @app_commands.describe(
        topic="The main topic or question you want to explore",
        source1="First information source (optional)",
        source2="Second information source (optional)"
    )
    async def connect_info_command(self, interaction: discord.Interaction, 
                                 topic: str, 
                                 source1: Optional[str] = None, 
                                 source2: Optional[str] = None):
        """
        Analyze and connect information from multiple sources to provide comprehensive insights
        
        Parameters:
        topic (str): The main topic or question to explore
        source1 (str, optional): First information source (document, website, or data source)
        source2 (str, optional): Second information source (document, website, or data source)
        """
        # Defer reply to give time for processing
        await interaction.response.defer()
        
        try:
            # Show typing indicator to indicate processing
            async with interaction.channel.typing():
                # Start a spinner indicator for long-running tasks
                message = await interaction.followup.send(f"🔄 Connecting information on: `{topic}`")
                
                # Track start time for monitoring
                start_time = asyncio.get_event_loop().time()
                
                # Build context from sources
                sources = []
                if source1:
                    sources.append(source1)
                if source2:
                    sources.append(source2)
                
                # Create enhanced context object
                context = {
                    "user": {
                        "id": str(interaction.user.id),
                        "name": interaction.user.display_name
                    },
                    "channel": {
                        "name": interaction.channel.name if hasattr(interaction.channel, "name") else "DM",
                        "type": str(interaction.channel.type)
                    },
                    "guild": {
                        "name": interaction.guild.name if interaction.guild else "DM",
                        "id": str(interaction.guild.id) if interaction.guild else None
                    },
                    "sources": sources
                }
                
                # Set problem description for sequential thinking
                problem = f"""
                I need to analyze and connect information about: {topic}
                
                User's specific sources to consider:
                {', '.join(sources) if sources else 'No specific sources provided, use general knowledge.'}
                
                My goal is to:
                1. Understand the core concepts related to this topic
                2. Identify key relationships between different aspects
                3. Connect information from different sources
                4. Draw meaningful conclusions
                5. Provide an insightful, comprehensive analysis
                """
                
                # Get LLM provider for sequential thinking
                llm_provider = None
                try:
                    from bot_utilities.ai_utils import get_ai_provider
                    llm_provider = await get_ai_provider()
                    # Set the LLM provider to our sequential thinking instance
                    await self.sequential_thinking.set_llm_provider(llm_provider)
                except Exception as e:
                    print(f"Error getting AI provider: {e}")
                
                # Run sequential thinking with increased depth
                success, response = await self.sequential_thinking.run(
                    problem=problem,
                    context=context,
                    # Use more structured sequential approach for connecting info
                    prompt_style="sequential",
                    # Use more thoughts for complex information connection
                    num_thoughts=8,
                    # Higher temperature for more creative connections
                    temperature=0.3,
                    # Need more tokens for comprehensive analysis
                    max_tokens=3000,
                    # Longer timeout for complex processing
                    timeout=120
                )
                
                # If failed, use backup approach
                if not success:
                    # Create a specialized system message for information integration
                    system_message = """You are an AI assistant specialized in connecting information from multiple sources.
                    
                    When analyzing a topic across different sources:
                    
                    1. Extract key concepts and facts from each source
                    2. Identify areas of agreement and contradiction
                    3. Synthesize a cohesive understanding that reconciles different perspectives
                    4. Draw connections that might not be obvious from any single source
                    5. Present a balanced view that incorporates all relevant information
                    
                    Use a structured approach and be transparent about your reasoning process. Cite specific information from sources when applicable.
                    """
                    
                    # Fall back to MCP agent with specialized system message
                    try:
                        response = await asyncio.wait_for(
                            self.mcp_manager.run_simple_mcp_agent(
                                query=f"Connect and analyze information about: {topic}. Sources to consider: {', '.join(sources) if sources else 'general knowledge'}",
                                system_message=system_message
                            ),
                            timeout=90
                        )
                        success = True
                    except Exception as e:
                        print(f"Error in backup approach for connect-info: {e}")
                        response = "I encountered an error while trying to connect information from multiple sources. Please try again with a more specific query."
                
                # Calculate execution time for monitoring
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Log the interaction for monitoring
                asyncio.create_task(self.monitor.log_interaction(
                    command_name="connect_info_command",
                    user_id=str(interaction.user.id),
                    server_id=str(interaction.guild.id) if interaction.guild else "DM",
                    execution_time=execution_time,
                    success=success
                ))
                
                # Format and send the response
                chunks = chunk_message(response)
                if len(chunks) > 1:
                    await message.edit(content=chunks[0])
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in connect-info command: {error_traceback}")
            
            # Log error
            if interaction and interaction.guild:
                asyncio.create_task(self.monitor.log_interaction(
                    command_name="connect_info_command",
                    user_id=str(interaction.user.id),
                    server_id=str(interaction.guild.id),
                    execution_time=0,
                    success=False,
                    error=str(e)
                ))
            
            await interaction.followup.send(f"💥 Error: I encountered a problem connecting information. Please try again with more specific sources.\n```{str(e)[:1500]}```")

async def setup(bot):
    await bot.add_cog(MCPAgentCog(bot)) 