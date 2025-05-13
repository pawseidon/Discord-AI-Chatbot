import discord
from discord.ext import commands
from discord import app_commands
import traceback
import asyncio
from typing import Dict, Any, Optional, List

from bot_utilities.mcp_utils import MCPToolsManager
from bot_utilities.monitoring import AgentMonitor
from bot_utilities.formatting_utils import chunk_message

class MCPAgentCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.mcp_manager = MCPToolsManager()
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
                
                # Track start time for monitoring
                start_time = asyncio.get_event_loop().time()
                
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
                
                # Run the agent with sequential thinking prompt
                response = await self.mcp_manager.run_simple_mcp_agent(
                    query=problem,
                    system_message=system_message
                )
                
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
                if len(chunks) > 1:
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
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
            
            await interaction.followup.send(f"💥 Error: I encountered a problem with sequential thinking. Please try again later.\n```{str(e)[:1500]}```")

async def setup(bot):
    await bot.add_cog(MCPAgentCog(bot)) 