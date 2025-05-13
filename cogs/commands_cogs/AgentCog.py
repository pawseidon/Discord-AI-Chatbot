import discord
from discord.ext import commands
from discord import app_commands
import traceback
import asyncio

from bot_utilities.agent_utils import run_agent
from bot_utilities.config_loader import config

class AgentCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        
    @app_commands.command(name="agent", description="Interact with the agent to perform tasks")
    @app_commands.describe(query="What would you like the agent to help you with?")
    async def agent_command(self, interaction: discord.Interaction, query: str):
        """
        Use the agent to perform tasks and answer complex questions
        
        Parameters:
        query (str): The question or task for the agent
        """
        # Defer the response since agent processing might take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Show typing indicator to indicate processing
            async with interaction.channel.typing():
                # Start a spinner indicator for long-running tasks
                message = await interaction.followup.send(f"🔄 Working on your request: `{query}`")
                
                # Run the agent with the query
                response = await run_agent(query)
                
                # Format the response for Discord
                # Split into chunks if needed
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in agent command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while processing your request. Please try again later.\n```{str(e)[:1500]}```")
    
    @app_commands.command(name="research", description="Research a topic in depth using multiple sources")
    @app_commands.describe(topic="What topic would you like to research?")
    async def research_command(self, interaction: discord.Interaction, topic: str):
        """
        Use the agent to perform in-depth research on a topic
        
        Parameters:
        topic (str): The topic to research
        """
        # Defer the response since research will take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Show typing indicator to indicate processing
            async with interaction.channel.typing():
                # Start a spinner indicator for long-running tasks
                message = await interaction.followup.send(f"🔍 Researching: `{topic}`\nThis may take a minute or two...")
                
                # Run the agent with a research-focused prompt
                research_prompt = f"I need a comprehensive research summary on {topic}. Please search for the most recent and reliable information, analyze multiple perspectives, and provide a detailed overview with cited sources."
                response = await run_agent(research_prompt)
                
                # Format the response for Discord
                # Split into chunks if needed
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in research command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while researching. Please try again later.\n```{str(e)[:1500]}```")

    @app_commands.command(name="automate", description="Automate a complex task with step-by-step guidance")
    @app_commands.describe(task="Describe the task you want to automate")
    async def automate_command(self, interaction: discord.Interaction, task: str):
        """
        Automate complex tasks with step-by-step guidance
        
        Parameters:
        task (str): The task to automate
        """
        # Defer the response since automation planning will take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Show typing indicator to indicate processing
            async with interaction.channel.typing():
                # Start a spinner indicator for long-running tasks
                message = await interaction.followup.send(f"⚙️ Creating automation plan for: `{task}`")
                
                # Run the agent with a automation-focused prompt
                automation_prompt = f"""
                Create a step-by-step guide to automate this task: {task}
                
                Please break this down into:
                1. Clear, executable steps
                2. Tools or software needed
                3. Potential challenges and solutions
                4. Time estimates for each step
                
                Make it detailed enough that someone could follow these instructions without further guidance.
                """
                
                response = await run_agent(automation_prompt)
                
                # Format the response for Discord
                # Split into chunks if needed
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in automation command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while creating your automation plan. Please try again later.\n```{str(e)[:1500]}```")

async def setup(bot):
    await bot.add_cog(AgentCog(bot)) 