import discord
from discord import app_commands
from discord.ext import commands
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional

from bot_utilities.discord_integration import InteractionContext, DiscordIntegration

logger = logging.getLogger("discord_commands")

class DiscordCommandsCog(commands.Cog):
    """
    Cog for registering and handling Discord slash commands
    """
    def __init__(self, bot):
        """
        Initialize commands cog
        
        Args:
            bot: Discord bot instance
        """
        self.bot = bot
        self.integration = getattr(bot, "discord_integration", None)
        
        # Create integration if not exists
        if not self.integration:
            from bot_utilities.discord_integration import setup_discord_integration
            self.integration = setup_discord_integration(bot)
            bot.discord_integration = self.integration
        
        # Register commands
        self._register_commands()
        
        logger.info("Discord commands cog initialized")
    
    def _register_commands(self):
        """Register slash commands with Discord"""
        # Define ask command
        @app_commands.command(name="ask", description="Ask the AI chatbot a question")
        @app_commands.describe(
            query="Your question or prompt for the AI",
            reasoning="Optional reasoning method to use (sequential, rag, crag, react, graph, speculative, reflexion)"
        )
        async def ask_command(interaction, 
                            query: str, 
                            reasoning: Optional[str] = None):
            """Handle the /ask command"""
            await self.integration.handle_interaction(interaction)
        
        # Define clear command
        @app_commands.command(name="clear", description="Clear your conversation history with the bot")
        async def clear_command(interaction):
            """Handle the /clear command"""
            await self.integration.handle_interaction(interaction)
        
        # Define reasoning command
        @app_commands.command(name="reasoning", description="Get information about different reasoning methods")
        @app_commands.describe(
            method="Reasoning method to learn about"
        )
        @app_commands.choices(method=[
            app_commands.Choice(name="Sequential Thinking", value="sequential"),
            app_commands.Choice(name="RAG", value="rag"),
            app_commands.Choice(name="Contextual RAG", value="crag"),
            app_commands.Choice(name="ReAct", value="react"),
            app_commands.Choice(name="Graph-of-Thought", value="graph"),
            app_commands.Choice(name="Speculative", value="speculative"),
            app_commands.Choice(name="Reflexion", value="reflexion")
        ])
        async def reasoning_command(interaction, method: str):
            """Get information about a reasoning method"""
            ctx = InteractionContext(interaction, self.integration)
            
            # Define explanations for each reasoning method
            explanations = {
                "sequential": "**Sequential Thinking**: A step-by-step approach to solving problems by breaking them down into logical steps and reasoning through each one sequentially.",
                "rag": "**Retrieval-Augmented Generation (RAG)**: Enhances responses by retrieving relevant information from external knowledge sources before generating an answer.",
                "crag": "**Contextual RAG**: An evolution of RAG that considers conversation context and user history in addition to retrieved information.",
                "react": "**ReAct (Reasoning+Acting)**: Combines reasoning with actions in a cycle of thought, action, and observation to solve problems that require tool use or external interactions.",
                "graph": "**Graph-of-Thought**: Represents reasoning as a graph of interconnected ideas rather than a linear chain, allowing exploration of multiple reasoning paths simultaneously.",
                "speculative": "**Speculative Reasoning**: Generates multiple candidate responses and verifies them, selecting the most accurate and relevant one.",
                "reflexion": "**Reflexion**: Incorporates self-reflection capabilities, analyzing initial responses and improving them through critical assessment."
            }
            
            if method in explanations:
                embed = discord.Embed(
                    title=f"Reasoning Method: {method.upper()}",
                    description=explanations[method],
                    color=discord.Color.blue()
                )
                
                # Add when to use this method
                usage_guide = {
                    "sequential": "Best for complex problems that benefit from structured thinking.",
                    "rag": "Ideal for factual questions or information retrieval tasks.",
                    "crag": "Perfect for follow-up questions or context-dependent queries.",
                    "react": "Excellent for tasks requiring tools or external actions.",
                    "graph": "Great for problems with multiple approaches or solutions.",
                    "speculative": "Useful for queries where accuracy is critical.",
                    "reflexion": "Valuable for nuanced questions requiring deeper consideration."
                }
                
                embed.add_field(name="When to Use", value=usage_guide[method], inline=False)
                
                # Add example prompt
                example_prompts = {
                    "sequential": "Explain how a computer processor works step by step.",
                    "rag": "What are the key provisions of the Paris Climate Agreement?",
                    "crag": "Based on our previous conversation, how does this relate to what we discussed earlier?",
                    "react": "Help me find the current weather in New York City.",
                    "graph": "Compare and contrast different approaches to machine learning.",
                    "speculative": "What is the most accurate way to calculate compound interest?",
                    "reflexion": "What are the philosophical implications of artificial intelligence?"
                }
                
                embed.add_field(name="Example Prompt", value=f"`{example_prompts[method]}`", inline=False)
                
                # Add tip on how to use
                embed.add_field(
                    name="How to Use",
                    value=f"Use the `/ask` command with the `reasoning` parameter set to `{method}`:\n`/ask query:Your question reasoning:{method}`",
                    inline=False
                )
                
                await ctx.respond("", embeds=[embed])
            else:
                await ctx.respond(f"Unknown reasoning method: {method}")
        
        # Define help command
        @app_commands.command(name="help", description="Get help with using the bot")
        async def help_command(interaction):
            """Handle the /help command"""
            ctx = InteractionContext(interaction, self.integration)
            
            embed = discord.Embed(
                title="AI Chatbot Help",
                description="This bot uses advanced reasoning methods to provide helpful responses to your questions.",
                color=discord.Color.blue()
            )
            
            # Add command section
            embed.add_field(
                name="Available Commands",
                value=(
                    "• `/ask query:Your question reasoning:optional_method` - Ask the AI a question\n"
                    "• `/clear` - Clear your conversation history\n"
                    "• `/reasoning method:name` - Learn about a specific reasoning method\n"
                    "• `/help` - Show this help message"
                ),
                inline=False
            )
            
            # Add reasoning methods section
            embed.add_field(
                name="Reasoning Methods",
                value=(
                    "• `sequential` - Step by step reasoning\n"
                    "• `rag` - Retrieval-Augmented Generation\n"
                    "• `crag` - Contextual RAG\n"
                    "• `react` - Reasoning and Acting\n"
                    "• `graph` - Graph-of-Thought\n"
                    "• `speculative` - Speculative Reasoning\n"
                    "• `reflexion` - Self-reflective reasoning"
                ),
                inline=False
            )
            
            # Add examples section
            embed.add_field(
                name="Examples",
                value=(
                    "• `/ask query:How does photosynthesis work?`\n"
                    "• `/ask query:Summarize the history of quantum physics reasoning:sequential`\n"
                    "• `/reasoning method:rag`"
                ),
                inline=False
            )
            
            # Add footer
            embed.set_footer(text="For more detailed help on reasoning methods, use /reasoning")
            
            await ctx.respond("", embeds=[embed])
        
        # Add the commands to the bot
        self.bot.tree.add_command(ask_command)
        self.bot.tree.add_command(clear_command)
        self.bot.tree.add_command(reasoning_command)
        self.bot.tree.add_command(help_command)
        
        # Register custom command handlers with the integration
        self.integration.register_command("reasoning", self._handle_reasoning_command)
        self.integration.register_command("help", self._handle_help_command)
    
    async def _handle_reasoning_command(self, ctx: InteractionContext):
        """Custom handler for reasoning command"""
        method = ctx.data.get("options", {}).get("method")
        
        # Get metrics from reasoning router if available
        metrics = {}
        if hasattr(self.bot, "reasoning_router"):
            try:
                metrics = await self.bot.reasoning_router.get_metrics()
            except Exception as e:
                logger.error(f"Error getting reasoning metrics: {e}")
        
        # Add usage metrics to response if available
        if method and metrics and "reasoning_type_selected" in metrics:
            usage_count = metrics["reasoning_type_selected"].get(method, 0)
            avg_time = metrics["avg_response_time"].get(method, 0)
            
            embed = discord.Embed(
                title=f"{method.upper()} Usage Statistics",
                color=discord.Color.gold()
            )
            
            embed.add_field(name="Times Used", value=str(usage_count), inline=True)
            embed.add_field(name="Average Response Time", value=f"{avg_time:.2f}s", inline=True)
            
            # Send as a follow-up message
            await ctx.interaction.followup.send(embeds=[embed])
    
    async def _handle_help_command(self, ctx: InteractionContext):
        """Custom handler for help command"""
        # Add any custom help logic here
        pass
    
    @commands.Cog.listener()
    async def on_interaction(self, interaction):
        """
        Handle Discord interactions
        
        Args:
            interaction: Discord interaction
        """
        # Use the integration to handle interactions
        if self.integration:
            await self.integration.handle_interaction(interaction)

async def setup(bot):
    """
    Set up the commands cog
    
    Args:
        bot: Discord bot instance
    """
    await bot.add_cog(DiscordCommandsCog(bot)) 