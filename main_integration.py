#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main integration module for Discord AI Chatbot

This module demonstrates how to properly integrate all components
of the new architecture: caching, reasoning, and Discord integration.
"""

import logging
import sys
import os
import asyncio
import discord
from discord.ext import commands

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('discord_bot.log')
    ]
)
logger = logging.getLogger("main_integration")

# Import components from new architecture
from features.caching import (
    create_cache_provider,
    create_context_aware_cache,
    create_semantic_cache,
    create_cache_integration,
    create_advanced_cache_system
)

from features.reasoning import (
    ReasoningRouter,
    ReasoningMethod,
    IntegratedReasoning,
    create_integrated_reasoning
)

from core.discord_integration import (
    create_discord_integration
)

# Optional components (if available)
try:
    from features.safety.hallucination_handler import create_hallucination_handler
    hallucination_available = True
except ImportError:
    hallucination_available = False
    logger.warning("Hallucination handler not available")

class DiscordAIBot:
    """
    Discord AI Chatbot with integrated reasoning and caching
    """
    
    def __init__(self, token: str, config: dict = None):
        """
        Initialize the Discord AI Bot
        
        Args:
            token: Discord bot token
            config: Optional configuration
        """
        self.token = token
        self.config = config or {}
        
        # Set up Discord client
        intents = discord.Intents.default()
        intents.message_content = True  # Required for processing message content
        intents.members = True  # Required for user info
        
        self.bot = commands.Bot(
            command_prefix=self.config.get('command_prefix', '!'),
            intents=intents,
            help_command=None
        )
        
        # Components
        self.base_cache = None
        self.context_cache = None
        self.semantic_cache = None
        self.cache_proxy = None
        self.reasoning_router = None
        self.integrated_reasoning = None
        self.hallucination_handler = None
        self.discord_integration = None
        
        # Setup status
        self.is_setup = False
    
    async def setup(self):
        """Set up the bot components"""
        logger.info("Setting up Discord AI Bot")
        
        # Create the cache system
        (
            self.base_cache,
            self.context_cache, 
            self.semantic_cache,
            self.cache_proxy
        ) = create_advanced_cache_system(self.config.get('cache_config'))
        
        logger.info("Cache system initialized")
        
        # Create hallucination handler if available
        if hallucination_available:
            self.hallucination_handler = create_hallucination_handler(
                self.config.get('hallucination_config')
            )
            logger.info("Hallucination handler initialized")
        
        # Create reasoning router
        self.reasoning_router = ReasoningRouter()
        self.reasoning_router.register_default_methods()
        logger.info("Reasoning router initialized with default methods")
        
        # Create integrated reasoning
        self.integrated_reasoning = create_integrated_reasoning(
            cache_proxy=self.cache_proxy,
            reasoning_router=self.reasoning_router,
            hallucination_handler=self.hallucination_handler,
            context_cache=self.context_cache,
            semantic_cache=self.semantic_cache
        )
        logger.info("Integrated reasoning system initialized")
        
        # Create Discord integration
        self.discord_integration = create_discord_integration(
            bot=self.bot,
            base_cache=self.base_cache,
            context_cache=self.context_cache,
            semantic_cache=self.semantic_cache,
            reasoning_router=self.reasoning_router,
            reasoning_integration=self.integrated_reasoning,
            hallucination_handler=self.hallucination_handler
        )
        logger.info("Discord integration initialized")
        
        # Register message handler
        @self.bot.event
        async def on_message(message):
            # Ignore messages from self
            if message.author == self.bot.user:
                return
            
            # Process with bot.process_commands for command handling
            await self.bot.process_commands(message)
            
            # For regular messages, process through our integration
            if not message.content.startswith(self.bot.command_prefix):
                try:
                    response = await self.discord_integration.process_message(message)
                    if response:
                        await message.channel.send(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await message.channel.send("I encountered an error while processing your message.")
        
        # Register default commands
        self._register_commands()
        
        self.is_setup = True
        logger.info("Discord AI Bot setup complete")
    
    def _register_commands(self):
        """Register default commands"""
        
        @self.bot.command(name="ask")
        async def ask_command(ctx, *, query: str):
            """Process a query with the specified reasoning method"""
            async with ctx.typing():
                response = await self.discord_integration.process_message(
                    ctx.message, 
                    reasoning_method=None  # Let the system decide
                )
                if response:
                    await ctx.send(response)
                else:
                    await ctx.send("I couldn't generate a response.")
        
        @self.bot.command(name="help")
        async def help_command(ctx):
            """Show help information"""
            help_embed = discord.Embed(
                title="Discord AI Bot Help",
                description="Here are the available commands:",
                color=discord.Color.blue()
            )
            
            help_embed.add_field(
                name=f"{self.bot.command_prefix}ask [query]",
                value="Ask a question or request information",
                inline=False
            )
            
            help_embed.add_field(
                name=f"{self.bot.command_prefix}clear",
                value="Clear your conversation history",
                inline=False
            )
            
            help_embed.add_field(
                name=f"{self.bot.command_prefix}help",
                value="Show this help message",
                inline=False
            )
            
            # Add reasoning methods field
            if self.reasoning_router:
                method_names = [m.value for m in ReasoningMethod]
                methods_str = ", ".join(method_names)
                
                help_embed.add_field(
                    name="Available Reasoning Methods",
                    value=f"The bot can use various reasoning methods: {methods_str}",
                    inline=False
                )
            
            await ctx.send(embed=help_embed)
        
        @self.bot.command(name="clear")
        async def clear_command(ctx):
            """Clear conversation history"""
            if self.discord_integration:
                count = await self.discord_integration.clear_user_context(
                    user_id=str(ctx.author.id),
                    guild_id=str(ctx.guild.id) if ctx.guild else None,
                    channel_id=str(ctx.channel.id)
                )
                
                await ctx.send(f"Your conversation history has been cleared! ({count} entries removed)")
            else:
                await ctx.send("Cache system is not available.")
    
    async def start(self):
        """Start the Discord bot"""
        if not self.is_setup:
            await self.setup()
        
        logger.info("Starting Discord AI Bot")
        await self.bot.start(self.token)
    
    async def close(self):
        """Close the Discord bot and cleanup resources"""
        logger.info("Closing Discord AI Bot")
        
        # Clean up cache resources
        if self.base_cache:
            await self.base_cache.cleanup()
        
        # Clean up Discord integration
        if self.discord_integration:
            await self.discord_integration.cleanup()
        
        # Close bot
        await self.bot.close()

async def main():
    """Main entry point"""
    # Load configuration and token
    import json
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning(f"Config file not found or invalid: {config_path}")
        config = {}
    
    # Get token from config or environment
    token = config.get("token") or os.environ.get("DISCORD_BOT_TOKEN")
    
    if not token:
        logger.error("Discord bot token not found! Set it in config.json or DISCORD_BOT_TOKEN environment variable.")
        return
    
    # Create and start bot
    bot = DiscordAIBot(token, config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await bot.close()

if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutting down")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc()) 