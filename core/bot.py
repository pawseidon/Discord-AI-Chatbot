import discord
from discord.ext import commands
import logging
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional, Union, Callable
import time

# Import core components
from core.discord_integration import create_discord_integration
from core.ai_provider import create_ai_provider
from caching import create_advanced_cache_system
from utils.hallucination_handler import create_hallucination_handler
from features.reasoning.reasoning_router import create_reasoning_router
from features.reasoning import create_reasoning_integration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)

logger = logging.getLogger("bot")

class DiscordAIChatBot:
    """
    Main Discord AI Chatbot class that coordinates all components
    """
    
    def __init__(self, token: str, prefix: str = "!"):
        """
        Initialize the Discord AI Chatbot
        
        Args:
            token: Discord bot token
            prefix: Command prefix
        """
        self.token = token
        self.prefix = prefix
        
        # Initialize bot with intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        self.bot = commands.Bot(command_prefix=prefix, intents=intents)
        
        # Core components
        self.base_cache = None
        self.context_cache = None
        self.semantic_cache = None
        self.ai_provider = None
        self.hallucination_handler = None
        self.reasoning_router = None
        self.reasoning_integration = None
        self.discord_integration = None
        
        # Startup time
        self.startup_time = time.time()
        
        # Register event handlers
        self._register_events()
        
        # Storage for loaded cogs
        self.loaded_cogs = []
    
    def _register_events(self):
        """Register bot event handlers"""
        @self.bot.event
        async def on_ready():
            """Called when the bot is ready"""
            logger.info(f"Logged in as {self.bot.user.name} (ID: {self.bot.user.id})")
            logger.info(f"Connected to {len(self.bot.guilds)} guilds")
            logger.info(f"Bot is ready in {time.time() - self.startup_time:.2f}s")
            
            # Set bot status
            await self.bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.listening,
                    name=f"{self.prefix}help | AI Chatbot"
                )
            )
        
        @self.bot.event
        async def on_message(message):
            """Called when a message is received"""
            # Process commands first
            await self.bot.process_commands(message)
            
            # Skip if message is from the bot itself
            if message.author == self.bot.user:
                return
            
            # Check if bot is mentioned or in DM channel
            is_mentioned = self.bot.user in message.mentions
            is_dm = isinstance(message.channel, discord.DMChannel)
            
            # Only respond if mentioned or in DM
            if is_mentioned or is_dm:
                # Remove mention from content
                content = message.content
                if is_mentioned:
                    content = content.replace(f"<@{self.bot.user.id}>", "").strip()
                
                if not content:
                    # Just a mention with no content
                    await message.channel.send("How can I help you?")
                    return
                
                # Process with discord integration
                if self.discord_integration:
                    response = await self.discord_integration.process_message(message)
                    if response:
                        await message.channel.send(response)
                else:
                    # Fallback if integration not set up
                    await message.channel.send("AI processing is not available right now.")
    
    async def init_components(self, config: Dict[str, Any] = None):
        """
        Initialize bot components
        
        Args:
            config: Configuration dictionary for components
        """
        config = config or {}
        
        # Initialize advanced caching system
        logger.info("Initializing advanced caching system")
        cache_config = {
            "cache_type": config.get("cache_type", "memory"),
            "cache_config": config.get("cache_config", {}),
            "context_cache_config": config.get("context_cache_config", {}),
            "semantic_cache_config": config.get("semantic_cache_config", {})
        }
        
        caches = create_advanced_cache_system(cache_config)
        self.base_cache = caches[0]
        self.context_cache = caches[1]
        self.semantic_cache = caches[2]
        
        # Initialize AI provider
        ai_type = config.get("ai_type", "base")
        ai_config = config.get("ai_config", {})
        logger.info(f"Initializing AI provider: {ai_type}")
        self.ai_provider = create_ai_provider(
            provider_type=ai_type,
            **ai_config
        )
        
        # Initialize hallucination handler
        logger.info("Initializing hallucination handler")
        self.hallucination_handler = create_hallucination_handler(
            response_cache=self.base_cache,
            llm_provider=self.ai_provider,
            **config.get("hallucination_config", {})
        )
        
        # Initialize reasoning router
        logger.info("Initializing reasoning router")
        self.reasoning_router = create_reasoning_router(
            ai_provider=self.ai_provider,
            response_cache=self.base_cache,
            hallucination_handler=self.hallucination_handler,
            **config.get("reasoning_config", {})
        )
        
        # Initialize integrated reasoning system
        logger.info("Initializing integrated reasoning system")
        self.reasoning_integration = create_reasoning_integration(
            reasoning_router=self.reasoning_router,
            ai_provider=self.ai_provider,
            context_cache=self.context_cache
        )
        
        # Initialize Discord integration
        logger.info("Initializing Discord integration")
        self.discord_integration = create_discord_integration(
            bot=self.bot,
            base_cache=self.base_cache,
            context_cache=self.context_cache,
            semantic_cache=self.semantic_cache,
            reasoning_router=self.reasoning_router,
            reasoning_integration=self.reasoning_integration,
            hallucination_handler=self.hallucination_handler
        )
        
        logger.info("All components initialized")
    
    async def load_cogs(self, cogs_path: str = "cogs"):
        """
        Load all cogs from the given path
        
        Args:
            cogs_path: Path to cogs directory
        """
        logger.info(f"Loading cogs from {cogs_path}")
        
        # Get all Python files in cogs directory and subdirectories
        for root, dirs, files in os.walk(cogs_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("_"):
                    # Convert path to module path
                    path = os.path.join(root, file)
                    module_path = path.replace("/", ".").replace("\\", ".")[:-3]  # Remove .py
                    
                    try:
                        logger.info(f"Loading cog: {module_path}")
                        await self.bot.load_extension(module_path)
                        self.loaded_cogs.append(module_path)
                    except Exception as e:
                        logger.error(f"Failed to load cog {module_path}: {e}")
        
        logger.info(f"Loaded {len(self.loaded_cogs)} cogs")
    
    async def inject_dependencies(self):
        """Inject dependencies into cogs"""
        logger.info("Injecting dependencies into cogs")
        
        # Create dependency map
        dependencies = {
            "bot": self.bot,
            "base_cache": self.base_cache,
            "context_cache": self.context_cache,
            "semantic_cache": self.semantic_cache,
            "ai_provider": self.ai_provider,
            "hallucination_handler": self.hallucination_handler,
            "reasoning_router": self.reasoning_router,
            "reasoning_integration": self.reasoning_integration,
            "discord_integration": self.discord_integration
        }
        
        # Inject dependencies into each cog
        for cog_name in self.loaded_cogs:
            cog = self.bot.get_cog(cog_name)
            if cog and hasattr(cog, "inject_dependencies"):
                logger.info(f"Injecting dependencies into {cog_name}")
                try:
                    await cog.inject_dependencies(**dependencies)
                except Exception as e:
                    logger.error(f"Error injecting dependencies into {cog_name}: {e}")
                    continue
        
        logger.info("Dependency injection complete")
    
    async def start(self):
        """Start the bot"""
        logger.info("Starting bot...")
        try:
            await self.bot.start(self.token)
        except discord.errors.LoginFailure:
            logger.error("Invalid bot token. Please check your token and try again.")
            raise
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise
    
    async def close(self):
        """Close the bot and clean up resources"""
        logger.info("Shutting down bot...")
        
        # Clean up Discord integration
        if self.discord_integration:
            await self.discord_integration.cleanup()
        
        # Close the bot
        await self.bot.close()

def create_bot(token: str, prefix: str = "!") -> DiscordAIChatBot:
    """
    Factory function to create a Discord AI Chatbot
    
    Args:
        token: Discord bot token
        prefix: Command prefix
        
    Returns:
        Configured DiscordAIChatBot instance
    """
    return DiscordAIChatBot(token=token, prefix=prefix)

async def initialize_bot(bot: DiscordAIChatBot, config: Dict[str, Any] = None):
    """
    Initialize all bot components and cogs
    
    Args:
        bot: Discord AI Chatbot instance
        config: Configuration dictionary
    """
    # Initialize components
    await bot.init_components(config)
    
    # Load cogs
    await bot.load_cogs(config.get("cogs_path", "cogs"))
    
    # Inject dependencies
    await bot.inject_dependencies() 