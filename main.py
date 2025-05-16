import os
from typing import Any

import discord
from discord.ext import commands
from dotenv import load_dotenv
import asyncio
import logging

from cogs import EVENT_HANDLERS
from cogs.commands_cogs import get_all_cogs
from bot_utilities.config_loader import config
from bot_utilities.ai_utils import get_ai_provider
from bot_utilities.services.agent_service import agent_service
from bot_utilities.services.workflow_service import workflow_service
from bot_utilities.services.memory_service import memory_service
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/bot.log'),
    ]
)
logger = logging.getLogger('discord_bot')

class AIBot(commands.AutoShardedBot):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if config['AUTO_SHARDING']:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(shard_count=1, *args, **kwargs)

    async def setup_hook(self) -> None:
        """
        Initialize services and load extensions when the bot starts up
        """
        logger.info("Starting bot initialization...")
        
        # Create necessary directories
        os.makedirs("bot_data/cache", exist_ok=True)
        os.makedirs("bot_data/memory", exist_ok=True)
        os.makedirs("bot_data/backups", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize AI services first
        try:
            logger.info("Initializing AI services...")
            llm_provider = await get_ai_provider()
            
            # Initialize critical services
            await agent_service.initialize(llm_provider)
            await workflow_service.initialize(llm_provider)
            await symbolic_reasoning_service.ensure_initialized()
            await memory_service.load_from_disk()
            
            # Register services after initialization to avoid circular imports
            workflow_service.register_services(agent_service=agent_service, memory_service=memory_service)
            
            logger.info("AI services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI services: {e}")
            logger.warning("Bot will continue startup but some AI features may not work correctly")
        
        # Load event handlers
        for cog in EVENT_HANDLERS:
            cog_name = cog.split('.')[-1]
            try:
                await self.load_extension(f"{cog}")
                logger.info(f"Loaded Event Handler {cog_name}")
            except Exception as e:
                logger.error(f"Failed to load Event Handler {cog_name}: {e}")
        
        # Load command cogs dynamically
        cog_modules = await get_all_cogs()
        for cog_name in cog_modules:
            try:
                await self.load_extension(f"cogs.commands_cogs.{cog_name}")
                logger.info(f"Loaded Command {cog_name}")
            except Exception as e:
                logger.error(f"Failed to load Command {cog_name}: {e}")
        
        logger.info("Syncing commands...")
        await self.tree.sync()
        logger.info(f"Loaded {len(self.commands)} commands")
        logger.info("Bot initialization complete")

    async def on_ready(self):
        """Called when the bot is ready"""
        logger.info(f'Logged in as {self.user.name} ({self.user.id})')
        logger.info(f'Connected to {len(self.guilds)} guilds')
        
        # Set custom status
        activity = discord.Activity(
            type=discord.ActivityType.listening, 
            name="your questions | mention me to chat"
        )
        await self.change_presence(activity=activity)

bot = AIBot(command_prefix=[], intents=discord.Intents.all(), help_command=None)

TOKEN = os.getenv('DISCORD_TOKEN')

if TOKEN is None:
    logger.error("Discord token not found in environment variables")
    print("\033[31mLooks like you haven't properly set up a Discord token environment variable in the `.env` file. (Secrets on replit)\033[0m")
    print("\033[33mNote: If you don't have a Discord token environment variable, you will have to input it every time. \033[0m")
    TOKEN = input("Please enter your Discord token: ")

def main():
    try:
        logger.info("Starting bot...")
        bot.run(TOKEN, reconnect=True)
    except discord.LoginFailure:
        logger.critical("Invalid Discord token")
        print("\033[31mInvalid Discord token. Please check your token and try again.\033[0m")
    except Exception as e:
        logger.critical(f"Error starting bot: {e}")
        print(f"\033[31mError starting bot: {e}\033[0m")

if __name__ == "__main__":
    main()
