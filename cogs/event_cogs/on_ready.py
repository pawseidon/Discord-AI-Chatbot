import asyncio
from itertools import cycle
import discord
from discord.ext import commands

from bot_utilities.config_loader import config
from bot_utilities.context_manager import start_context_manager_tasks
from bot_utilities.router_init import get_router
from ..common import presences_disabled, current_language, presences

class OnReady(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.context_manager_initialized = False
        self.router_initialized = False

    @commands.Cog.listener()
    async def on_ready(self):
        presences_cycle = cycle(presences + [current_language['help_footer']])
        print(f"{self.bot.user} aka {self.bot.user.name} has connected to Discord!")
        invite_link = discord.utils.oauth_url(
            self.bot.user.id,
            permissions=discord.Permissions(),
            scopes=("bot", "applications.commands")
        )
        print(f"Invite link: {invite_link}")
        
        # Initialize context manager if not already done
        if not self.context_manager_initialized:
            print("Initializing context manager...")
            asyncio.create_task(self.initialize_context_manager())
            self.context_manager_initialized = True
            
        # Initialize router if not already done
        if not self.router_initialized:
            print("Initializing router...")
            asyncio.create_task(self.initialize_router())
            self.router_initialized = True
        
        if presences_disabled:
            return
        while True:
            presence = next(presences_cycle)
            presence_with_count = presence.replace("{guild_count}", str(len(self.bot.guilds)))
            delay = config['PRESENCES_CHANGE_DELAY']
            await self.bot.change_presence(activity=discord.Game(name=presence_with_count))
            await asyncio.sleep(delay)
    
    async def initialize_context_manager(self):
        """Initialize and start context manager background tasks"""
        try:
            await start_context_manager_tasks()
            print("Context manager successfully initialized.")
        except Exception as e:
            print(f"Error initializing context manager: {e}")
            
    async def initialize_router(self):
        """Initialize the router for message processing"""
        try:
            router = await get_router()
            if router:
                print("Router successfully initialized.")
            else:
                print("Failed to initialize router.")
        except Exception as e:
            print(f"Error initializing router: {e}")

async def setup(bot):
    await bot.add_cog(OnReady(bot))
