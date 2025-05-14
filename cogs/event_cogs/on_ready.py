import asyncio
import json
import os
from itertools import cycle
import discord
from discord.ext import commands

from bot_utilities.config_loader import config
from ..common import presences_disabled, current_language, presences

# Directory for storing bot data
BOT_DATA_DIR = os.path.join("bot_data", "user_tracking")
os.makedirs(BOT_DATA_DIR, exist_ok=True)

class OnReady(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.user_data_file = os.path.join(BOT_DATA_DIR, "user_data.json")

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
        
        # Collect user IDs from all guilds
        await self.collect_user_ids()
        
        if presences_disabled:
            return
        while True:
            presence = next(presences_cycle)
            presence_with_count = presence.replace("{guild_count}", str(len(self.bot.guilds)))
            delay = config['PRESENCES_CHANGE_DELAY']
            await self.bot.change_presence(activity=discord.Game(name=presence_with_count))
            await asyncio.sleep(delay)
            
    async def collect_user_ids(self):
        """Collect and store user IDs from all guilds the bot is in"""
        print("Collecting user IDs from all guilds...")
        
        # Initialize user data dictionary
        user_data = {}
        
        # Load existing data if available
        if os.path.exists(self.user_data_file):
            try:
                with open(self.user_data_file, 'r') as f:
                    user_data = json.load(f)
            except Exception as e:
                print(f"Error loading user data: {e}")
                user_data = {}
        
        # Collect users from each guild
        for guild in self.bot.guilds:
            guild_id = str(guild.id)
            guild_name = guild.name
            
            # Initialize guild data if not present
            if guild_id not in user_data:
                user_data[guild_id] = {
                    "name": guild_name,
                    "users": {}
                }
                
            # Update guild name in case it changed
            user_data[guild_id]["name"] = guild_name
            
            # Add each user in the guild
            for member in guild.members:
                if not member.bot:  # Skip bot accounts
                    user_id = str(member.id)
                    user_data[guild_id]["users"][user_id] = {
                        "name": member.display_name,
                        "joined_at": member.joined_at.isoformat() if member.joined_at else None,
                        "last_seen": None  # Will be updated when user is active
                    }
        
        # Save user data
        try:
            with open(self.user_data_file, 'w') as f:
                json.dump(user_data, f, indent=2)
            print(f"Saved data for {sum(len(g['users']) for g in user_data.values())} users across {len(user_data)} guilds")
        except Exception as e:
            print(f"Error saving user data: {e}")
    
    @commands.Cog.listener()
    async def on_member_join(self, member):
        """Track when a new user joins a guild"""
        if member.bot:
            return
            
        try:
            # Load existing data
            user_data = {}
            if os.path.exists(self.user_data_file):
                with open(self.user_data_file, 'r') as f:
                    user_data = json.load(f)
            
            guild_id = str(member.guild.id)
            guild_name = member.guild.name
            user_id = str(member.id)
            
            # Initialize guild data if not present
            if guild_id not in user_data:
                user_data[guild_id] = {
                    "name": guild_name,
                    "users": {}
                }
            
            # Add the new user
            user_data[guild_id]["users"][user_id] = {
                "name": member.display_name,
                "joined_at": member.joined_at.isoformat() if member.joined_at else None,
                "last_seen": None
            }
            
            # Save updated data
            with open(self.user_data_file, 'w') as f:
                json.dump(user_data, f, indent=2)
                
        except Exception as e:
            print(f"Error updating user data on member join: {e}")

async def setup(bot):
    await bot.add_cog(OnReady(bot))
