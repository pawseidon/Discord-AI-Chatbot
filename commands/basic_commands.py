import discord
from discord.ext import commands
from discord import app_commands
import logging
import time
from typing import Dict, Any, List, Optional
import platform
import sys
import psutil

logger = logging.getLogger("basic_commands")

class BasicCommandsCog(commands.Cog):
    """
    Basic commands for the Discord AI Chatbot
    """
    
    def __init__(self, bot: commands.Bot):
        """
        Initialize the basic commands cog
        
        Args:
            bot: Discord bot instance
        """
        self.bot = bot
        self.start_time = time.time()
        
        # Injected dependencies
        self.discord_integration = None
        self.cache_provider = None
        self.reasoning_router = None
        
        # System info
        self.python_version = platform.python_version()
        self.discord_py_version = discord.__version__
    
    def set_discord_integration(self, discord_integration):
        """Set the Discord integration service"""
        self.discord_integration = discord_integration
    
    def set_cache_provider(self, cache_provider):
        """Set the cache provider service"""
        self.cache_provider = cache_provider
    
    def set_reasoning_router(self, reasoning_router):
        """Set the reasoning router service"""
        self.reasoning_router = reasoning_router
    
    @commands.command(name="ping")
    async def ping(self, ctx: commands.Context):
        """Check the bot's latency"""
        start_time = time.time()
        message = await ctx.send("Pinging...")
        end_time = time.time()
        
        discord_latency = round(self.bot.latency * 1000)
        api_latency = round((end_time - start_time) * 1000)
        
        embed = discord.Embed(
            title="🏓 Pong!",
            color=discord.Color.green()
        )
        embed.add_field(name="Discord WebSocket", value=f"{discord_latency}ms", inline=True)
        embed.add_field(name="Discord API", value=f"{api_latency}ms", inline=True)
        
        await message.edit(content=None, embed=embed)
    
    @commands.command(name="uptime")
    async def uptime(self, ctx: commands.Context):
        """Check how long the bot has been running"""
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(int(uptime_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        days, hours = divmod(hours, 24)
        
        uptime_str = f"{days}d {hours}h {minutes}m {seconds}s"
        
        embed = discord.Embed(
            title="⏱️ Bot Uptime",
            description=f"Bot has been online for: **{uptime_str}**",
            color=discord.Color.blue()
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name="info")
    async def info(self, ctx: commands.Context):
        """Display information about the bot"""
        # System info
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Bot info
        guild_count = len(self.bot.guilds)
        user_count = sum(g.member_count for g in self.bot.guilds)
        
        # Uptime
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(int(uptime_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        days, hours = divmod(hours, 24)
        uptime_str = f"{days}d {hours}h {minutes}m {seconds}s"
        
        embed = discord.Embed(
            title="ℹ️ Bot Information",
            color=discord.Color.blue()
        )
        
        # General info
        embed.add_field(name="Bot Name", value=self.bot.user.name, inline=True)
        embed.add_field(name="Servers", value=str(guild_count), inline=True)
        embed.add_field(name="Users", value=str(user_count), inline=True)
        
        # System info
        embed.add_field(name="Python Version", value=self.python_version, inline=True)
        embed.add_field(name="Discord.py Version", value=self.discord_py_version, inline=True)
        embed.add_field(name="Memory Usage", value=f"{memory_usage:.2f} MB", inline=True)
        
        # Runtime info
        embed.add_field(name="Uptime", value=uptime_str, inline=True)
        embed.add_field(name="Ping", value=f"{round(self.bot.latency * 1000)}ms", inline=True)
        
        # Add footer
        embed.set_footer(text="Discord AI Chatbot | Developed with ❤️")
        
        await ctx.send(embed=embed)
    
    @commands.command(name="help")
    async def help_command(self, ctx: commands.Context, command: Optional[str] = None):
        """Display help information for commands"""
        prefix = self.bot.command_prefix
        if isinstance(prefix, list):
            prefix = prefix[0]
        
        if command:
            # Help for specific command
            cmd = self.bot.get_command(command)
            if cmd:
                embed = discord.Embed(
                    title=f"Help: {prefix}{cmd.name}",
                    description=cmd.help or "No description provided",
                    color=discord.Color.blue()
                )
                
                # Add usage
                usage = f"{prefix}{cmd.name}"
                if cmd.signature:
                    usage = f"{prefix}{cmd.name} {cmd.signature}"
                embed.add_field(name="Usage", value=f"`{usage}`", inline=False)
                
                # Add aliases if any
                if cmd.aliases:
                    aliases = ", ".join([f"{prefix}{alias}" for alias in cmd.aliases])
                    embed.add_field(name="Aliases", value=aliases, inline=False)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"Command `{command}` not found.")
        else:
            # General help
            embed = discord.Embed(
                title="Bot Commands",
                description=f"Here are the available commands. Use `{prefix}help <command>` for detailed help on a command.",
                color=discord.Color.blue()
            )
            
            # Group commands by cog
            cog_commands = {}
            for cmd in sorted(self.bot.commands, key=lambda x: x.name):
                cog_name = cmd.cog_name or "No Category"
                if cog_name not in cog_commands:
                    cog_commands[cog_name] = []
                cog_commands[cog_name].append(cmd)
            
            # Add fields for each cog
            for cog_name, commands_list in sorted(cog_commands.items()):
                if cog_name == "No Category":
                    continue  # Skip uncategorized commands
                
                commands_text = ", ".join([f"`{prefix}{cmd.name}`" for cmd in commands_list])
                embed.add_field(name=cog_name, value=commands_text, inline=False)
            
            # Add uncategorized commands if any
            if "No Category" in cog_commands:
                commands_text = ", ".join([f"`{prefix}{cmd.name}`" for cmd in cog_commands["No Category"]])
                embed.add_field(name="Other Commands", value=commands_text, inline=False)
            
            # Add application commands if any
            if self.bot.tree.get_commands():
                app_commands_text = ", ".join([f"`/{cmd.name}`" for cmd in self.bot.tree.get_commands()])
                embed.add_field(name="Slash Commands", value=app_commands_text, inline=False)
            
            await ctx.send(embed=embed)
    
    @app_commands.command(name="ping", description="Check the bot's latency")
    async def slash_ping(self, interaction: discord.Interaction):
        """Slash command version of ping"""
        start_time = time.time()
        discord_latency = round(self.bot.latency * 1000)
        
        await interaction.response.defer(thinking=True)
        
        end_time = time.time()
        api_latency = round((end_time - start_time) * 1000)
        
        embed = discord.Embed(
            title="🏓 Pong!",
            color=discord.Color.green()
        )
        embed.add_field(name="Discord WebSocket", value=f"{discord_latency}ms", inline=True)
        embed.add_field(name="Discord API", value=f"{api_latency}ms", inline=True)
        
        await interaction.followup.send(embed=embed)
    
    @app_commands.command(name="info", description="Display information about the bot")
    async def slash_info(self, interaction: discord.Interaction):
        """Slash command version of info"""
        # System info
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Bot info
        guild_count = len(self.bot.guilds)
        user_count = sum(g.member_count for g in self.bot.guilds)
        
        # Uptime
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(int(uptime_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        days, hours = divmod(hours, 24)
        uptime_str = f"{days}d {hours}h {minutes}m {seconds}s"
        
        embed = discord.Embed(
            title="ℹ️ Bot Information",
            color=discord.Color.blue()
        )
        
        # General info
        embed.add_field(name="Bot Name", value=self.bot.user.name, inline=True)
        embed.add_field(name="Servers", value=str(guild_count), inline=True)
        embed.add_field(name="Users", value=str(user_count), inline=True)
        
        # System info
        embed.add_field(name="Python Version", value=self.python_version, inline=True)
        embed.add_field(name="Discord.py Version", value=self.discord_py_version, inline=True)
        embed.add_field(name="Memory Usage", value=f"{memory_usage:.2f} MB", inline=True)
        
        # Runtime info
        embed.add_field(name="Uptime", value=uptime_str, inline=True)
        embed.add_field(name="Ping", value=f"{round(self.bot.latency * 1000)}ms", inline=True)
        
        # Add footer
        embed.set_footer(text="Discord AI Chatbot | Developed with ❤️")
        
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(
        name="reasoning", 
        description="Get information about available reasoning methods"
    )
    async def slash_reasoning(self, interaction: discord.Interaction):
        """Display information about available reasoning methods"""
        if not self.reasoning_router:
            await interaction.response.send_message(
                "Reasoning router is not available.",
                ephemeral=True
            )
            return
        
        methods = self.reasoning_router.get_supported_methods()
        
        embed = discord.Embed(
            title="🧠 Available Reasoning Methods",
            description="The bot can use different reasoning methods to answer your questions.",
            color=discord.Color.blue()
        )
        
        for method in methods:
            embed.add_field(
                name=f"{method['emoji']} {method['name'].title()}",
                value=method['description'],
                inline=False
            )
        
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(
        name="ask",
        description="Ask a question using a specific reasoning method"
    )
    @app_commands.describe(
        question="Your question",
        reasoning_method="Reasoning method to use (optional)"
    )
    async def slash_ask(
        self, 
        interaction: discord.Interaction, 
        question: str,
        reasoning_method: Optional[str] = None
    ):
        """Ask a question with an optional specific reasoning method"""
        if not self.discord_integration:
            await interaction.response.send_message(
                "AI processing is not available right now.",
                ephemeral=True
            )
            return
        
        await interaction.response.defer(thinking=True)
        
        # Process with discord integration
        response_data = await self.discord_integration.process_interaction(
            interaction,
            query=question,
            reasoning_method=reasoning_method
        )
        
        if not response_data:
            await interaction.followup.send(
                "I couldn't generate a response to your question."
            )
            return
        
        # Format response for Discord
        formatted_response = self.discord_integration.format_response_for_discord(response_data)
        
        await interaction.followup.send(
            content=formatted_response["content"],
            embed=formatted_response.get("embed")
        )
    
    @commands.command(name="metrics")
    @commands.is_owner()
    async def metrics(self, ctx: commands.Context):
        """Display metrics for the bot (owner only)"""
        if not self.discord_integration:
            await ctx.send("Discord integration is not available.")
            return
        
        metrics = await self.discord_integration.get_metrics()
        
        embed = discord.Embed(
            title="📊 Bot Metrics",
            color=discord.Color.gold()
        )
        
        # Discord integration metrics
        embed.add_field(
            name="Discord Integration",
            value=f"Total Messages: {metrics.get('total_messages', 0)}\n"
                  f"Total Responses: {metrics.get('total_responses', 0)}\n"
                  f"Average Response Time: {metrics.get('avg_response_time', 0):.2f}s\n"
                  f"Cached Responses: {metrics.get('cached_responses', 0)}\n"
                  f"Errors: {metrics.get('errors', 0)}",
            inline=False
        )
        
        # Reasoning metrics
        if "reasoning" in metrics:
            reasoning_metrics = metrics["reasoning"]
            reasoning_text = ""
            for method, method_metrics in reasoning_metrics.items():
                if method_metrics.get("calls", 0) > 0:
                    reasoning_text += f"{method.title()}: {method_metrics.get('calls', 0)} calls, "
                    reasoning_text += f"{method_metrics.get('avg_time', 0):.2f}s avg time\n"
            
            if reasoning_text:
                embed.add_field(
                    name="Reasoning Methods",
                    value=reasoning_text,
                    inline=False
                )
        
        # Cache metrics
        if "cache" in metrics and metrics["cache"].get("total_entries", 0) > 0:
            cache_metrics = metrics["cache"]
            cache_text = f"Total Entries: {cache_metrics.get('total_entries', 0)}\n"
            cache_text += f"Hits: {cache_metrics.get('hits', 0)}\n"
            cache_text += f"Misses: {cache_metrics.get('misses', 0)}\n"
            
            if "cache_counts" in cache_metrics:
                cache_text += "\nCache Types:\n"
                for cache_type, count in cache_metrics["cache_counts"].items():
                    if count > 0:
                        cache_text += f"{cache_type}: {count} entries\n"
            
            embed.add_field(
                name="Cache",
                value=cache_text,
                inline=False
            )
        
        await ctx.send(embed=embed)

async def setup(bot: commands.Bot):
    """Set up the basic commands cog"""
    await bot.add_cog(BasicCommandsCog(bot)) 