import discord
from discord.ext import commands
from discord import app_commands
import traceback
import asyncio
import io
from datetime import datetime, timedelta

from bot_utilities.monitoring import AgentMonitor

class StatsCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.monitor = AgentMonitor()
        
    @app_commands.command(name="stats", description="Get bot usage statistics")
    @app_commands.describe(period="Time period for stats (days)")
    async def stats_command(self, interaction: discord.Interaction, period: int = 7):
        """
        Get bot usage statistics
        
        Parameters:
        period (int): Time period for stats in days (default: 7)
        """
        # Limit the period to a reasonable range
        if period < 1:
            period = 1
        elif period > 30:
            period = 30
        
        # Defer the response since generating stats might take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Generate a performance report
            report = await self.monitor.generate_performance_report(days=period)
            
            # Generate a usage chart
            chart_buffer = await self.monitor.generate_usage_chart(days=period)
            
            # Create a file from the chart buffer
            chart_file = discord.File(chart_buffer, filename="usage_chart.png")
            
            # Create an embed for the report
            embed = discord.Embed(
                title=f"Bot Statistics (Last {period} Days)",
                description="Here are the usage statistics for the bot.",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            # Add the chart as an image
            embed.set_image(url="attachment://usage_chart.png")
            
            # Send the response
            await interaction.followup.send(content=report, file=chart_file, embed=embed)
            
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in stats command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while generating statistics. Please try again later.\n```{str(e)[:1500]}```")
    
    @app_commands.command(name="user-stats", description="Get statistics for a specific user")
    @app_commands.describe(user="The user to get statistics for (default: you)")
    async def user_stats_command(self, interaction: discord.Interaction, user: discord.User = None):
        """
        Get statistics for a specific user
        
        Parameters:
        user (discord.User): The user to get statistics for (default: the command user)
        """
        # Use the interaction user if no user is specified
        if user is None:
            user = interaction.user
        
        # Defer the response
        await interaction.response.defer(thinking=True)
        
        try:
            # Get user stats
            user_id = str(user.id)
            user_stats = await self.monitor.get_user_stats(user_id)
            
            if not user_stats:
                await interaction.followup.send(f"No statistics available for {user.display_name}.")
                return
            
            # Create an embed for the user stats
            embed = discord.Embed(
                title=f"User Statistics for {user.display_name}",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            
            # Add basic stats
            embed.add_field(
                name="Total Interactions",
                value=str(user_stats.get("total_interactions", 0)),
                inline=True
            )
            
            embed.add_field(
                name="Success Rate",
                value=f"{user_stats.get('success_rate', 0):.2%}",
                inline=True
            )
            
            embed.add_field(
                name="Avg. Response Time",
                value=f"{user_stats.get('average_execution_time', 0):.2f}s",
                inline=True
            )
            
            # Add token usage if available
            total_tokens = user_stats.get("total_tokens", {}).get("total", 0)
            if total_tokens > 0:
                embed.add_field(
                    name="Total Tokens Used",
                    value=f"{total_tokens:,}",
                    inline=True
                )
            
            # Add last interaction time if available
            last_interaction = user_stats.get("last_interaction")
            if last_interaction:
                try:
                    last_time = datetime.fromisoformat(last_interaction)
                    time_ago = datetime.now() - last_time
                    if time_ago < timedelta(days=1):
                        time_str = f"{time_ago.seconds // 3600} hours ago"
                        if time_ago.seconds < 3600:
                            time_str = f"{time_ago.seconds // 60} minutes ago"
                    else:
                        time_str = f"{time_ago.days} days ago"
                    
                    embed.add_field(
                        name="Last Interaction",
                        value=time_str,
                        inline=True
                    )
                except:
                    pass
            
            # Add most used commands
            commands_used = user_stats.get("commands_used", {})
            if commands_used:
                # Sort commands by usage
                sorted_commands = sorted(commands_used.items(), key=lambda x: x[1], reverse=True)
                command_str = "\n".join([f"**{cmd}**: {count} times" for cmd, count in sorted_commands[:5]])
                
                embed.add_field(
                    name="Most Used Commands",
                    value=command_str or "No commands used",
                    inline=False
                )
            
            # Set the user's avatar as the thumbnail
            embed.set_thumbnail(url=user.display_avatar.url)
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in user-stats command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while generating user statistics. Please try again later.\n```{str(e)[:1500]}```")

async def setup(bot):
    await bot.add_cog(StatsCog(bot)) 