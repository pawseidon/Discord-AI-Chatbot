import discord
from discord.ext import commands
from discord import app_commands
import asyncio
import traceback

from bot_utilities.sentiment_utils import SentimentAnalyzer
from bot_utilities.monitoring import UserActivityMonitor
from bot_utilities.config_loader import config

class SentimentCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.sentiment_analyzer = SentimentAnalyzer()
        self.activity_monitor = UserActivityMonitor()
        
    @app_commands.command(name="sentiment", description="Analyze the sentiment of a message")
    @app_commands.describe(
        text="The text to analyze (leave blank to analyze a replied message)",
        public="Whether to show the analysis publicly (default: false)"
    )
    async def sentiment_command(self, interaction: discord.Interaction, text: str = None, public: bool = False):
        """
        Analyze the sentiment of a message or provided text
        
        Parameters:
        text (str, optional): Text to analyze. If not provided, will check for a replied message.
        public (bool, optional): Whether to show the analysis publicly.
        """
        await interaction.response.defer(ephemeral=not public, thinking=True)
        
        try:
            # Determine the text to analyze
            content_to_analyze = text
            source_text = text
            source_author = interaction.user
            
            # If no text provided, check if replying to a message
            if not content_to_analyze and hasattr(interaction.message, 'reference') and interaction.message.reference:
                # Get the message being replied to
                referenced_message_id = interaction.message.reference.message_id
                referenced_message = await interaction.channel.fetch_message(referenced_message_id)
                
                # Set the content to analyze and keep track of the source
                content_to_analyze = referenced_message.content
                source_text = referenced_message.content
                source_author = referenced_message.author
            
            # Validate we have text to analyze
            if not content_to_analyze:
                await interaction.followup.send(
                    "Please provide text to analyze or use this command in reply to a message.", 
                    ephemeral=True
                )
                return
            
            # Perform sentiment analysis
            analysis_result = await self.sentiment_analyzer.analyze_sentiment(content_to_analyze)
            
            # Format the results for display
            formatted_result = await self.sentiment_analyzer.format_sentiment_analysis(analysis_result)
            
            # Create an embed
            embed = self._create_sentiment_embed(formatted_result, source_text, source_author)
            
            # Send the response
            await interaction.followup.send(embed=embed, ephemeral=not public)
            
            # Log the activity
            asyncio.create_task(self.activity_monitor.log_command_usage(
                user_id=str(interaction.user.id),
                command_name="sentiment",
                guild_id=str(interaction.guild.id) if interaction.guild else "DM",
                success=True
            ))
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in sentiment command: {error_traceback}")
            await interaction.followup.send(
                f"❌ Error: I encountered a problem while analyzing sentiment.\n```{str(e)[:1500]}```", 
                ephemeral=True
            )
    
    def _create_sentiment_embed(self, formatted_result, source_text, source_author):
        """Create a rich embed for sentiment analysis results"""
        
        # Determine embed color based on sentiment
        color_map = {
            "Positive": discord.Color.green(),
            "Negative": discord.Color.red(),
            "Neutral": discord.Color.light_gray(),
            "Mixed": discord.Color.gold()
        }
        embed_color = color_map.get(formatted_result["sentiment"], discord.Color.blurple())
        
        # Create embed
        embed = discord.Embed(
            title=f"Sentiment Analysis {formatted_result['sentiment_emoji']}",
            description=formatted_result["summary"],
            color=embed_color
        )
        
        # Add sentiment information
        embed.add_field(
            name="Overall Sentiment",
            value=f"{formatted_result['sentiment_emoji']} {formatted_result['sentiment']} (Confidence: {formatted_result['confidence']})",
            inline=False
        )
        
        # Add detected emotions if available
        if formatted_result["formatted_emotions"]:
            embed.add_field(
                name="Detected Emotions",
                value="\n".join(formatted_result["formatted_emotions"]),
                inline=False
            )
        
        # Add the analyzed text (truncated if too long)
        max_text_length = 1000
        truncated_text = source_text[:max_text_length] + ("..." if len(source_text) > max_text_length else "")
        embed.add_field(
            name="Analyzed Text",
            value=f"```{truncated_text}```",
            inline=False
        )
        
        # Set author information
        embed.set_author(
            name=f"Requested by {source_author.display_name}",
            icon_url=source_author.display_avatar.url
        )
        
        # Set footer
        embed.set_footer(text="AI Sentiment Analysis")
        
        return embed

async def setup(bot):
    await bot.add_cog(SentimentCog(bot)) 