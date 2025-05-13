import discord
from discord.ext import commands
from discord import app_commands
import aiohttp
import traceback
import asyncio

from bot_utilities.multimodal_utils import ImageProcessor
from bot_utilities.monitoring import UserActivityMonitor
from bot_utilities.formatting_utils import chunk_message
from bot_utilities.config_loader import config

class VoiceCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.processor = ImageProcessor()
        self.activity_monitor = UserActivityMonitor()
        
    @app_commands.command(name="transcribe", description="Transcribe a voice message to text")
    @app_commands.describe(voice_message="The voice message to transcribe")
    async def transcribe_command(self, interaction: discord.Interaction, voice_message: discord.Attachment = None):
        """
        Transcribe a voice message to text
        
        Parameters:
        voice_message (Attachment, optional): Voice message to transcribe. If not provided, will check for a replied message with a voice attachment.
        """
        await interaction.response.defer(thinking=True)
        
        try:
            # Check if a voice message was provided directly
            if voice_message and voice_message.content_type and "audio" in voice_message.content_type:
                transcript = await self.processor.transcribe_audio(voice_message.url)
                await self._send_transcript_response(interaction, transcript, voice_message.url)
                return
                
            # If no direct attachment, check if the command is used in reply to a message with voice attachment
            if interaction.message and hasattr(interaction.message, 'reference') and interaction.message.reference:
                # Get the message being replied to
                referenced_message_id = interaction.message.reference.message_id
                referenced_message = await interaction.channel.fetch_message(referenced_message_id)
                
                # Check for voice attachment in the referenced message
                for attachment in referenced_message.attachments:
                    if attachment.content_type and "audio" in attachment.content_type:
                        transcript = await self.processor.transcribe_audio(attachment.url)
                        await self._send_transcript_response(interaction, transcript, attachment.url)
                        return
            
            # No valid voice message found
            await interaction.followup.send("Please provide a voice message to transcribe or use this command in reply to a message with a voice attachment.", ephemeral=True)
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in transcribe command: {error_traceback}")
            await interaction.followup.send(f"❌ Error: I encountered a problem while transcribing the voice message.\n```{str(e)[:1500]}```", ephemeral=True)
    
    async def _send_transcript_response(self, interaction, transcript, voice_url):
        """Send the transcript response with proper formatting"""
        # Create an embed for the response
        embed = discord.Embed(
            title="Voice Message Transcription",
            description=transcript,
            color=discord.Color.blue()
        )
        
        # Add the voice message link
        embed.add_field(name="Original Voice Message", value=f"[Voice Message]({voice_url})")
        
        # Add footer with attribution
        embed.set_footer(text=f"Transcribed for {interaction.user.display_name}")
        
        # Send the response
        await interaction.followup.send(embed=embed)
        
        # Log the activity
        asyncio.create_task(self.activity_monitor.log_command_usage(
            user_id=str(interaction.user.id),
            command_name="transcribe",
            guild_id=str(interaction.guild.id) if interaction.guild else "DM",
            success=True
        ))

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listen for voice messages with specific activation command"""
        # Don't respond to bot messages
        if message.author.bot:
            return
            
        # Check if the message has voice attachments and includes a transcription request
        if any(attachment.content_type and "audio" in attachment.content_type for attachment in message.attachments):
            # Check for a "!transcribe" or similar command at start of message
            if message.content.lower().startswith(("!transcribe", "?transcribe", ".transcribe")):
                # Get the voice attachment
                for attachment in message.attachments:
                    if "audio" in attachment.content_type:
                        # Send typing indicator
                        async with message.channel.typing():
                            # Process the voice message
                            transcript = await self.processor.transcribe_audio(attachment.url)
                            
                            # Create an embed for the response
                            embed = discord.Embed(
                                title="Voice Message Transcription",
                                description=transcript,
                                color=discord.Color.blue()
                            )
                            
                            # Add footer with attribution
                            embed.set_footer(text=f"Transcribed for {message.author.display_name}")
                            
                            # Send the response
                            await message.reply(embed=embed)
                            
                            # Log the activity
                            asyncio.create_task(self.activity_monitor.log_command_usage(
                                user_id=str(message.author.id),
                                command_name="transcribe_text_command",
                                guild_id=str(message.guild.id) if message.guild else "DM",
                                success=True
                            ))
                            
                            # Process only one voice attachment
                            break

async def setup(bot):
    await bot.add_cog(VoiceCog(bot)) 