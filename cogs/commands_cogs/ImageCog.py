import discord
from discord.ext import commands
from discord import app_commands
import traceback
import asyncio
import io
from typing import Optional

from bot_utilities.multimodal_utils import ImageProcessor
from bot_utilities.formatting_utils import chunk_message

class ImageCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.image_processor = ImageProcessor()
        
    @app_commands.command(name="analyze-image", description="Analyze an image and describe it")
    @app_commands.describe(
        image="Image to analyze",
        prompt="Custom prompt for analysis (optional)"
    )
    async def analyze_image_command(self, interaction: discord.Interaction, 
                                    image: discord.Attachment, 
                                    prompt: str = "Describe this image in detail."):
        """
        Analyze an uploaded image and generate a detailed description
        
        Parameters:
        image (Attachment): The image to analyze
        prompt (str, optional): Custom instructions for analysis
        """
        # Defer reply to give time for processing
        await interaction.response.defer()
        
        # Check if the file is an image
        if not image.content_type or not image.content_type.startswith('image/'):
            await interaction.followup.send("❌ The uploaded file is not a recognized image format. Please upload a valid image.")
            return
        
        try:
            # Show typing indicator to indicate processing
            async with interaction.channel.typing():
                # Start a spinner indicator for long-running tasks
                message = await interaction.followup.send(f"🔄 Analyzing image: `{image.filename}`")
                
                # Process the image
                analysis = await self.image_processor.process_discord_attachment(image, prompt)
                
                # Format the response for Discord
                # Split into chunks if needed
                chunks = chunk_message(analysis)
                if len(chunks) > 1:
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=analysis)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in analyze-image command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while analyzing the image. Please try again later.\n```{str(e)[:1500]}```")
    
    @app_commands.command(name="ocr", description="Extract text from an image")
    @app_commands.describe(image="Image containing text to extract")
    async def ocr_command(self, interaction: discord.Interaction, image: discord.Attachment):
        """
        Extract text from an image (OCR)
        
        Parameters:
        image (discord.Attachment): The image containing text
        """
        # Check if the attachment is an image
        if not image.content_type or not image.content_type.startswith('image/'):
            await interaction.response.send_message("Please provide a valid image attachment.")
            return
        
        # Defer the response since OCR might take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Show typing indicator to indicate processing
            async with interaction.channel.typing():
                # Start a spinner indicator for long-running tasks
                message = await interaction.followup.send(f"🔄 Extracting text from image: `{image.filename}`")
                
                # Download the image
                image_data = await self.image_processor.download_image(image.url)
                
                if not image_data:
                    await message.edit(content="I couldn't download this image.")
                    return
                
                # Extract text from the image
                extracted_text = await self.image_processor.extract_text_from_image(image_data)
                
                # Format the response for Discord
                response = f"**Extracted Text:**\n\n{extracted_text}"
                
                # Split into chunks if needed
                if len(response) > 2000:
                    chunks = chunk_message(response)
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in ocr command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while extracting text from the image. Please try again later.\n```{str(e)[:1500]}```")

async def setup(bot):
    await bot.add_cog(ImageCog(bot)) 