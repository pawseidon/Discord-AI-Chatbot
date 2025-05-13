import discord
from discord.ext import commands
from discord import app_commands
import traceback
import asyncio
import io
import random
import time
from typing import Optional, List

from bot_utilities.multimodal_utils import ImageProcessor, ImageGenerator, MultimodalMessage
from bot_utilities.formatting_utils import chunk_message
from bot_utilities.config_loader import config

# Modern UI models for Prodia
MODERN_MODELS = [
    ("🌟 DreamShaper XL", "dreamshaperXL"),
    ("✨ MidJourney Style XL", "mjXLStyle"),
    ("🔮 Realistic Vision", "realvisxl"),
    ("🏙️ RPG", "rpg"),
    ("🖼️ Absolute Reality", "absolutereality"),
    ("🌅 Juggernaut", "juggernaut"),
    ("🧠 DreamShaper 8", "dreamshaper-8"),
    ("🌈 OpenJourney", "openjourney")
]

# Sampling methods
SAMPLING_METHODS = [
    ("DPM++ 2M Karras", "DPM++ 2M Karras"),
    ("Euler a", "Euler a"),
    ("DPM++ SDE Karras", "DPM++ SDE Karras")
]

class ImageCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.image_processor = ImageProcessor()
        self.image_generator = ImageGenerator()
        self.generation_cooldowns = {}  # User ID -> last generation time
        self.cooldown_time = 15  # seconds
        
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
                chunks = chunk_message(response)
                if len(chunks) > 1:
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

    @app_commands.command(name="generate", description="Generate an image from a text description")
    @app_commands.describe(
        prompt="Describe the image you want to generate",
        style="Select an image generation style/model",
        enhance="Automatically enhance your prompt for better results"
    )
    @app_commands.choices(style=[
        app_commands.Choice(name=name, value=value) for name, value in MODERN_MODELS
    ])
    async def generate_image_command(self, 
                                   interaction: discord.Interaction, 
                                   prompt: str,
                                   style: app_commands.Choice[str] = None,
                                   enhance: bool = True):
        """
        Generate an image from a text description
        
        Parameters:
        prompt (str): Description of the image to generate
        style (Choice, optional): The style/model to use
        enhance (bool, optional): Whether to enhance your prompt
        """
        # Check user cooldown to prevent spam
        user_id = str(interaction.user.id)
        current_time = time.time()
        
        if user_id in self.generation_cooldowns:
            time_since_last = current_time - self.generation_cooldowns[user_id]
            if time_since_last < self.cooldown_time:
                await interaction.response.send_message(
                    f"⏱️ Please wait {int(self.cooldown_time - time_since_last)} more seconds before generating another image.",
                    ephemeral=True
                )
                return
        
        # Set cooldown
        self.generation_cooldowns[user_id] = current_time
        
        # Defer reply to give time for processing
        await interaction.response.defer()
        
        # Get the selected model or use default
        model = style.value if style else "dreamshaper-8"
        
        # Check if prompt is too short
        if len(prompt.strip()) < 3:
            await interaction.followup.send("❌ Your prompt is too short. Please provide a more detailed description.")
            return
            
        # Create a status message
        status_message = await interaction.followup.send(
            f"🖌️ Generating image: `{prompt}`\n\n" + 
            f"💭 Style: {style.name if style else 'DreamShaper'}\n" +
            "⏳ This may take a moment..."
        )
        
        try:
            # Generate the image
            success, result, metadata = await self.image_generator.generate_image(
                prompt=prompt,
                provider="prodia",
                model=model,
                enhance=enhance
            )
            
            if not success:
                # Handle generation failure
                await status_message.edit(content=f"❌ Image generation failed: {result}")
                return
                
            # Create an embed for the result
            embed = discord.Embed(
                title="🎨 Generated Image",
                description=f"**Prompt:** {metadata.get('enhanced_prompt', prompt)}",
                color=discord.Color.from_rgb(99, 64, 255)
            )
            
            # Add style information
            embed.add_field(
                name="Style",
                value=style.name if style else "DreamShaper",
                inline=True
            )
            
            # Add generation time
            gen_time = metadata.get("generation_time", 0)
            embed.add_field(
                name="Generation Time", 
                value=f"{gen_time:.1f} seconds", 
                inline=True
            )
            
            # Create the file attachment
            image_file = discord.File(
                result, 
                filename="generated_image.png", 
                description=prompt
            )
            
            # Set the image in the embed
            embed.set_image(url="attachment://generated_image.png")
            
            # Add user attribution
            embed.set_footer(
                text=f"Generated for {interaction.user.display_name}",
                icon_url=interaction.user.display_avatar.url if interaction.user.display_avatar else None
            )
            
            # Update the message with the embed and image
            await status_message.edit(content=None, embed=embed, attachments=[image_file])
            
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in generate-image command: {error_traceback}")
            await status_message.edit(content=f"💥 Error: I encountered a problem while generating the image. Please try again later.\n```{str(e)[:1500]}```")

    @app_commands.command(name="imagine", description="Create multiple images from a prompt")
    @app_commands.describe(
        prompt="Describe what you want to see in the image",
        count="Number of images to generate (1-4)",
        public="Make the images visible to everyone in the channel"
    )
    async def imagine_command(self, 
                            interaction: discord.Interaction, 
                            prompt: str, 
                            count: int = 1,
                            public: bool = False):
        """
        Generate multiple images from a text prompt
        
        Parameters:
        prompt (str): Description of the images to generate
        count (int): Number of images to generate (1-4)
        public (bool): Whether to make the images visible to everyone
        """
        # Check user cooldown to prevent spam
        user_id = str(interaction.user.id)
        current_time = time.time()
        
        if user_id in self.generation_cooldowns:
            time_since_last = current_time - self.generation_cooldowns[user_id]
            if time_since_last < self.cooldown_time:
                await interaction.response.send_message(
                    f"⏱️ Please wait {int(self.cooldown_time - time_since_last)} more seconds before generating another image.",
                    ephemeral=True
                )
                return
        
        # Set cooldown
        self.generation_cooldowns[user_id] = current_time
        
        # Limit count to reasonable range
        count = max(1, min(4, count))
        
        # Defer reply
        await interaction.response.defer(ephemeral=not public)
        
        # Create a status message
        status_message = await interaction.followup.send(
            f"🎨 Generating {count} image{'s' if count > 1 else ''} for prompt: `{prompt}`\n⏳ Please wait...",
            ephemeral=not public
        )
        
        try:
            # Pick a random model for variety
            random_model = random.choice(MODERN_MODELS)[1]
            
            # Generate multiple images
            results = []
            success_count = 0
            
            for i in range(count):
                success, result, metadata = await self.image_generator.generate_image(
                    prompt=prompt,
                    provider="prodia",
                    model=random_model,
                    enhance=True
                )
                
                if success:
                    results.append(result)
                    success_count += 1
                
                # Update status message with progress
                await status_message.edit(content=f"🎨 Generating {count} image{'s' if count > 1 else ''}: {i+1}/{count} complete")
            
            # Create files from the results
            files = []
            for idx, image_data in enumerate(results):
                file = discord.File(image_data, filename=f"imagine_{idx+1}.png")
                files.append(file)
            
            # Create a new message with all the images
            if success_count > 0:
                # Create an embed for the result
                embed = discord.Embed(
                    title="✨ Imagination Results",
                    description=f"**Prompt:** {prompt}",
                    color=discord.Color.from_rgb(99, 64, 255)
                )
                
                # Add user attribution
                embed.set_footer(
                    text=f"Generated for {interaction.user.display_name}",
                    icon_url=interaction.user.display_avatar.url if interaction.user.display_avatar else None
                )
                
                await interaction.followup.send(
                    embed=embed,
                    files=files,
                    ephemeral=not public
                )
                
                # Delete the status message
                await status_message.delete()
            else:
                await status_message.edit(content="❌ Failed to generate any images. Please try a different prompt.")
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in imagine command: {error_traceback}")
            await status_message.edit(content=f"💥 Error: I encountered a problem while generating images. Please try again later.\n```{str(e)[:1500]}```")

async def setup(bot):
    await bot.add_cog(ImageCog(bot)) 