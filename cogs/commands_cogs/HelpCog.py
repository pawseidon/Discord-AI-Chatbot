import discord
from discord.ext import commands


from ..common import current_language


class HelpCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.hybrid_command(name="help", description=current_language["help"])
    async def help(self, ctx):
        # Create the main help embed
        embed = discord.Embed(
            title="Bot Commands & Features",
            description="You can interact with me using natural language - no need for slash commands! Just chat like you would with a person.",
            color=0x03a64b
        )
        embed.set_thumbnail(url=self.bot.user.avatar.url)
        
        # Add info about admin slash commands
        embed.add_field(
            name="📋 Administrative Commands",
            value=(
                "`/help` - Show this help message\n"
                "`/toggleactive` - Enable/disable the bot in a channel\n"
            ),
            inline=False
        )
        
        # Add natural language conversation examples
        embed.add_field(
            name="💬 Natural Conversation",
            value=(
                "Just mention me or use trigger words to start a conversation.\n"
                "Examples:\n"
                "• `@Bot tell me about quantum computing`\n"
                "• `Hey bot, what's the weather like today?`\n"
                "• Reply to my messages to continue our conversation"
            ),
            inline=False
        )
        
        # Add image generation examples
        embed.add_field(
            name="🎨 Image Generation",
            value=(
                "Ask me to create images with natural language.\n"
                "Examples:\n"
                "• `Generate an image of a sunset over mountains`\n"
                "• `Draw a picture of a cat wearing sunglasses`\n"
                "• `Imagine a futuristic cityscape`"
            ),
            inline=False
        )
        
        # Add image analysis examples
        embed.add_field(
            name="🔍 Image Analysis",
            value=(
                "Upload an image with a prompt to analyze it.\n"
                "Examples:\n"
                "• `What do you see in this image?` (with attachment)\n"
                "• `Describe this picture` (with attachment)\n"
                "• `Extract text from this screenshot` (with attachment)"
            ),
            inline=False
        )
        
        # Add voice transcription examples
        embed.add_field(
            name="🎙️ Voice Transcription",
            value=(
                "Upload a voice message to have it transcribed.\n"
                "Examples:\n"
                "• Send a voice message with text `!transcribe`\n"
                "• Upload voice message and ask `Transcribe this voice message`\n"
                "• Reply to a voice message with `What does this say?`"
            ),
            inline=False
        )
        
        # Add sentiment analysis examples
        embed.add_field(
            name="😊 Sentiment Analysis",
            value=(
                "Ask me to analyze the sentiment of text.\n"
                "Examples:\n"
                "• `What's the sentiment of \"I'm really happy with this product\"?`\n"
                "• `Analyze the emotions in this message` (in reply to a message)\n"
                "• `Is this review positive or negative?`"
            ),
            inline=False
        )
        
        # Add web search examples
        embed.add_field(
            name="🔎 Web Search",
            value=(
                "Ask me to search the web for information.\n"
                "Examples:\n"
                "• `Search for the latest news about AI`\n"
                "• `Find information about climate change`\n"
                "• `Research the history of the Roman Empire`"
            ),
            inline=False
        )
        
        # Add additional agent features
        embed.add_field(
            name="🧠 Advanced Features",
            value=(
                "I can perform many other tasks seamlessly:\n"
                "• Create threads for longer conversations\n"
                "• Remember context from previous messages\n"
                "• Analyze complex topics step-by-step\n"
                "• And much more!"
            ),
            inline=False
        )

        embed.set_footer(text=f"{current_language['help_footer']}")
        
        await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(HelpCog(bot))
