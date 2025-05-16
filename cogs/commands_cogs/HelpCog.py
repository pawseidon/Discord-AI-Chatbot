import discord
from discord.ext import commands


from ..common import current_language


class HelpCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.help_trigger_phrases = [
            "help", "how do i use", "how to use", "what can you do", 
            "commands", "features", "capabilities", "bot help",
            "show help", "user guide", "instructions", "how does this work",
            "what commands", "assistance", "tutorial", "guide me"
        ]

    # Listen for regular messages asking for help
    @commands.Cog.listener()
    async def on_message(self, message):
        # Skip if message is from a bot
        if message.author.bot:
            return
            
        # Check if this is a help request
        is_help_request = False
        
        # Check if the bot is mentioned directly
        is_mentioned = self.bot.user in message.mentions
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Get the content without the mention
        content = message.content.lower()
        if is_mentioned:
            # Remove the mention from the content
            content = content.replace(f'<@{self.bot.user.id}>', '').strip()
            content = content.replace(f'<@!{self.bot.user.id}>', '').strip()
            
        # Check for help triggers
        for trigger in self.help_trigger_phrases:
            if trigger in content:
                is_help_request = True
                break
                
        # Process help request if:
        # 1. It's a help request AND (bot is mentioned OR in DM)
        # 2. The message is exactly "help" (case insensitive)
        if (is_help_request and (is_mentioned or is_dm)) or content.strip() == "help":
            await self.send_help(message.channel)

    async def send_help(self, channel):
        # Create the main help embed
        embed = discord.Embed(
            title="Bot Features & Context-Aware AI Assistant",
            description="I'm a fully context-aware AI assistant. Just talk to me naturally and I'll understand what you need - no commands necessary!",
            color=0x03a64b
        )
        embed.set_thumbnail(url=self.bot.user.avatar.url)
        
        # Add info about natural language interaction
        embed.add_field(
            name="💬 Natural Conversation",
            value=(
                "Just mention me or use my name to start a conversation. I'll automatically detect what you need!\n"
                "Examples:\n"
                "• `@Bot tell me about quantum computing`\n"
                "• `Hey bot, what's the weather like today?`\n"
                "• Reply to my messages to continue our conversation"
            ),
            inline=False
        )
        
        # Add reasoning system information
        embed.add_field(
            name="🧠 Context-Aware Reasoning System",
            value=(
                "I automatically analyze your query and select the most appropriate reasoning approach:\n\n"
                "• 🧠 **Sequential Thinking** - Complex step-by-step reasoning\n"
                "• 🔍 **Information Retrieval** - Search and fact-finding\n"
                "• 💬 **Conversational** - General chats and discussions\n"
                "• 📚 **Knowledge Base** - Educational explanations\n"
                "• ✅ **Verification** - Fact-checking and validation\n"
                "• 🕸️ **Graph-of-Thought** - Mapping relationships between ideas\n"
                "• ⛓️ **Chain-of-Thought** - Logical reasoning progression\n"
                "• 🔄 **ReAct Reasoning** - Reasoning with action capabilities\n"
                "• 🎨 **Creative Mode** - Imagination and creative content\n"
                "• 🔎 **Step-Back Analysis** - High-level perspective\n"
                "• 👥 **Multi-Agent** - Multiple perspectives on a topic"
            ),
            inline=False
        )
        
        # Add reasoning preference instructions
        embed.add_field(
            name="⚙️ Setting Your Preferences",
            value=(
                "You can control how I respond to you through natural language:\n"
                "• `Set my reasoning mode to sequential` - Use sequential thinking\n"
                "• `Change reasoning mode to creative` - Switch to creative mode\n"
                "• Include the emoji at the start of your message (e.g., 🔍 for search)\n"
                "• Ask `Explain reasoning modes` to learn more"
            ),
            inline=False
        )
        
        # Add privacy controls
        embed.add_field(
            name="🔒 Privacy Controls",
            value=(
                "You can control your data with these natural language commands:\n"
                "• `Clear my data` - Remove all your data from my memory\n"
                "• `Delete my history` - Alternative way to clear your data\n"
                "• `Forget me` - Remove all your personal information\n"
                "\nYou can also use the `/clear` command to clear your data"
            ),
            inline=False
        )
        
        # Add conversation reset option
        embed.add_field(
            name="🔄 Conversation Reset",
            value=(
                "To start a fresh conversation:\n"
                "• Say `Reset our conversation` to clear current context\n"
                "• Use the `/reset` command to reset the current conversation\n"
                "This keeps your preferences but clears the current conversation context"
            ),
            inline=False
        )
        
        # Add advanced reasoning examples
        embed.add_field(
            name="🧩 Example Reasoning Triggers",
            value=(
                "Different query styles automatically trigger appropriate reasoning:\n"
                "• `Step by step, how does nuclear fusion work?` → Sequential thinking\n"
                "• `Map the connections between climate change and economics` → Graph-of-thought\n"
                "• `Verify whether bananas contain more potassium than oranges` → Verification\n"
                "• `Search for the latest news about AI` → Information retrieval\n"
                "• `Create a story about a magical forest` → Creative mode"
            ),
            inline=False
        )
        
        # Add multi-agent capabilities
        embed.add_field(
            name="👥 Multi-Agent Capabilities",
            value=(
                "My advanced architecture uses multiple specialized agents working together:\n"
                "• Agents automatically select the best reasoning approach for your query\n"
                "• Complex tasks are broken down and delegated to specialized agents\n"
                "• Agents can access tools like web search, calculators, and more\n"
                "• Everything happens automatically - just ask your question naturally!"
            ),
            inline=False
        )

        # Add emoji toggle feature
        embed.add_field(
            name="**Emoji Controls**",
            value=(
                "• You can control emojis with natural language: \"disable emojis\" or \"enable emojis\""
            ),
            inline=False
        )

        embed.set_footer(text=f"{current_language['help_footer']}")
        
        await channel.send(embed=embed)
        
    @commands.hybrid_command(name="help", description="Display help information about the bot")
    async def help_command(self, ctx):
        """Display help information about the bot"""
        await self.send_help(ctx.channel)

async def setup(bot):
    await bot.add_cog(HelpCog(bot))
