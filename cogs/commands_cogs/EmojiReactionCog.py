import discord
from discord.ext import commands
import asyncio
from typing import Dict, List, Set, Any, Optional

from bot_utilities.services.agent_service import agent_service

class EmojiReactionCog(commands.Cog):
    """A cog to handle emoji reactions for different reasoning types"""
    
    def __init__(self, bot):
        self.bot = bot
        self.message_reasoning_types = {}  # Maps message IDs to sets of reasoning types
        self.processing_messages = {}  # Maps message IDs to processing status and timestamps
    
    async def get_reasoning_emoji(self, reasoning_type: str) -> str:
        """Get the emoji for a specific reasoning type"""
        return await agent_service.get_agent_emoji(reasoning_type)
    
    async def add_reasoning_reactions(self, message: discord.Message, reasoning_types: List[str]) -> None:
        """Add emoji reactions based on reasoning types used"""
        if not message:
            return
            
        # Store the reasoning types for this message
        self.message_reasoning_types[message.id] = set(reasoning_types)
        
        # Add reactions for each reasoning type
        for reasoning_type in reasoning_types:
            emoji = await self.get_reasoning_emoji(reasoning_type)
            try:
                await message.add_reaction(emoji)
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
            except discord.errors.HTTPException:
                # This can happen if emoji is invalid or permissions are missing
                pass
    
    async def update_reasoning_reactions(self, message: discord.Message, new_reasoning_types: List[str]) -> None:
        """Update emoji reactions when reasoning types change"""
        if not message or message.id not in self.message_reasoning_types:
            return await self.add_reasoning_reactions(message, new_reasoning_types)
            
        current_types = self.message_reasoning_types[message.id]
        new_types_set = set(new_reasoning_types)
        
        # Remove reactions for reasoning types no longer used
        types_to_remove = current_types - new_types_set
        for reasoning_type in types_to_remove:
            emoji = await self.get_reasoning_emoji(reasoning_type)
            try:
                await message.remove_reaction(emoji, self.bot.user)
                await asyncio.sleep(0.5)
            except discord.errors.HTTPException:
                pass
        
        # Add reactions for new reasoning types
        types_to_add = new_types_set - current_types
        for reasoning_type in types_to_add:
            emoji = await self.get_reasoning_emoji(reasoning_type)
            try:
                await message.add_reaction(emoji)
                await asyncio.sleep(0.5)
            except discord.errors.HTTPException:
                pass
        
        # Update stored reasoning types
        self.message_reasoning_types[message.id] = new_types_set
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Listen for bot messages to add reactions"""
        # Skip messages not from the bot
        if message.author.id != self.bot.user.id:
            return
            
        # Skip system messages
        if not message.content:
            return
            
        # Skip messages that are ephemeral or in DMs
        if not message.guild:
            return
            
        # Try to detect reasoning types from the message content
        reasoning_types = await self.detect_reasoning_types_from_message(message.content)
            
        # Add reactions
        if reasoning_types:
            await self.add_reasoning_reactions(message, reasoning_types)
    
    async def detect_reasoning_types_from_message(self, content: str) -> List[str]:
        """Detect reasoning types from message content"""
        reasoning_types = []
        
        # Check for emoji indicators at the start of the message
        emoji_map = {
            "💬": "conversational",
            "📚": "rag",
            "🔄": "sequential",
            "🧠": "knowledge",
            "✅": "verification",
            "🎨": "creative",
            "🧮": "calculation",
            "📝": "planning",
            "📊": "graph",
            "👥": "multi_agent",
            "⚡": "react",
            "🔍": "cot",
            "🔙": "step_back",
            "🔗": "workflow",
            "🪞": "reflection",
            "🧩": "synthesis"
        }
        
        # Check for emoji indicators in the message
        for emoji, reasoning_type in emoji_map.items():
            if emoji in content[:100]:
                reasoning_types.append(reasoning_type)
                
        # If no emoji indicators, check for keywords
        if not reasoning_types:
            if any(term in content.lower() for term in ["step by step", "sequentially", "break down", "analyze"]):
                reasoning_types.append("sequential")
                
            if any(term in content.lower() for term in ["search", "find", "latest", "information about", "recent"]):
                reasoning_types.append("rag")
                
            if any(term in content.lower() for term in ["verify", "check", "confirm", "fact check"]):
                reasoning_types.append("verification")
                
            if any(term in content.lower() for term in ["create", "imagine", "story", "creative", "generate"]):
                reasoning_types.append("creative")
                
            if any(term in content.lower() for term in ["calculate", "compute", "solve", "equation"]):
                reasoning_types.append("calculation")
                
            if any(term in content.lower() for term in ["map", "graph", "network", "connections", "relations"]):
                reasoning_types.append("graph")
                
            if any(term in content.lower() for term in ["multiple agents", "perspectives", "collaborate"]):
                reasoning_types.append("multi_agent")
        
        # If no reasoning types detected, default to conversational
        if not reasoning_types:
            reasoning_types.append("conversational")
            
        return reasoning_types
        
    @commands.hybrid_command(name="toggleactive", description="Toggle the bot active/inactive in the current channel")
    @commands.has_permissions(administrator=True)
    async def toggleactive_command(self, ctx):
        """Toggle the bot active/inactive in the current channel"""
        # Get the ChatConfigCog to handle the command
        chat_config_cog = self.bot.get_cog('ChatConfigCog')
        if chat_config_cog:
            # Invoke the actual implementation
            await chat_config_cog.toggle_channel(ctx)
        else:
            await ctx.reply("❌ The channel configuration system isn't loaded.")
        
    @commands.hybrid_command(name="toggleinactive", description="Toggle the bot inactive/active in the current channel")
    @commands.has_permissions(administrator=True)
    async def toggleinactive_command(self, ctx):
        """Toggle the bot inactive/active in the current channel - alias for toggleactive"""
        await self.toggleactive_command(ctx)
        
    @commands.hybrid_command(name="help", description="Display help information about the bot")
    async def help_command(self, ctx):
        """Display help information about the bot"""
        # Get the HelpCog to handle the command
        help_cog = self.bot.get_cog('HelpCog')
        if help_cog:
            # Invoke the actual implementation
            await help_cog.help_command(ctx)
        else:
            # Fallback help if HelpCog is not available
            embed = discord.Embed(
                title="Discord AI Chatbot Help",
                description="I'm a versatile AI chatbot with multi-agent reasoning capabilities.",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="👋 Getting Started",
                value="Mention me or reply to my messages to chat with me.",
                inline=False
            )
            
            embed.add_field(
                name="📚 Commands",
                value="`/help` - Show this help message\n"
                      "`/clear` - Clear your conversation history\n"
                      "`/reset` - Reset current conversation\n"
                      "`/toggleactive` - Toggle the bot in a channel (admin)",
                inline=False
            )
            
            embed.add_field(
                name="🧠 Reasoning Types",
                value="I use different reasoning modes based on your query:\n"
                      "- 💬 Conversational: Natural chat\n"
                      "- 📚 RAG: Research and information retrieval\n"
                      "- 🔄 Sequential: Step-by-step analysis\n"
                      "- ✅ Verification: Fact-checking\n"
                      "- And many more!",
                inline=False
            )
            
            await ctx.reply(embed=embed)
        
    @commands.hybrid_command(name="clear", description="Clear your conversation history and data")
    async def clear_command(self, ctx):
        """Clear your conversation history and data"""
        # Get the ReasoningCog to handle the command
        reasoning_cog = self.bot.get_cog('ReasoningCog')
        if reasoning_cog:
            # Invoke the actual implementation
            await reasoning_cog.clear_data_command(ctx)
        else:
            await ctx.reply("❌ Sorry, I couldn't clear your data at this time.")
        
    @commands.hybrid_command(name="reset", description="Reset the current conversation but keep your preferences")
    async def reset_command(self, ctx):
        """Reset the current conversation but keep your preferences"""
        # Get the ReasoningCog to handle the command
        reasoning_cog = self.bot.get_cog('ReasoningCog')
        if reasoning_cog:
            # Invoke the actual implementation
            await reasoning_cog.reset_conversation_command(ctx)
        else:
            await ctx.reply("❌ Sorry, I couldn't reset the conversation at this time.")

async def setup(bot):
    await bot.add_cog(EmojiReactionCog(bot)) 