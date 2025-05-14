"""
EmojiReactionCog - Handles emoji reactions for different reasoning types
"""

import discord
from discord.ext import commands
import logging
import asyncio
from typing import Dict, List, Set, Any, Optional

from bot_utilities.services.agent_service import agent_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('emoji_reaction_cog')

class EmojiReactionCog(commands.Cog):
    """Cog for handling emoji reactions to bot messages"""
    
    def __init__(self, bot):
        self.bot = bot
        self.message_reactions = {}  # Maps message IDs to added reaction types
        self.message_reasoning_types = {}  # Maps message IDs to sets of reasoning types
        self.messages_processed = set()  # Track messages that have already been processed
        self.reaction_queue = asyncio.Queue()  # Queue for rate-limited reaction adding
        self.bot.loop.create_task(self.process_reaction_queue())
    
    async def process_reaction_queue(self):
        """Process reaction queue to avoid rate limits"""
        while True:
            try:
                # Get next reaction task from queue
                message, emoji, operation = await self.reaction_queue.get()
                
                try:
                    if operation == "add":
                        await message.add_reaction(emoji)
                    elif operation == "remove":
                        await message.remove_reaction(emoji, self.bot.user)
                except discord.errors.HTTPException as e:
                    if e.status == 429:  # Rate limited
                        logger.warning(f"Rate limited when {operation}ing reaction {emoji}, waiting longer")
                        # Put back in queue with longer delay
                        await asyncio.sleep(2.0)
                        await self.reaction_queue.put((message, emoji, operation))
                    else:
                        logger.warning(f"Failed to {operation} reaction {emoji}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Unexpected error {operation}ing reaction {emoji}: {str(e)}")
                
                # Mark task as done
                self.reaction_queue.task_done()
                
                # Wait between operations to avoid rate limits
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in reaction queue processor: {str(e)}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def get_reasoning_emoji(self, reasoning_type: str) -> str:
        """Get the emoji for a specific reasoning type"""
        return await agent_service.get_agent_emoji(reasoning_type)
    
    async def add_reasoning_reactions(self, message: discord.Message, reasoning_types: List[str]) -> None:
        """Add emoji reactions based on reasoning types used"""
        if not message:
            return
        
        # Check if we've already processed this message
        if message.id in self.messages_processed:
            return
            
        # Mark this message as processed
        self.messages_processed.add(message.id)
            
        # Store the reasoning types for this message
        self.message_reasoning_types[message.id] = set(reasoning_types)
        
        # Add reactions for each reasoning type with rate limit handling
        for reasoning_type in reasoning_types:
            emoji = await self.get_reasoning_emoji(reasoning_type)
            
            # Queue the reaction add operation
            await self.reaction_queue.put((message, emoji, "add"))
    
    async def update_reasoning_reactions(self, message: discord.Message, new_reasoning_types: List[str]) -> None:
        """Update emoji reactions when reasoning types change"""
        if not message:
            return
            
        # If message not previously processed, just add reactions
        if message.id not in self.message_reasoning_types:
            return await self.add_reasoning_reactions(message, new_reasoning_types)
            
        current_types = self.message_reasoning_types[message.id]
        new_types_set = set(new_reasoning_types)
        
        # Remove reactions for reasoning types no longer used
        types_to_remove = current_types - new_types_set
        for reasoning_type in types_to_remove:
            emoji = await self.get_reasoning_emoji(reasoning_type)
            # Queue the reaction remove operation
            await self.reaction_queue.put((message, emoji, "remove"))
        
        # Add reactions for new reasoning types
        types_to_add = new_types_set - current_types
        for reasoning_type in types_to_add:
            emoji = await self.get_reasoning_emoji(reasoning_type)
            # Queue the reaction add operation
            await self.reaction_queue.put((message, emoji, "add"))
        
        # Update stored reasoning types
        self.message_reasoning_types[message.id] = new_types_set
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Listen for bot messages to add reactions"""
        # Skip non-bot messages or messages from other bots
        if not message.author.bot or message.author.id != self.bot.user.id:
            return
            
        # Skip system messages or ephemeral messages
        if not message.content or not message.guild:
            return
            
        # Check if we've already processed this message
        if message.id in self.messages_processed:
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

async def setup(bot):
    await bot.add_cog(EmojiReactionCog(bot)) 