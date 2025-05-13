import sys
import time
import traceback
import discord
import asyncio
import json
import random
import datetime
import logging
import re
import os
import inspect
from typing import Dict, List, Any, Tuple, Optional, Union
from discord.ext import commands

from bot_utilities.response_utils import split_response
from bot_utilities.ai_utils import generate_response, get_crypto_price, text_to_speech, get_bot_names_and_triggers
from bot_utilities.ai_utils import should_use_sequential_thinking, is_bot_master, add_owner_context
from bot_utilities.memory_utils import UserPreferences, process_conversation_history, get_enhanced_instructions
from bot_utilities.news_utils import get_news_context
from bot_utilities.formatting_utils import format_response_for_discord, find_and_format_user_mentions, chunk_message
from bot_utilities.config_loader import config, load_current_language, load_instructions
from bot_utilities.multimodal_utils import ImageProcessor, ImageGenerator
from bot_utilities.sentiment_utils import SentimentAnalyzer
from bot_utilities.agent_utils import run_agent
from bot_utilities.reasoning_router import create_reasoning_router
from bot_utilities.router_compatibility import create_router_adapter
from bot_utilities.sequential_thinking import create_sequential_thinking
from bot_utilities.context_manager import get_context_manager, start_context_manager_tasks
from ..common import message_history, replied_messages, active_channels, server_settings, smart_mention
from bot_utilities.mcp_utils import MCPToolsManager

# Get config values
MAX_HISTORY = config.get('MAX_HISTORY', 8)
prevent_nsfw = config.get('AI_NSFW_CONTENT_FILTER', True)
blacklisted_words = config.get('BLACKLIST_WORDS', [])
allow_dm = config.get('ALLOW_DM', True)  # Get DM setting from config

# UX Configuration
USE_EMBEDS = config.get('USE_EMBEDS', False)  # Whether to use embeds for responses
SHOW_THINKING_INDICATOR = config.get('SHOW_THINKING_INDICATOR', True)  # Show typing/thinking indicators
USE_MARKDOWN_FORMATTING = config.get('USE_MARKDOWN_FORMATTING', True)  # Use Discord markdown for formatting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('on_message')

# Helper function to check if a channel is in the active channels list
def is_active_channel(channel, guild_id):
    """Check if a channel is in the active channels list for the guild"""
    guild_id = str(guild_id)
    channel_id = str(channel.id)
    
    # If the guild is not in active_channels, all channels are active by default
    if guild_id not in active_channels:
        return True
    
    # If the guild has no specific channels listed, all channels are active
    if not active_channels[guild_id]:
        return True
    
    # Check if this channel is in the active list
    return channel_id in active_channels[guild_id]

class OnMessage(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_channels = active_channels
        self.instructions = load_instructions()  # Load instructions directly
        self.thread_contexts = {}  # Store parent message context for threads
        self.STREAM_UPDATE_INTERVAL = 0.5  # Update streaming messages every 0.5 seconds
        self.replied_messages = replied_messages
        
        # Initialize processors for various capabilities
        self.image_processor = ImageProcessor()
        self.image_generator = ImageGenerator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.mcp_manager = MCPToolsManager()
        # Initialize sequential thinking
        self.sequential_thinking = create_sequential_thinking(llm_provider=None)
        
        # Start context manager tasks
        asyncio.create_task(self.start_context_manager())

    async def start_context_manager(self):
        """Start the context manager background tasks"""
        await start_context_manager_tasks()
        # Initialize AI provider for sequential thinking
        await self._init_sequential_thinking()

    async def _init_sequential_thinking(self):
        """Initialize the sequential thinking AI provider"""
        try:
            from bot_utilities.ai_utils import get_ai_provider
            provider = await get_ai_provider()
            
            # Safely set the provider on the instance
            if hasattr(self, 'sequential_thinking') and self.sequential_thinking:
                await self.sequential_thinking.set_llm_provider(provider)
                print("Successfully initialized sequential thinking provider")
        except Exception as e:
            print(f"Warning: Could not initialize sequential thinking provider: {e}")
            # Continue without a provider - will use fallbacks

    def get_custom_username(self, message):
        """Get a formatted username for the message author"""
        if message.guild and message.author.nick:
            return message.author.nick
        return message.author.display_name

    async def clean_message_content(self, message):
        """Clean and format message content for processing"""
        # Start with the raw content
        content = message.content
        
        # Remove mentions to the bot itself
        if self.bot.user in message.mentions:
            content = content.replace(f'<@{self.bot.user.id}>', '').replace(f'<@!{self.bot.user.id}>', '')
        
        # Remove command prefixes
        bot_names, triggers, prefixes, suffixes = get_bot_names_and_triggers()
        for prefix in prefixes:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()
                break
                
        # Clean up extra whitespace
        content = content.strip()
        
        return content

    async def get_agent_prompt(self, message):
        """Get agent prompt instructions for advanced reasoning"""
        # Base agent instructions
        agent_prompt = """You are an AI assistant with advanced reasoning capabilities.
You can use tools, search for information, and solve complex problems.
Think step-by-step and consider all available information before responding."""
        
        # Add user context if available
        if hasattr(message, 'author') and message.author:
            agent_prompt += f"\n\nYou are talking to {message.author.display_name}."
            
        # Add server context if available
        if message.guild:
            agent_prompt += f"\nYou are in the server: {message.guild.name}"
            
        # Add channel context if available
        if hasattr(message, 'channel') and message.channel:
            agent_prompt += f"\nThe conversation is happening in #{message.channel.name}"
            
        return agent_prompt

    async def detect_intent(self, message_content, message):
        """Detect user intent from message content"""
        message_lower = message_content.lower()
        
        # Image generation intent patterns
        image_gen_patterns = [
            r"generate (?:an |a |some )?image(s)? of (.+)",
            r"create (?:an |a |some )?image(s)? of (.+)",
            r"show (?:me |us )?(?:an |a |some )?image(s)? of (.+)",
            r"imagine (.+)",
            r"visualize (.+)",
            r"draw (?:an |a |some )?(image|picture) of (.+)"
        ]
        
        # Image analysis intent patterns
        image_analysis_patterns = [
            r"(?:analyze|describe|what's in|what is in|tell me about) (?:this |the |my )?(?:image|picture|photo)",
            r"explain (?:this |the |my )?(?:image|picture|photo)",
            r"what (?:can you |do you |)see in (?:this |the |my )?(?:image|picture|photo)",
            r"ocr (?:this |the |my )?(?:image|picture|photo|screenshot)",
            r"extract text from (?:this |the |my )?(?:image|picture|photo|screenshot)"
        ]
        
        # Voice transcription intent patterns
        transcription_patterns = [
            r"(?:transcribe|convert) (?:this |the |my )?(?:voice message|audio|speech)",
            r"what does (?:this |the |my )?(?:voice message|audio|speech) say",
            r"!transcribe"
        ]
        
        # Sentiment analysis intent patterns
        sentiment_patterns = [
            r"(?:analyze|what is|what's) the sentiment of (.+)",
            r"(?:how does|what's the emotion|what emotion|what feeling|what's the feeling) (?:in|of|behind) (.+)",
            r"is (.+) (?:positive|negative|neutral)",
            r"sentiment analysis(?: for| of)? (.+)",
            r"analyze emotions(?: in| of)? (.+)"
        ]
        
        # Web search intent patterns
        search_patterns = [
            r"(?:search|find|look up|google|research) (.+)",
            r"what is the latest(?: news| information) (?:about|on) (.+)",
            r"find information (?:about|on) (.+)"
        ]
        
        # Sequential thinking intent patterns
        sequential_thinking_patterns = [
            r"(?:use |try |apply |do )?sequential thinking(?: for| on| about)? (.+)",
            r"(?:solve |think through |break down |analyze |approach )(?:this |the |my )?(?:problem|question|task)(?: step by step| sequentially| step-by-step)? (.+)",
            r"(?:step by step|step-by-step|sequentially) (?:solve|approach|analyze|think about) (.+)",
            r"(?:help me|can you|please) (?:solve|approach|analyze|work through) (?:this|the following) (?:step by step|systematically|sequentially) (.+)"
        ]
        
        # MCP Agent intent patterns
        mcp_agent_patterns = [
            r"(?:use |try |apply |with )?(?:mcp|agent)(?: for| on| about)? (.+)",
            r"(?:agent|assistant)(?: help me| can you help| please help)(?: with)? (.+)"
        ]
        
        # Check for attached images for analysis
        has_image = False
        image_attachment = None
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                has_image = True
                image_attachment = attachment
                break
        
        # Check for voice message for transcription
        has_voice = False
        voice_attachment = None
        for attachment in message.attachments:
            if attachment.content_type and "audio" in attachment.content_type:
                has_voice = True
                voice_attachment = attachment
                break
        
        # Check if the message contains image generation intent
        for pattern in image_gen_patterns:
            match = re.search(pattern, message_lower)
            if match:
                # Get the prompt from the last capturing group
                groups = match.groups()
                if len(groups) >= 1:
                    # The last group contains the prompt
                    prompt = groups[-1].strip()
                    return {"intent": "generate_image", "prompt": prompt}
        
        # Check if there's an image to analyze
        if has_image:
            for pattern in image_analysis_patterns:
                if re.search(pattern, message_lower):
                    # Check for OCR specifically
                    if "ocr" in message_lower or "extract text" in message_lower:
                        return {"intent": "ocr_image", "attachment": image_attachment}
                    else:
                        return {"intent": "analyze_image", "attachment": image_attachment}
            
            # If image is attached without specific instruction, default to analysis
            return {"intent": "analyze_image", "attachment": image_attachment}
        
        # Check if there's a voice message to transcribe
        if has_voice:
            for pattern in transcription_patterns:
                if re.search(pattern, message_lower):
                    return {"intent": "transcribe_voice", "attachment": voice_attachment}
            
            # If voice is attached with "!transcribe", default to transcription
            if "!transcribe" in message_lower:
                return {"intent": "transcribe_voice", "attachment": voice_attachment}
        
        # Check for sentiment analysis intent
        for pattern in sentiment_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    # If there's a specified text to analyze
                    text_to_analyze = groups[0].strip()
                    return {"intent": "analyze_sentiment", "text": text_to_analyze}
                elif message.reference and message.reference.resolved:
                    # If no text provided but it's a reply, analyze the replied message
                    return {"intent": "analyze_sentiment", "referenced_message": message.reference.resolved}
        
        # Check for sequential thinking intent
        for pattern in sequential_thinking_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    problem = groups[0].strip()
                    return {"intent": "sequential_thinking", "problem": problem}
        
        # Check for MCP agent intent
        for pattern in mcp_agent_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "mcp_agent", "query": query}
        
        # Check for web search intent
        for pattern in search_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "web_search", "query": query}
        
        # Default: No specific intent detected
        return {"intent": "chat", "message": message_content}

    async def process_message(self, message):
        """
        Process an incoming message and generate an appropriate response
        """
        # Ignore messages from self to prevent loops
        if message.author == self.bot.user:
            return

        # Check if the message is a DM
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # If DMs are not allowed and this is a DM, ignore
        if not allow_dm and is_dm:
            return
            
        # Get clean message content without bot mention
        clean_content = await self.clean_message_content(message)
        
        # For a Guild (server) message, check if the bot should respond
        if not is_dm:
            # If no content after removing mention, ignore
            if not clean_content.strip():
                return
                
            # Check if the bot was mentioned or if the channel is active
            was_mentioned = self.bot.user.mentioned_in(message)
            is_active = is_active_channel(message.channel, message.guild.id)
            is_smart_mention = smart_mention(clean_content, self.bot)
            
            # Handle thread messages with a reply
            in_thread_with_reply = isinstance(message.channel, discord.Thread) and message.type == discord.MessageType.reply
            
            # If not mentioned and not in an active channel, ignore
            if not (was_mentioned or is_active or is_smart_mention or in_thread_with_reply):
                return
        
        try:
            # Detect intent
            intent_data = await self.detect_intent(clean_content, message)
            
            # Handle the detected intent
            if intent_data and isinstance(intent_data, dict):
                await self.handle_intent(message, intent_data)
            else:
                # Fallback to general reasoning if intent detection fails
                username = self.get_custom_username(message)
                user_id = str(message.author.id)
                channel_id = str(message.channel.id)
                await self.handle_message_with_reasoning_router(message, clean_content, username, user_id, channel_id)
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
            traceback.print_exc()

    async def handle_message_with_reasoning_router(self, message, clean_content, username, user_id, channel_id):
        """
        Handle message processing using the reasoning router
        
        This handles complex routing to the appropriate reasoning method
        """
        try:
            # Start processing indicator
            initial_message = None
            if SHOW_THINKING_INDICATOR:
                try:
                    # Use an animated emoji for the thinking indicator if possible
                    initial_message = await message.channel.send("Thinking... <a:thinking:1174880300425388042>")
                except:
                    # Fallback to simple text if emoji not available
                    try:
                        initial_message = await message.channel.send("Thinking...")
                    except:
                        pass  # If we can't send the indicator, just continue
            
            # Get thread_id for context tracking if we're in a thread
            thread_id = channel_id if isinstance(message.channel, discord.Thread) else None
            
            # Create response context
            context_manager = get_context_manager()
            context = {}
            if context_manager:
                try:
                    context = context_manager.build_context(
                        user_id=user_id,
                        channel_id=channel_id,
                        thread_id=thread_id
                    )
                except AttributeError:
                    # Fallback if build_context method doesn't exist
                    print("Warning: context_manager.build_context method not found")
            
            # Add enhanced instructions from configurator if available
            guild_id = str(message.guild.id) if message.guild else None
            enhanced_instructions = None
            
            # Analyze query to see if it needs sequential thinking
            should_use_sequential = self._should_use_sequential_thinking(clean_content)
            
            # Generator router instance if needed
            router = None
            if not hasattr(self, 'router_adapter') or not self.router_adapter:
                try:
                    from bot_utilities.router_compatibility import create_router_adapter
                    self.router_adapter = create_router_adapter()
                    router = self.router_adapter
                except Exception as e:
                    print(f"Error creating router adapter: {e}")
            else:
                router = self.router_adapter
            
            if router is None:
                # Fallback if we can't get a router
                await message.channel.send("I'm having trouble processing your request. Please try again later.")
                return
            
            # Generate content from router
            force_method = "sequential" if should_use_sequential else None
            response = await router.process(
                query=clean_content,
                username=username,
                user_id=user_id,
                channel_id=channel_id,
                guild_id=guild_id,
                history=context.get("history", []),
                enhanced_instructions=enhanced_instructions,
                enhanced_context=context,
                force_method=force_method
            )
            
            # Determine if this is a sequential thinking style response
            metadata = response[1] if isinstance(response, tuple) and len(response) > 1 else {}
            method = metadata.get("method", "standard")
            
            # Check for sequential thinking indicators in the response
            response_text = response[0] if isinstance(response, tuple) and len(response) > 0 else str(response)
            has_sequential_markers = "**Step" in response_text or "**Thought" in response_text
            
            # Determine if we should treat this as sequential thinking
            is_sequential = (
                method in ["sequential", "cot", "got", "list", "cov"] or 
                has_sequential_markers or
                should_use_sequential
            )
            
            # Use our unified response method
            await self.send_response(
                message=message, 
                response=response, 
                is_sequential=is_sequential,
                problem=clean_content,
                initial_message=initial_message
            )
            
        except Exception as e:
            print(f"Error in handle_message_with_reasoning_router: {e}")
            traceback.print_exc()
            try:
                await message.channel.send(
                    "I encountered an error while processing your message. Please try again later.",
                    reference=message
                )
            except:
                pass
            
    def _should_use_sequential_thinking(self, query):
        """
        Determine if a query would benefit from sequential thinking
        
        Args:
            query: The user's question/query
            
        Returns:
            bool: True if sequential thinking should be used
        """
        # Check if sequential thinking is mentioned
        if re.search(r'sequential\s+thinking|step\s+by\s+step|breakdown|break\s+down', query.lower()):
            return True
            
        # Check for complexity indicators
        complexity_indicators = [
            # Question complexity
            r'why', r'how', r'explain', r'analyze', r'compare', r'contrast', 
            r'pros\s+and\s+cons', r'advantages', r'disadvantages',
            
            # Topic complexity 
            r'complex', r'complicated', r'difficult', r'advanced',
            r'in\s+depth', r'detailed', r'comprehensive', r'thorough',
            
            # Multi-step problems
            r'steps?', r'process', r'procedure', r'method', r'approach',
            r'algorithm', r'formula', r'equation', r'calculate', r'solve',
            
            # Abstract concepts
            r'concept', r'theory', r'principle', r'philosophy', r'framework',
            
            # Logic and reasoning
            r'logic', r'reasoning', r'argument', r'fallacy', r'critique',
            r'evaluate', r'assess', r'judge', r'review',
            
            # Historical/political topics
            r'history', r'politics', r'war', r'conflict', r'revolution',
            r'movement', r'era', r'period', r'century', r'decade',
            
            # Scientific topics
            r'science', r'physics', r'chemistry', r'biology', r'mathematics',
            r'psychology', r'sociology', r'economics', r'research'
        ]
        
        # Check if any complexity indicators are present
        if any(re.search(pattern, query.lower()) for pattern in complexity_indicators):
            # Only use sequential for longer queries (likely more complex questions)
            if len(query.split()) >= 8:
                return True
                
        # Check for question marks and length
        if query.count('?') >= 2 or len(query.split()) >= 15:
            return True
            
        # Default to standard response
        return False

    async def handle_image_generation(self, message, prompt):
        """Handle image generation intent"""
        async with message.channel.typing():
            # Send a processing message
            processing_message = await message.channel.send(f"Generating an image of: **{prompt}**...")
            
            try:
                # Extract options from prompt if any
                options = {}
                # Simplified option extraction - could be more sophisticated
                if "realistic" in prompt.lower():
                    options["model"] = "realistic-vision-v4"
                elif "anime" in prompt.lower() or "manga" in prompt.lower():
                    options["model"] = "anime-vision"
                else:
                    options["model"] = "dreamshaper-8"
                
                # Generate the image
                options["enhance"] = True  # Always enhance prompts
                success, image_data, metadata = await self.image_generator.generate_image(
                    prompt=prompt,
                    **options
                )
                
                if success:
                    # Create a file from the BytesIO object
                    discord_file = discord.File(fp=image_data, filename="generated_image.png")
                    
                    # Create embed for better presentation
                    embed = discord.Embed(
                        title="Generated Image",
                        description=f"**Prompt**: {prompt}",
                        color=discord.Color.blue()
                    )
                    embed.set_image(url="attachment://generated_image.png")
                    
                    # Add metadata if available
                    if metadata:
                        embed.add_field(
                            name="Details", 
                            value=f"Model: {metadata.get('model', 'Unknown')}\nGeneration Time: {metadata.get('generation_time', 0):.2f}s",
                            inline=False
                        )
                    
                    # Send the image
                    await message.channel.send(file=discord_file, embed=embed)
                    
                    # Delete the processing message
                    await processing_message.delete()
                else:
                    # If generation failed, send error message
                    await processing_message.edit(content=f"❌ Failed to generate image: {image_data}")
            
            except Exception as e:
                print(f"Error generating image: {e}")
                await processing_message.edit(content=f"❌ Error generating image: {str(e)}")
    
    async def handle_image_analysis(self, message, attachment):
        """Handle image analysis intent"""
        async with message.channel.typing():
            # Send a processing message
            processing_message = await message.channel.send("Analyzing the image...")
            
            try:
                # Analyze the image
                analysis = await self.image_processor.analyze_image(attachment.url)
                
                # Create embed for better presentation
                embed = discord.Embed(
                    title="Image Analysis",
                    description=analysis,
                    color=discord.Color.blue()
                )
                embed.set_thumbnail(url=attachment.url)
                
                # Send the analysis
                await message.channel.send(embed=embed)
                
                # Delete the processing message
                await processing_message.delete()
            
            except Exception as e:
                print(f"Error analyzing image: {e}")
                await processing_message.edit(content=f"❌ Error analyzing image: {str(e)}")
    
    async def handle_image_ocr(self, message, attachment):
        """Handle OCR (text extraction) intent"""
        async with message.channel.typing():
            # Send a processing message
            processing_message = await message.channel.send("Extracting text from the image...")
            
            try:
                # Extract text from the image
                extracted_text = await self.image_processor.extract_text_from_image(attachment.url)
                
                # Create embed for better presentation
                embed = discord.Embed(
                    title="Text Extracted from Image",
                    description=extracted_text,
                    color=discord.Color.blue()
                )
                embed.set_thumbnail(url=attachment.url)
                
                # Send the extracted text
                await message.channel.send(embed=embed)
                
                # Delete the processing message
                await processing_message.delete()
            
            except Exception as e:
                print(f"Error extracting text from image: {e}")
                await processing_message.edit(content=f"❌ Error extracting text: {str(e)}")
    
    async def handle_voice_transcription(self, message, attachment):
        """Handle voice transcription intent"""
        async with message.channel.typing():
            # Send a processing message
            processing_message = await message.channel.send("Transcribing the voice message...")
            
            try:
                # Transcribe the voice message
                transcript = await self.image_processor.transcribe_audio(attachment.url)
                
                # Create embed for better presentation
                embed = discord.Embed(
                    title="Voice Message Transcription",
                    description=transcript,
                    color=discord.Color.blue()
                )
                
                # Add the voice message link
                embed.add_field(
                    name="Original Voice Message", 
                    value=f"[Voice Message]({attachment.url})"
                )
                
                # Add footer with attribution
                embed.set_footer(text=f"Transcribed for {message.author.display_name}")
                
                # Send the transcription
                await message.channel.send(embed=embed)
                
                # Delete the processing message
                await processing_message.delete()
            
            except Exception as e:
                print(f"Error transcribing voice message: {e}")
                await processing_message.edit(content=f"❌ Error transcribing voice message: {str(e)}")
    
    async def handle_sentiment_analysis_text(self, message, text):
        """Handle sentiment analysis on provided text"""
        async with message.channel.typing():
            # Send a processing message
            processing_message = await message.channel.send(f"Analyzing sentiment of the text...")
            
            try:
                # Analyze the sentiment
                analysis_result = await self.sentiment_analyzer.analyze_sentiment(text)
                
                # Format the results for display
                formatted_result = await self.sentiment_analyzer.format_sentiment_analysis(analysis_result)
                
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
                truncated_text = text[:max_text_length] + ("..." if len(text) > max_text_length else "")
                embed.add_field(
                    name="Analyzed Text",
                    value=f"```{truncated_text}```",
                    inline=False
                )
                
                # Set author information
                embed.set_author(
                    name=f"Requested by {message.author.display_name}",
                    icon_url=message.author.display_avatar.url
                )
                
                # Set footer
                embed.set_footer(text="AI Sentiment Analysis")
                
                # Send the analysis
                await message.channel.send(embed=embed)
                
                # Delete the processing message
                await processing_message.delete()
            
            except Exception as e:
                print(f"Error analyzing sentiment: {e}")
                await processing_message.edit(content=f"❌ Error analyzing sentiment: {str(e)}")
    
    async def handle_sentiment_analysis_reference(self, message, referenced_message):
        """Handle sentiment analysis on a referenced message"""
        if not referenced_message.content:
            await message.channel.send("The referenced message doesn't contain any text to analyze.")
            return
            
        await self.handle_sentiment_analysis_text(message, referenced_message.content)
    
    async def handle_web_search(self, message, query):
        """Handle web search intent"""
        try:
            # Show typing indicator to indicate processing
            async with message.channel.typing():
                # Use the agent to run a web search
                # This likely needs to be updated based on your specific agent implementation
                result = await run_agent(
                    query=query
                )
                
                # Send the response
                await message.reply(result)
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in handle_web_search: {error_traceback}")
            await message.reply(f"I encountered an error while searching: {str(e)[:1500]}")
            
    async def handle_sequential_thinking(self, message, problem):
        """
        Handle sequential thinking for a user's query
        """
        try:
            # Show thinking indicator
            initial_message = None
            if SHOW_THINKING_INDICATOR:
                try:
                    initial_message = await message.channel.send("Thinking... <a:thinking:1174880300425388042>")
                except:
                    try:
                        initial_message = await message.channel.send("Thinking...")
                    except:
                        pass
            
            # Setup context for better reasoning
            user_id = str(message.author.id)
            channel_id = str(message.channel.id)
            thread_id = channel_id if isinstance(message.channel, discord.Thread) else None
            
            # Get context
            context_manager = get_context_manager()
            context = {}
            if context_manager:
                try:
                    context = context_manager.build_context(
                        user_id=user_id,
                        channel_id=channel_id,
                        thread_id=thread_id
                    )
                except:
                    # If build_context fails, try alternative methods
                    try:
                        context = {
                            "history": context_manager.get_conversation_history(
                                user_id=user_id,
                                channel_id=channel_id,
                                thread_id=thread_id,
                                limit=10
                            )
                        }
                    except:
                        # Fallback to empty context
                        context = {}
            
            # Determine the optimal thinking style
            thinking_style = self._determine_thinking_style(problem)
            
            # Get sequential thinking instance
            if not hasattr(self, 'sequential_thinking') or not self.sequential_thinking:
                try:
                    from bot_utilities.sequential_thinking import create_sequential_thinking
                    self.sequential_thinking = create_sequential_thinking()
                    
                    # Attempt to set the provider
                    try:
                        from bot_utilities.ai_utils import get_ai_provider
                        provider = await get_ai_provider()
                        await self.sequential_thinking.set_llm_provider(provider)
                    except:
                        pass  # Continue without a provider, will use fallback
                except Exception as e:
                    print(f"Error initializing sequential thinking: {e}")
                    await message.channel.send("I'm sorry, I couldn't initialize my thinking process.", reference=message)
                    return
            
            # Run sequential thinking with the appropriate style
            success, response = await self.sequential_thinking.run(
                problem=problem,
                context=context,
                prompt_style=thinking_style,
                num_thoughts=5,
                temperature=0.7
            )
            
            if not success:
                await message.channel.send(
                    "I couldn't complete the sequential thinking process for your query.",
                    reference=message
                )
                return
            
            # Create metadata for the response
            metadata = {
                "method": thinking_style,
                "method_name": self._get_thinking_style_name(thinking_style),
                "method_emoji": self._get_thinking_style_emoji(thinking_style)
            }
            
            # Package response with metadata
            full_response = (response, metadata)
            
            # Send the response using our unified method
            await self.send_response(
                message=message,
                response=full_response,
                is_sequential=True,
                problem=problem,
                initial_message=initial_message
            )
            
        except Exception as e:
            print(f"Error in handle_sequential_thinking: {e}")
            await message.channel.send(
                "I encountered an error while processing your message. Please try again later.",
                reference=message
            )

    def _determine_thinking_style(self, query):
        """
        Determine the best thinking style for a query
        
        Args:
            query: The user query
            
        Returns:
            str: The thinking style to use
        """
        query_lower = query.lower()
        
        # Direct request for specific styles
        if any(pattern in query_lower for pattern in ["verify", "verification", "check fact", "confirm"]):
            return "cov"  # Chain-of-verification
        
        if any(pattern in query_lower for pattern in ["multi angle", "different perspective", "compare approach", "graph of thought"]):
            return "got"  # Graph-of-thought
        
        if any(pattern in query_lower for pattern in ["list", "bullet point", "point by point", "enumerat"]):
            return "list"  # List-based thinking
        
        # Complex topics that need fact verification
        verification_topics = ["history", "politic", "science", "research", "fact", "statistic", "data"]
        if any(topic in query_lower for topic in verification_topics) and len(query.split()) > 10:
            return "cov"
        
        # Multi-perspective topics
        perspective_topics = ["debate", "argument", "controversy", "opinion", "perspective", "viewpoint"]
        if any(topic in query_lower for topic in perspective_topics):
            return "got"
        
        # Default to standard chain-of-thought for most complex topics
        return "cot"

    def _get_thinking_style_name(self, style):
        """Get a human-readable name for a thinking style"""
        style_names = {
            "sequential": "Sequential Thinking",
            "cot": "Chain of Thought",
            "got": "Graph of Thought",
            "cov": "Chain of Verification",
            "list": "Structured List Thinking"
        }
        return style_names.get(style, "Sequential Thinking")

    def _get_thinking_style_emoji(self, style):
        """Get an emoji representing a thinking style"""
        style_emojis = {
            "sequential": "🔄",
            "cot": "⛓️",
            "got": "🕸️",
            "cov": "✅",
            "list": "📋"
        }
        return style_emojis.get(style, "🤔")

    async def handle_intent(self, message, intent_data):
        """
        Handle intents detected from user messages
        
        Args:
            message: The Discord message
            intent_data: Dictionary of intent information
        """
        intent = intent_data.get("intent", "chat")
        
        # Handle each intent type
        if intent == "generate_image":
            await self.handle_image_generation(message, intent_data.get("prompt", ""))
        elif intent == "analyze_image":
            await self.handle_image_analysis(message, intent_data.get("attachment"))
        elif intent == "ocr_image":
            await self.handle_image_ocr(message, intent_data.get("attachment"))
        elif intent == "transcribe_voice":
            await self.handle_voice_transcription(message, intent_data.get("attachment"))
        elif intent == "analyze_sentiment":
            text = intent_data.get("text")
            referenced_message = intent_data.get("referenced_message")
            if text:
                await self.handle_sentiment_analysis_text(message, text)
            elif referenced_message:
                await self.handle_sentiment_analysis_reference(message, referenced_message)
        elif intent == "sequential_thinking":
            await self.handle_sequential_thinking(message, intent_data.get("problem", ""))
        elif intent == "web_search":
            await self.handle_web_search(message, intent_data.get("query", ""))
        elif intent == "mcp_agent":
            await self.handle_mcp_agent(message, intent_data.get("query", ""))
        else:
            # Default to reasoning router for any chat intent
            clean_content = intent_data.get("message", "")
            username = self.get_custom_username(message)
            user_id = str(message.author.id)
            channel_id = str(message.channel.id)
            await self.handle_message_with_reasoning_router(message, clean_content, username, user_id, channel_id)

    async def handle_mcp_agent(self, message, query):
        """Handle MCP agent intent"""
        try:
            # Show typing indicator to indicate processing
            async with message.channel.typing():
                # Send an initial message with a spinning indicator
                initial_message = await message.reply(f"🔄 Working on: `{query}`")
                
                # Track start time
                start_time = time.time()
                
                # Get MCP Manager from cogs
                mcp_cog = self.bot.get_cog("MCPAgentCog")
                if mcp_cog and hasattr(mcp_cog, "mcp_manager"):
                    # Use the MCP manager to run the agent
                    response = await mcp_cog.mcp_manager.run_simple_mcp_agent(query)
                else:
                    # Fallback if MCP cog is not available
                    # Use standard AI response
                    user_id = str(message.author.id)
                    channel_id = message.channel.id
                    key = f"{user_id}-{channel_id}"
                    
                    # Initialize or get existing history
                    message_history[key] = message_history.get(key, [])
                    
                    # Add problem to history
                    message_history[key].append({"role": "user", "content": query})
                    
                    # Get response with agent capabilities
                    agent_instruction = f"{self.instructions['Agent']} You have access to tools and can search for information if needed."
                    response = await self.generate_response(agent_instruction, message_history[key])
                    message_history[key].append({"role": "assistant", "content": response})
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Format the response
                chunks = chunk_message(response)
                if len(chunks) > 1:
                    await initial_message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await message.channel.send(chunk)
                else:
                    await initial_message.edit(content=response)
                
                # Log activity
                if hasattr(self, "activity_monitor"):
                    asyncio.create_task(self.activity_monitor.log_command_usage(
                        user_id=str(message.author.id),
                        command_name="mcp_agent_text",
                        guild_id=str(message.guild.id) if message.guild else "DM",
                        success=True
                    ))
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in handle_mcp_agent: {error_traceback}")
            await message.reply(f"💥 Error: I encountered a problem with the agent. Please try again later.\n```{str(e)[:1500]}```")

    async def generate_response(self, instructions, history, stream=False):
        return await generate_response(instructions=instructions, history=history, stream=stream)
    
    async def process_streaming_response(self, initial_message, response_stream):
        """Process a streaming response, updating the message periodically"""
        
        accumulated_response = ""
        buffer = ""
        last_update_time = time.time()
        
        # Collect response chunks
        try:
            async for chunk in response_stream:
                buffer += chunk
                accumulated_response += chunk
                
                # Update message periodically to avoid rate limits
                current_time = time.time()
                if current_time - last_update_time >= self.STREAM_UPDATE_INTERVAL:
                    try:
                        # Format the response for display
                        if len(buffer) > 0:
                            # Keep it under Discord's character limit
                            content = buffer[:1990] if len(buffer) > 1990 else buffer
                            await initial_message.edit(content=content)
                            last_update_time = current_time
                    except Exception as e:
                        print(f"Error updating message: {e}")
        except Exception as e:
            print(f"Error processing stream: {e}")
            # If we have no accumulated response at all, use a fallback message
            if not accumulated_response:
                accumulated_response = "I encountered an error while generating a response. Please try again."
                try:
                    await initial_message.edit(content=accumulated_response)
                except:
                    pass
        
        # Ensure we do a final update with the complete response
        try:
            # Check if we have any accumulated response
            if not accumulated_response:
                accumulated_response = "I'm experiencing technical difficulties. Please try again later."
                await initial_message.edit(content=accumulated_response)
                return accumulated_response
                
            # Need to format final response properly
            user_id = str(initial_message.reference.resolved.author.id) if initial_message.reference and initial_message.reference.resolved else None
            user_prefs = await UserPreferences.get_user_preferences(user_id) if user_id else {}
            use_embeds = user_prefs.get('use_embeds', False)
            
            # Process mentions in the response before formatting
            if initial_message.reference and initial_message.reference.resolved:
                formatted_response = find_and_format_user_mentions(initial_message.reference.resolved, accumulated_response, self.bot)
            else:
                formatted_response = accumulated_response
                
            # Format the response for Discord
            content, embed, chunks = format_response_for_discord(
                formatted_response,
                use_embeds,
                author_name=initial_message.reference.resolved.author.display_name if initial_message.reference and initial_message.reference.resolved else "User",
                avatar_url=initial_message.reference.resolved.author.avatar.url if initial_message.reference and initial_message.reference.resolved and initial_message.reference.resolved.author.avatar else None
            )
            
            # Update the message with the formatted response
            if embed:
                await initial_message.edit(content=content, embed=embed)
            else:
                # If the response is in chunks, handle them appropriately
                if chunks:
                    channel = initial_message.channel
                    
                    # Update the initial message with the first chunk
                    await initial_message.edit(content=chunks[0])
                    
                    # Send additional chunks as separate messages
                    for chunk in chunks[1:]:
                        await channel.send(content=chunk)
                else:
                    # Single response within limits
                    await initial_message.edit(content=content)
            
            return accumulated_response
        
        except Exception as e:
            print(f"Error finalizing streamed response: {e}")
            
            # Fallback to simple chunking if anything fails
            try:
                # If response is still too long, chunk it
                if len(accumulated_response) > 2000:
                    chunks = chunk_message(accumulated_response)
                    
                    # Update initial message with first chunk
                    await initial_message.edit(content=chunks[0])
                    
                    # Send additional chunks
                    for chunk in chunks[1:]:
                        await initial_message.channel.send(content=chunk)
                else:
                    # Just update with what we have (truncated if needed)
                    await initial_message.edit(content=accumulated_response[:2000])
            except:
                # Last resort fallback
                try:
                    await initial_message.edit(content="I was unable to complete my response. Please try again.")
                except:
                    pass
            
            return accumulated_response

    async def send_response(self, message, response, message_reference=None, is_sequential=False, problem=None, initial_message=None):
        """
        Send a response to Discord with proper formatting while handling tuple responses
        
        Args:
            message: The Discord message to respond to
            response: The response to send (string or tuple)
            message_reference: Optional message reference for threading
            is_sequential: If True, use sequential thinking formatting
            problem: Original problem (for sequential thinking context storage)
            initial_message: Optional message to delete (like "thinking..." message)
        """
        try:
            # Extract text from tuple responses and remove metadata
            response_text = response
            metadata = {}
            
            # Handle tuple responses (response_text, metadata)
            if isinstance(response, tuple) and len(response) >= 1:
                response_text = response[0]  # Extract just the response text
                if len(response) >= 2 and isinstance(response[1], dict):
                    metadata = response[1]   # Extract metadata for context storage
            # Check if the response might be a string representation of a tuple
            elif isinstance(response, str) and response.startswith("(") and ("method" in response or "{" in response):
                try:
                    # Try to extract the text part using regex
                    import re
                    match = re.search(r'^\s*\(\s*[\'"](.+?)[\'"]\s*,', response, re.DOTALL)
                    if match:
                        response_text = match.group(1)
                        # Try to extract metadata too if possible
                        metadata_match = re.search(r',\s*(\{.+\})\s*\)$', response, re.DOTALL)
                        if metadata_match:
                            try:
                                # Safely evaluate the metadata dictionary string
                                import ast
                                metadata_str = metadata_match.group(1).replace("'", '"')
                                metadata = ast.literal_eval(metadata_str)
                            except:
                                # If metadata parsing fails, continue with empty metadata
                                metadata = {}
                except:
                    # If regex fails, keep the original response
                    response_text = response
            
            # Ensure we're working with a string
            if not isinstance(response_text, str):
                try:
                    response_text = str(response_text)
                except:
                    response_text = "Error formatting response"
            
            # Clean response text from any tuple or metadata artifacts
            # Remove tuple-like patterns
            if "(" in response_text and ")" in response_text:
                try:
                    match = re.search(r"^\s*\(\s*['\"](.+?)['\"]", response_text, re.DOTALL)
                    if match:
                        response_text = match.group(1)
                except:
                    pass
            
            # Remove metadata patterns
            response_text = re.sub(r"\{'method':.*?\}", "", response_text).strip()
            response_text = re.sub(r"\('.*?', \{'method':.*?\}\)", "", response_text).strip()
            
            # Remove conclusion markers that might be in sequential thinking responses
            response_text = response_text.replace("Conclusion: ", "")
            
            # Remove any remaining formatting artifacts
            response_text = response_text.strip('"\'() ')
            
            # Apply Discord markdown formatting if enabled
            if USE_MARKDOWN_FORMATTING:
                # For sequential thinking, enhance the formatting with Discord markdown
                if is_sequential and "**Step" in response_text:
                    # Format sequential thinking steps with better Discord markdown
                    response_text = re.sub(r'\*\*Step (\d+)\*\*:', r'__**Step \1**__:', response_text)
                    response_text = re.sub(r'\*\*Thought (\d+)\*\*:', r'__**Thought \1**__:', response_text)
                    response_text = re.sub(r'\*\*Conclusion\*\*:', r'__**Conclusion**__:', response_text)
                    response_text = re.sub(r'\*\*Verification\*\*:', r'__**Verification**__:', response_text)
                
                # Highlight code blocks with proper Discord markdown
                response_text = re.sub(r'```(\w+)?', r'```\1', response_text)  # Ensure code blocks have language
                
                # Ensure blockquotes are properly formatted
                response_text = re.sub(r'^>', r'> ', response_text, flags=re.MULTILINE)
            
            # Delete initial message if provided (like "thinking..." message)
            if initial_message:
                try:
                    await initial_message.delete()
                except:
                    pass
            
            # Choose formatting approach based on response type
            if is_sequential:
                # Format as sequential thinking (preserve steps/thoughts markers)
                response_chunks = []
                current_chunk = ""
                
                for line in response_text.split("\n"):
                    # If adding this line would exceed Discord's message length limit, start a new chunk
                    if len(current_chunk) + len(line) + 1 > 1900:  # Leave room for formatting
                        response_chunks.append(current_chunk)
                        current_chunk = line
                    else:
                        current_chunk += "\n" + line if current_chunk else line
                    
                # Add the last chunk
                if current_chunk:
                    response_chunks.append(current_chunk)
            else:
                # Format for regular responses using Discord's formatting
                content, embed, response_chunks = format_response_for_discord(
                    response_text,
                    use_embeds=USE_EMBEDS,
                    author_name=self.bot.user.display_name,
                    avatar_url=self.bot.user.avatar.url if self.bot.user.avatar else None
                )
            
            # Get the method emoji (if available) for visual indicator
            method_emoji = metadata.get("method_emoji", "🤖") if isinstance(metadata, dict) else "🤖"
            
            # Send the first chunk with the message reference for reply threading
            if response_chunks:
                # Add emoji prefix to first message if available and not in sequential mode
                first_chunk = response_chunks[0]
                if not is_sequential and method_emoji and not first_chunk.startswith(method_emoji):
                    first_chunk = f"{method_emoji} {first_chunk}"
                    
                await message.channel.send(
                    first_chunk,
                    reference=message_reference or message
                )
                
                # Send any remaining chunks
                for chunk in response_chunks[1:]:
                    await message.channel.send(chunk)
            
            # Store the response in context manager for conversation continuity
            context_manager = get_context_manager()
            if context_manager:
                # Get metadata for storage if available
                method = metadata.get("method", "sequential" if is_sequential else "standard")
                method_name = metadata.get("method_name", "Sequential Thinking" if is_sequential else "Standard Response")
                is_fallback = metadata.get("is_fallback", False)
                
                # Calculate thread ID (threads have the same ID as their channel)
                thread_id = str(message.channel.id) if isinstance(message.channel, discord.Thread) else None
                
                # Store in conversation history
                context_manager.add_message(
                    user_id=str(message.author.id),
                    channel_id=str(message.channel.id),
                    thread_id=thread_id,
                    message={
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": time.time(),
                        "type": method,
                        "method": method_name,
                        "is_fallback": is_fallback,
                        "in_response_to": problem if problem else ""
                    }
                )
                
                # Store thinking process if applicable
                if is_sequential and problem:
                    context_manager.add_thinking_process(
                        user_id=str(message.author.id),
                        channel_id=str(message.channel.id),
                        thread_id=thread_id,
                        thinking={
                            "problem": problem,
                            "solution": response_text,
                            "prompt_style": method,
                            "method_name": method_name,
                            "is_complex": True,
                            "timestamp": time.time(),
                            "success": True,
                            "is_fallback": is_fallback
                        }
                    )
        except Exception as e:
            print(f"Error sending response: {e}")
            try:
                await message.channel.send(
                    "I'm having trouble sending my response. Please try again later.",
                    reference=message_reference or message
                )
            except:
                pass

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        Handle incoming messages and generate AI responses when appropriate
        """
        # Ignore messages from self to prevent loops
        if message.author == self.bot.user:
            return

        # Check if the message is a DM
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # If DMs are not allowed and this is a DM, ignore
        if not allow_dm and is_dm:
            return
            
        # Get clean message content without bot mention
        clean_content = await self.clean_message_content(message)
        
        # For a Guild (server) message, check if the bot is mentioned or in an active channel
        if not is_dm:
            # If no content after removing mention, ignore
            if not clean_content.strip():
                return
                
            # Check if the bot was mentioned or if the channel is active
            was_mentioned = self.bot.user.mentioned_in(message)
            is_active = is_active_channel(message.channel, message.guild.id)
            is_smart_mention = smart_mention(clean_content, self.bot)
            
            # Handle thread messages with a reply
            in_thread_with_reply = isinstance(message.channel, discord.Thread) and message.type == discord.MessageType.reply
            
            # If not mentioned and not in an active channel, ignore
            if not (was_mentioned or is_active or is_smart_mention or in_thread_with_reply):
                return
        
        try:
            # Get the user's custom username or display name
            username = self.get_custom_username(message)
            user_id = str(message.author.id)
            channel_id = str(message.channel.id)
            
            # Process message with reasoning router
            await self.handle_message_with_reasoning_router(message, clean_content, username, user_id, channel_id)
        except Exception as e:
            logger.error(f"Error in on_message: {e}")
            traceback.print_exc()

async def setup(bot):
    await bot.add_cog(OnMessage(bot))