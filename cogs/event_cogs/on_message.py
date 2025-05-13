import discord
from discord.ext import commands
import asyncio
import json
import os
import re
import time
import traceback
import random
import aiohttp
import io
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import sys

from bot_utilities.response_utils import split_response
from bot_utilities.ai_utils import generate_response, get_crypto_price, text_to_speech, get_bot_names_and_triggers
from bot_utilities.memory_utils import UserPreferences, process_conversation_history, get_enhanced_instructions
from bot_utilities.news_utils import get_news_context
from bot_utilities.formatting_utils import format_response_for_discord, find_and_format_user_mentions, chunk_message
from bot_utilities.config_loader import config, load_active_channels
from bot_utilities.multimodal_utils import ImageProcessor, ImageGenerator
from bot_utilities.sentiment_utils import SentimentAnalyzer
from bot_utilities.agent_utils import run_agent
from ..common import allow_dm, smart_mention, MAX_HISTORY, instructions, message_history
from bot_utilities.mcp_utils import MCPToolsManager
from bot_utilities.sequential_thinking import create_sequential_thinking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('on_message')

class OnMessage(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_channels = load_active_channels
        self.instructions = instructions
        self.thread_contexts = {}  # Store parent message context for threads
        self.STREAM_UPDATE_INTERVAL = 0.5  # Update streaming messages every 0.5 seconds
        self.replied_messages = {}  # Store message IDs for reference
        
        # Initialize processors for various capabilities
        self.image_processor = ImageProcessor()
        self.image_generator = ImageGenerator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.mcp_manager = MCPToolsManager()
        # Initialize sequential thinking
        self.sequential_thinking = create_sequential_thinking(llm_provider=None)

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
        """Process messages sent by users"""
        
        # Skip messages from the bot itself
        if message.author == self.bot.user:
            return
        
        # Skip messages from other bots (unless debug_mode is enabled)
        if message.author.bot and not config.get('DEBUG_MODE', False):
            return
            
        # Check if bot should respond in this channel
        channel_id = str(message.channel.id)
        active_channels = self.active_channels()  # Call the function to get active channels
        is_active_channel = channel_id in active_channels
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Only proceed if in an active channel, DM, or bot is mentioned
        is_bot_mentioned = self.bot.user in message.mentions
        was_replied_to = message.reference and message.reference.resolved and message.reference.resolved.author == self.bot.user
        
        # Get triggers from bot_utilities
        bot_names, base_trigger_words = get_bot_names_and_triggers()
        
        # Process special placeholder triggers (%BOT_USERNAME%, %BOT_NICKNAME%)
        trigger_words = []
        for trigger in base_trigger_words:
            if "%BOT_USERNAME%" in trigger:
                # Replace with bot's username
                trigger_words.append(trigger.replace("%BOT_USERNAME%", self.bot.user.name.lower()))
            elif "%BOT_NICKNAME%" in trigger and hasattr(message, "guild") and message.guild:
                # Replace with bot's nickname in this guild if it has one
                member = message.guild.get_member(self.bot.user.id)
                if member and member.nick:
                    trigger_words.append(trigger.replace("%BOT_NICKNAME%", member.nick.lower()))
            else:
                trigger_words.append(trigger)
        
        # Add the bot's actual Discord username as a trigger
        if self.bot.user.name.lower() not in trigger_words:
            trigger_words.append(self.bot.user.name.lower())
        
        # Check if message content contains a bot name/trigger
        message_content = message.content.lower()
        contains_trigger = any(trigger.lower() in message_content for trigger in trigger_words)
        
        # Smart mention feature
        smart_mention_active = smart_mention and not message.author.bot
        recently_active = False  # This would be determined by channel activity tracking
        
        # Determine if bot should respond
        should_respond = (
            is_active_channel or 
            is_dm or 
            is_bot_mentioned or 
            was_replied_to or
            contains_trigger or
            (smart_mention_active and recently_active)
        )
        
        if not should_respond:
            return
            
        # Get clean message content for processing
        clean_content = message.content
        
        # Remove bot mentions from the message
        for mention in message.mentions:
            if mention == self.bot.user:
                # Replace mentions with an empty string
                clean_content = clean_content.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
        
        # Remove bot trigger words/names from the beginning of messages
        for trigger in trigger_words:
            if clean_content.lower().startswith(trigger.lower()):
                clean_content = clean_content[len(trigger):].strip()
                
        # Remove leading/trailing whitespace
        clean_content = clean_content.strip()
        
        # If message is empty after removing mentions, use a default prompt
        if not clean_content:
            clean_content = "Hello!"
        
        # Detect intent from message content
        intent_data = await self.detect_intent(clean_content, message)
        
        # Log the detected intent for debugging
        logger.info(f"Detected intent: {intent_data['intent']} for message: {clean_content[:50]}...")
        
        # Handle the detected intent
        try:
            if intent_data['intent'] == 'generate_image':
                await self.handle_image_generation(message, intent_data['prompt'])
                return
                
            elif intent_data['intent'] == 'analyze_image':
                await self.handle_image_analysis(message, intent_data['attachment'])
                return
                
            elif intent_data['intent'] == 'ocr_image':
                await self.handle_image_ocr(message, intent_data['attachment'])
                return
                
            elif intent_data['intent'] == 'transcribe_voice':
                await self.handle_voice_transcription(message, intent_data['attachment'])
                return
                
            elif intent_data['intent'] == 'analyze_sentiment':
                if 'text' in intent_data:
                    await self.handle_sentiment_analysis_text(message, intent_data['text'])
                else:
                    await self.handle_sentiment_analysis_reference(message, intent_data['referenced_message'])
                return
                
            elif intent_data['intent'] == 'web_search':
                await self.handle_web_search(message, intent_data['query'])
                return
                
            elif intent_data['intent'] == 'sequential_thinking':
                await self.handle_sequential_thinking(message, intent_data['problem'])
                return
                
            elif intent_data['intent'] == 'mcp_agent':
                await self.handle_mcp_agent(message, intent_data['query'])
                return
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error handling intent {intent_data['intent']}: {error_traceback}")
            await message.channel.send(f"I encountered an error while processing your request: {str(e)[:1500]}")
            return
                
        # Check for sequential thinking based on complexity for regular chat messages
        # This happens only if none of the specific intents were detected
        try:
            from bot_utilities.ai_utils import should_use_sequential_thinking
            use_sequential, complexity_score, reasoning = await should_use_sequential_thinking(clean_content)
            
            if use_sequential:
                # Log why we're using sequential thinking
                logger.info(f"Using sequential thinking for message: {reasoning} (score: {complexity_score:.2f})")
                await self.handle_sequential_thinking(message, clean_content)
                return
        except Exception as e:
            logger.error(f"Error checking for sequential thinking complexity: {e}")
        
        # Default path: Regular AI chat interaction
        try:
            # Check if we should stream the response
            stream = config.get('STREAM_RESPONSES', False)
            
            # Show typing indicator while generating response
            async with message.channel.typing():
                # Call the AI model to generate a response
                response = await self.generate_response(
                    instructions=self.instructions,
                    history=await process_conversation_history(message, self.bot),
                    stream=stream  # Whether to stream the response
                )
                
                # If streaming is enabled, handle the streaming response
                if stream:
                    await self.process_streaming_response(message, response)
                else:
                    # Check if voice response is requested
                    use_voice = any(phrase in clean_content.lower() for phrase in ["speak to me", "talk to me", "use voice", "voice message"])
                    
                    # Send the response
                    await self.send_response(message, response, use_voice=use_voice)
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in process_message: {error_traceback}")
            await message.channel.send(f"I encountered an error while processing your message: {str(e)[:1500]}")

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
        """Handle sequential thinking intent"""
        # Create a tracking variable for the initial message
        initial_message = None
        message_reference = None  # Store the initial message reference for tracking
        
        try:
            # Show typing indicator to indicate processing
            async with message.channel.typing():
                # Send an initial message with a spinning indicator
                try:
                    initial_message = await message.reply(f"🔄 Starting sequential thinking for: `{problem}`")
                    message_reference = initial_message.id  # Store the ID for reference
                except Exception as e:
                    print(f"Error sending initial message: {e}")
                    # Try a direct channel send if reply fails
                    try:
                        initial_message = await message.channel.send(f"🔄 Starting sequential thinking for: `{problem}`")
                        message_reference = initial_message.id  # Store the ID
                    except Exception as e2:
                        print(f"Critical error sending message: {e2}")
                
                # Detect if this is a complex topic that needs special handling
                is_complex_topic = False
                use_tools = False
                
                # Check for keywords that suggest complex topics
                complex_topic_keywords = [
                    # General problem-solving keywords
                    "analyze", "investigate", "research", "examine", "explore", "explain", "solve", 
                    "strategize", "compare", "contrast", "evaluate", "assess", "critique", "review",
                    
                    # Technical and analytical
                    "algorithm", "technical", "architecture", "framework", "system", "design", "process",
                    "mathematical", "computation", "physics", "engineering", "chemistry", "biology",
                    
                    # Business and organizational
                    "business", "strategy", "management", "organization", "planning", "operations",
                    "marketing", "finance", "investment", "product", "service", "customer", "market",
                    
                    # Creative and conceptual
                    "conceptualize", "create", "design", "develop", "innovate", "imagine", "brainstorm",
                    "synthesize", "generate", "formulate", "construct", "compose", "craft", 
                    
                    # Social and human topics
                    "social", "cultural", "psychological", "behavioral", "ethical", "moral", "philosophical",
                    "educational", "learning", "communication", "community", "collaboration",
                    
                    # Geopolitical/historical (retained from original)
                    "conflict", "war", "dispute", "politics", "history", "crisis", "relation", "tension", 
                    "peace", "treaty", "agreement", "border", "territory", "military", "diplomatic", 
                    "economic", "global", "international", "regional", "historical",
                    
                    # Decision making
                    "decision", "choice", "tradeoff", "prioritize", "optimize", "maximize", "minimize",
                    "balance", "weigh", "consider", "judge", "determine", "select", "choose",
                    
                    # Time-based
                    "plan", "schedule", "timeline", "roadmap", "forecast", "predict", "project",
                    "future", "trend", "evolution", "development", "progress", "history",
                    
                    # Sequential keywords
                    "step", "process", "procedure", "method", "approach", "technique", "workflow",
                    "sequential", "thinking", "reasoning", "logic", "analysis", "breakdown",
                    
                    # Explicit triggers
                    "sequential thinking", "think step by step", "break down", "complex problem"
                ]
                
                # Detect if this is a complex topic
                is_complex_topic = any(keyword in problem.lower() for keyword in complex_topic_keywords)
                
                # Check for tool integration keywords
                tool_keywords = ["search", "web", "browse", "information", "data", "lookup", 
                                "db", "database", "fetch", "retrieve", "knowledge"]
                
                use_tools = any(keyword in problem.lower() for keyword in tool_keywords)
                
                # Update the message to show appropriate processing
                try:
                    # Safely check if message exists and is accessible
                    if initial_message:
                        try:
                            # Try to access the message - this will fail if it's deleted
                            if use_tools:
                                await initial_message.edit(content=f"🔄 Researching and solving: `{problem}`")
                            elif is_complex_topic:
                                await initial_message.edit(content=f"🔄 Analyzing complex topic: `{problem}`")
                            else:
                                await initial_message.edit(content=f"🔄 Processing: `{problem}`")
                        except discord.errors.NotFound:
                            # Message was deleted or not found
                            print(f"Warning: Initial message not found (ID: {message_reference}), creating a new one")
                            try:
                                initial_message = await message.channel.send(f"🔄 Continuing to solve: `{problem}`")
                                message_reference = initial_message.id  # Update reference
                            except Exception as e:
                                print(f"Error creating replacement message: {e}")
                except Exception as e:
                    print(f"Warning: Could not edit message, continuing with processing: {e}")
                
                # Build context for enhanced reasoning
                context = {
                    "user": {
                        "id": str(message.author.id),
                        "name": message.author.display_name,
                    },
                    "channel": {
                        "name": message.channel.name if hasattr(message.channel, "name") else "DM",
                        "is_dm": isinstance(message.channel, discord.DMChannel),
                        "is_thread": isinstance(message.channel, discord.Thread),
                    },
                    "guild": {
                        "name": message.guild.name if message.guild else "DM",
                        "id": str(message.guild.id) if message.guild else None,
                    }
                }
                
                # Get LLM provider for sequential thinking
                llm_provider = None
                try:
                    from bot_utilities.ai_utils import get_ai_provider
                    llm_provider = await get_ai_provider()
                    # Set the LLM provider to our sequential thinking instance
                    await self.sequential_thinking.set_llm_provider(llm_provider)
                except Exception as e:
                    print(f"Error getting AI provider: {e}")
                    # Will use fallback mechanisms in sequential_thinking.py
                
                # Standard sequential thinking path
                try:
                    # Try updating message to show we're processing
                    try:
                        if initial_message:
                            try:
                                await initial_message.edit(content=f"🔄 Thinking through: `{problem}`")
                            except discord.errors.NotFound:
                                pass
                    except Exception as e:
                        print(f"Error updating message for standard processing: {e}")
                    
                    # Choose the appropriate thinking style based on problem complexity
                    prompt_style = "sequential"  # Default style
                    
                    # For complex topics that might benefit from verification, use chain-of-verification
                    verification_keywords = ["fact", "accuracy", "verify", "confirm", "check", "research",
                                          "evidence", "proof", "source", "citation", "reference",
                                          "correct", "accurate", "precise", "exact", "valid"]
                    
                    # Use Chain-of-Verification for complex topics that need factual accuracy
                    needs_verification = is_complex_topic and any(keyword in problem.lower() for keyword in verification_keywords)
                    
                    # Check for non-linear problem-solving needs (Graph-of-Thought)
                    got_keywords = ["compare", "contrast", "multiple", "alternatives", "pros and cons", 
                                    "different approaches", "various methods", "options", "pathways",
                                    "interconnected", "relationship", "related", "network", "graph",
                                    "multi-faceted", "complex", "trade-offs", "decision tree",
                                    "matrix", "framework", "perspectives", "viewpoints", "angles"]
                    
                    needs_got = is_complex_topic and any(keyword in problem.lower() for keyword in got_keywords)
                    
                    if needs_verification or "verify" in problem.lower() or "verification" in problem.lower():
                        prompt_style = "cov"
                        if initial_message:
                            try:
                                await initial_message.edit(content=f"🔄 Verifying facts while solving: `{problem}`")
                            except discord.errors.NotFound:
                                pass
                    elif needs_got or any(phrase in problem.lower() for phrase in ["graph of thought", "different angles", "multiple perspectives"]):
                        prompt_style = "got"  # Use graph-of-thought for non-linear problems
                        if initial_message:
                            try:
                                await initial_message.edit(content=f"🔄 Exploring multiple thought paths for: `{problem}`")
                            except discord.errors.NotFound:
                                pass
                    elif is_complex_topic:
                        prompt_style = "cot"  # Use chain-of-thought for general complex topics
                    
                    # Use our sequential thinking implementation
                    success, response = await self.sequential_thinking.run(
                        problem=problem,
                        context=context,
                        # Use the chosen prompt style
                        prompt_style=prompt_style,
                        num_thoughts=7 if is_complex_topic else 5,
                        temperature=0.3 if is_complex_topic else 0.2,
                        max_tokens=2500 if is_complex_topic else 2000,
                        timeout=120 if is_complex_topic else 90
                    )
                    
                    if success:
                        # Send the response
                        await self.send_sequential_thinking_response(
                            message, problem, response, initial_message=initial_message,
                            message_reference=message_reference
                        )
                        return
                    else:
                        print(f"Sequential thinking failed: {response}")
                        # We'll fall back to standard AI response below
                except Exception as e:
                    print(f"Error in sequential thinking: {e}")
                    # We'll fall back to standard AI response
                
                # If we get here, both approaches failed, so use standard AI response
                try:
                    # Update message to show we're using fallback approach
                    try:
                        if initial_message:
                            try:
                                await initial_message.edit(content=f"🔄 Using fallback approach for: `{problem}`")
                            except discord.errors.NotFound:
                                pass
                    except Exception as e:
                        print(f"Error updating message for fallback: {e}")
                
                    # Use MCP agent as fallback with sequential thinking prompt
                    system_message = """You are an AI assistant that solves problems using sequential thinking.
                    
                    Approach each problem by:
                    
                    1. Breaking it down into smaller, manageable parts
                    2. Addressing each part in a logical sequence
                    3. Building on previous steps to reach the final solution
                    4. Checking your work at each stage
                    5. Summarizing your approach and final answer
                    
                    Think step-by-step and show your reasoning process clearly. Explicitly state when you're moving from one step to the next.
                    """
                    
                    # Use timeout to prevent hanging
                    try:
                        response = await asyncio.wait_for(
                            self.mcp_manager.run_simple_mcp_agent(
                                query=problem,
                                system_message=system_message
                            ),
                            timeout=60
                        )
                        
                        # Send the response as sequential thinking
                        await self.send_sequential_thinking_response(
                            message, problem, response, initial_message=initial_message,
                            message_reference=message_reference, is_fallback=True
                        )
                        return
                    except asyncio.TimeoutError:
                        print("MCP agent fallback timed out")
                        # Continue to standard AI response
                    except Exception as e:
                        print(f"Error in MCP fallback: {e}")
                        # Continue to standard AI response
                except Exception as e:
                    print(f"Error in fallback approach: {e}")
                
                # Final fallback: standard AI response
                try:
                    # Update message
                    try:
                        if initial_message:
                            try:
                                await initial_message.edit(content=f"🔄 Generating standard response for: `{problem}`")
                            except discord.errors.NotFound:
                                pass
                    except Exception as e:
                        print(f"Error updating message for standard response: {e}")
                    
                    # Generate standard AI response
                    response = await self.generate_response(
                        f"""Solve this problem using sequential thinking: {problem}
                        
                        Show your step-by-step reasoning process in a clear, structured format with numbered steps.
                        """,
                        []
                    )
                    
                    # Send the standard response
                    await self.send_sequential_thinking_response(
                        message, problem, response, initial_message=initial_message,
                        message_reference=message_reference, is_fallback=True
                    )
                except Exception as e:
                    print(f"Error generating standard response: {e}")
                    # We'll reply with a simple error message
                    if initial_message:
                        try:
                            await initial_message.edit(content=f"❌ I encountered an error while thinking through this problem. Please try again with a simpler query.")
                        except:
                            await message.reply("❌ I encountered an error while thinking through this problem. Please try again with a simpler query.")
                    else:
                        await message.reply("❌ I encountered an error while thinking through this problem. Please try again with a simpler query.")
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in handle_sequential_thinking: {error_traceback}")
            await message.reply(f"I encountered an error while thinking through this problem: {str(e)[:1500]}")

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

    async def send_response(self, message, response, use_voice=False, create_thread=False):
        """
        Send the AI response to the user
        
        Args:
            message (discord.Message): The original message
            response (str): The AI response
            use_voice (bool): Whether to send a voice response
            create_thread (bool): Whether to create a thread
            
        Returns:
            discord.Message: The first message sent in response
        """
        first_message = None
        
        try:
            # Get user preferences
            user_prefs = await UserPreferences.get_user_preferences(message.author.id)
            use_embeds = user_prefs.get('use_embeds', False)
            
            # Format response for Discord
            response = find_and_format_user_mentions(message, response, self.bot)
            content, embed, chunks = format_response_for_discord(response, use_embeds, 
                                                        author_name=message.author.display_name, 
                                                        avatar_url=message.author.avatar.url if message.author.avatar else None)
            
            # If the response is in chunks, send each chunk sequentially
            if chunks:
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        # First message
                        if use_voice:
                            # Create voice response in the background
                            voice_task = asyncio.create_task(self.create_voice_response(response))
                            
                        # Send the first message
                        if embed:
                            first_message = await message.channel.send(embed=embed)
                        else:
                            first_message = await message.channel.send(content=chunk)
                            
                        # Create thread if requested and possible
                        if create_thread and first_message and hasattr(message.channel, 'create_thread'):
                            try:
                                thread_name = message.content[:95] + "..." if len(message.content) > 95 else message.content
                                await first_message.create_thread(name=thread_name)
                            except Exception as e:
                                print(f"Error creating thread: {e}")
                        
                        # Send voice file if available
                        if use_voice:
                            try:
                                voice_file = await voice_task
                                if voice_file:
                                    await message.channel.send(file=discord.File(voice_file, filename="response.mp3"))
                            except Exception as e:
                                print(f"Error sending voice response: {e}")
                    else:
                        # Additional chunks
                        await message.channel.send(content=chunk)
            else:
                # Single message response
                if use_voice:
                    # Create voice response in the background
                    voice_task = asyncio.create_task(self.create_voice_response(response))
                    
                # Send the message
                if embed:
                    first_message = await message.channel.send(embed=embed)
                else:
                    first_message = await message.channel.send(content=content)
                    
                # Create thread if requested and possible
                if create_thread and first_message and hasattr(message.channel, 'create_thread'):
                    try:
                        thread_name = message.content[:95] + "..." if len(message.content) > 95 else message.content
                        await first_message.create_thread(name=thread_name)
                    except Exception as e:
                        print(f"Error creating thread: {e}")
                
                # Send voice file if available
                if use_voice:
                    try:
                        voice_file = await voice_task
                        if voice_file:
                            await message.channel.send(file=discord.File(voice_file, filename="response.mp3"))
                    except Exception as e:
                        print(f"Error sending voice response: {e}")
            
            # Store the message ID for future reference
            self.replied_messages[message.id] = first_message.id if first_message else None
            
            return first_message
            
        except Exception as e:
            print(f"Error sending response: {e}")
            # Fallback to plain text if anything fails
            try:
                if len(response) > 2000:
                    chunks = chunk_message(response)
                    first_message = await message.channel.send(content=chunks[0])
                    for chunk in chunks[1:]:
                        await message.channel.send(content=chunk)
                else:
                    first_message = await message.channel.send(content=response[:2000])
                
                # Store the message ID for future reference
                self.replied_messages[message.id] = first_message.id if first_message else None
                
                return first_message
            except:
                print("Critical error sending any response")
                return None

    async def create_voice_response(self, text):
        """Create a voice file from the response text"""
        try:
            # Use a simpler first sentence for the voice response
            first_sentence = re.split(r'(?<=[.!?])\s+', text)[0]
            voice_text = first_sentence[:500]  # Limit to 500 chars
            return await text_to_speech(voice_text)
        except Exception as e:
            print(f"Error creating voice response: {e}")
            return None

    async def send_sequential_thinking_response(self, message, problem, response, initial_message=None,
                                       message_reference=None, is_tools_response=False, is_fallback=False):
        """Send a response for sequential thinking"""
        try:
            # Format the response for better readability
            try:
                # Format the response using our sequential thinking implementation
                formatted_response = response
                if not is_fallback:  # If it's not already formatted through a fallback
                    # The response should already be formatted from sequential_thinking.py
                    # But in case someone calls this function directly with raw text:
                    if "**Thought" not in response and "**Step" not in response:
                        formatted_response = self.sequential_thinking.format_response(response)
                    else:
                        formatted_response = response
            except Exception as e:
                print(f"Error formatting response: {e}")
                formatted_response = response  # Fall back to the raw response
                
            # Check if response is too long for Discord
            if len(formatted_response) > 2000:
                # Split into multiple messages
                chunks = [formatted_response[i:i+1990] for i in range(0, len(formatted_response), 1990)]
                
                if initial_message:
                    try:
                        # Update the initial message with the first chunk
                        await initial_message.edit(content=chunks[0])
                    except discord.errors.NotFound:
                        try:
                            # Try to find it by ID if available
                            if message_reference:
                                try:
                                    found_message = await message.channel.fetch_message(message_reference)
                                    await found_message.edit(content=chunks[0])
                                except:
                                    # Send as a new message if can't find by ID
                                    await message.reply(chunks[0])
                            else:
                                await message.reply(chunks[0])
                        except Exception as e:
                            print(f"Error handling first chunk with not found: {e}")
                            await message.reply(chunks[0])
                    except Exception as e:
                        print(f"Error updating initial message with chunk: {e}")
                        await message.reply(chunks[0])
                else:
                    # Send as a direct reply if no initial message is available
                    await message.reply(chunks[0])
                
                # Send remaining chunks
                for chunk in chunks[1:]:
                    try:
                        await message.channel.send(chunk)
                    except Exception as e:
                        print(f"Error sending chunk: {e}")
                        try:
                            # Try after a brief pause
                            await asyncio.sleep(1)
                            await message.channel.send(chunk)
                        except:
                            pass
            else:
                # Response fits in a single message
                if initial_message:
                    try:
                        # Update the initial message with the full response
                        await initial_message.edit(content=formatted_response)
                    except discord.errors.NotFound:
                        # Try to find by ID if available
                        try:
                            if message_reference:
                                try:
                                    found_message = await message.channel.fetch_message(message_reference)
                                    await found_message.edit(content=formatted_response)
                                except:
                                    # Send as a new message if can't find by ID
                                    await message.reply(formatted_response)
                            else:
                                await message.reply(formatted_response)
                        except Exception as e:
                            print(f"Error handling response with not found: {e}")
                            await message.reply(formatted_response)
                    except Exception as e:
                        print(f"Error updating initial message: {e}")
                        await message.reply(formatted_response)
                else:
                    # Send as a direct reply if no initial message is available
                    await message.reply(formatted_response)
            
            # Log this interaction for metrics/monitoring
            try:
                from bot_utilities.monitoring import AgentMonitor
                monitor = AgentMonitor()
                asyncio.create_task(monitor.log_interaction(
                    command_name="sequential_thinking_text",
                    user_id=str(message.author.id),
                    server_id=str(message.guild.id) if message.guild else "DM",
                    execution_time=0,  # We don't have this measurement here
                    success=True
                ))
            except Exception as e:
                print(f"Error logging interaction: {e}")
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in send_sequential_thinking_response: {error_traceback}")
            
            # Try a simple reply if everything else failed
            try:
                if initial_message:
                    await initial_message.edit(content=f"❌ Error sending response: {str(e)[:200]}")
                else:
                    await message.reply(f"❌ Error sending response: {str(e)[:200]}")
            except:
                try:
                    await message.channel.send(f"❌ Error sending response: {str(e)[:200]}")
                except:
                    pass

    @commands.Cog.listener()
    async def on_message(self, message):
        """Event handler for incoming messages"""
        
        # Ignore messages from the bot itself
        if message.author.id == self.bot.user.id:
            return
            
        # Process the message
        await self.process_message(message)


async def setup(bot):
    await bot.add_cog(OnMessage(bot))