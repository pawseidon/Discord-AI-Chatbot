import logging
import asyncio
import time
import discord
from discord.ext import commands
from typing import Dict, Any, List, Optional, Union, Callable
import traceback

logger = logging.getLogger("discord_integration")

class InteractionContext:
    """
    Context wrapper for Discord interactions
    Provides standardized access to different interaction types
    """
    def __init__(self, interaction, cache_provider=None):
        """
        Initialize an interaction context
        
        Args:
            interaction: Discord interaction
            cache_provider: Optional cache provider
        """
        self.interaction = interaction
        self.cache_provider = cache_provider
        self.start_time = time.time()
        self.interaction_type = self._get_interaction_type()
        
        # Extract common properties regardless of interaction type
        self.user = interaction.user
        self.user_id = str(interaction.user.id)
        self.channel_id = str(interaction.channel_id) if interaction.channel else None
        self.guild_id = str(interaction.guild_id) if interaction.guild else None
        
        # Store interaction-specific data
        self.data = self._extract_data()
        
        # Response tracking
        self.has_responded = False
        self.response_time = None
        self.deferred = False
    
    def _get_interaction_type(self) -> str:
        """Determine the type of interaction"""
        if hasattr(self.interaction, 'command_name'):
            return "slash_command"
        elif hasattr(self.interaction, 'custom_id'):
            if hasattr(self.interaction, 'message'):
                return "component"
            else:
                return "modal"
        else:
            return "unknown"
    
    def _extract_data(self) -> Dict[str, Any]:
        """Extract relevant data based on interaction type"""
        data = {
            "interaction_type": self.interaction_type,
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "guild_id": self.guild_id,
            "timestamp": self.start_time
        }
        
        # Add interaction-specific data
        if self.interaction_type == "slash_command":
            data["command_name"] = self.interaction.command_name
            data["options"] = self._extract_options()
        
        elif self.interaction_type == "component":
            data["custom_id"] = self.interaction.custom_id
            data["component_type"] = str(self.interaction.type)
            
            # Extract values from select menus if applicable
            if hasattr(self.interaction, 'values') and self.interaction.values:
                data["values"] = self.interaction.values
        
        elif self.interaction_type == "modal":
            data["custom_id"] = self.interaction.custom_id
            data["components"] = self._extract_modal_values()
        
        return data
    
    def _extract_options(self) -> Dict[str, Any]:
        """Extract options from a slash command"""
        if not hasattr(self.interaction, 'options'):
            return {}
        
        options = {}
        for option in self.interaction.options:
            if hasattr(option, 'options') and option.options:
                # Handle subcommands
                options[option.name] = self._extract_suboptions(option)
            else:
                options[option.name] = option.value
        return options
    
    def _extract_suboptions(self, option) -> Dict[str, Any]:
        """Extract options from a subcommand"""
        if not hasattr(option, 'options'):
            return {}
        
        suboptions = {}
        for subopt in option.options:
            suboptions[subopt.name] = subopt.value
        return suboptions
    
    def _extract_modal_values(self) -> Dict[str, str]:
        """Extract values from modal components"""
        if not hasattr(self.interaction, 'data') or not hasattr(self.interaction.data, 'components'):
            return {}
        
        values = {}
        try:
            for component in self.interaction.data.components:
                for child in component.components:
                    values[child.custom_id] = child.value
        except Exception as e:
            logger.error(f"Error extracting modal values: {e}")
        
        return values
    
    async def defer(self, ephemeral: bool = False) -> None:
        """Defer the response"""
        if not self.has_responded:
            await self.interaction.response.defer(ephemeral=ephemeral, thinking=True)
            self.deferred = True
    
    async def respond(self, content: str, ephemeral: bool = False, 
                     embeds: List[discord.Embed] = None,
                     view: discord.ui.View = None) -> None:
        """Respond to the interaction appropriately"""
        embeds = embeds or []
        
        try:
            if not self.has_responded:
                # Initial response
                await self.interaction.response.send_message(
                    content=content,
                    embeds=embeds,
                    ephemeral=ephemeral,
                    view=view
                )
                self.has_responded = True
            elif self.deferred:
                # Follow-up after deferring
                await self.interaction.followup.send(
                    content=content,
                    embeds=embeds,
                    ephemeral=ephemeral,
                    view=view
                )
            else:
                # Edit original response
                await self.interaction.edit_original_response(
                    content=content,
                    embeds=embeds,
                    view=view
                )
            
            self.response_time = time.time() - self.start_time
            
            # Cache interaction data if cache provider available
            if self.cache_provider:
                await self._cache_interaction()
        
        except Exception as e:
            logger.error(f"Error responding to interaction: {e}")
            # Try to send a follow-up if main response fails
            try:
                await self.interaction.followup.send(
                    content="I encountered an error while processing your request.",
                    ephemeral=True
                )
            except:
                pass
    
    async def _cache_interaction(self) -> None:
        """Cache interaction data and response"""
        if not self.cache_provider:
            return
        
        try:
            # Cache interaction data
            cache_data = {
                **self.data,
                "response_time": self.response_time,
                "timestamp": time.time()
            }
            
            # Use appropriate cache type based on interaction
            if self.interaction_type == "slash_command":
                cache_type = "interaction"
            else:
                cache_type = "interaction"  # Can be specialized if needed
            
            key = f"interaction:{int(time.time())}"
            await self.cache_provider.set(
                key,
                cache_data,
                cache_type=cache_type,
                user_id=self.user_id,
                channel_id=self.channel_id,
                guild_id=self.guild_id
            )
        except Exception as e:
            logger.error(f"Error caching interaction: {e}")

class DiscordIntegration:
    """
    Integration layer between Discord and the AI chatbot components
    
    This class handles the integration between Discord's API and the bot's
    internal components like caching, reasoning, etc.
    """
    
    def __init__(self, 
               bot: commands.Bot, 
               base_cache=None,
               context_cache=None,
               semantic_cache=None,
               reasoning_router=None, 
               reasoning_integration=None,
               hallucination_handler=None):
        """
        Initialize Discord integration layer
        
        Args:
            bot: Discord bot instance
            base_cache: Base cache provider
            context_cache: Context-aware cache for conversations
            semantic_cache: Semantic cache for similar queries
            reasoning_router: Router for reasoning methods
            reasoning_integration: Enhanced reasoning integration with context-awareness
            hallucination_handler: Handler for hallucinations
        """
        self.bot = bot
        self.base_cache = base_cache
        self.context_cache = context_cache
        self.semantic_cache = semantic_cache
        self.reasoning_router = reasoning_router
        self.reasoning_integration = reasoning_integration
        self.hallucination_handler = hallucination_handler
        
        # Keep track of active conversations
        self.active_conversations = {}
        
        # Command handlers
        self.command_handlers = {}
        self.component_handlers = {}
        self.modal_handlers = {}
        
        # Response metrics
        self.metrics = {
            "total_messages": 0,
            "total_responses": 0,
            "avg_response_time": 0,
            "total_response_time": 0,
            "cached_responses": 0,
            "semantic_matches": 0,
            "context_hits": 0,
            "integrated_reasoning_used": 0,
            "errors": 0
        }
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register Discord event handlers"""
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Bot is ready! Logged in as {self.bot.user}")
        
        @self.bot.event
        async def on_interaction(interaction):
            await self.handle_interaction(interaction)
    
    def register_command(self, name: str, handler: Callable):
        """
        Register a slash command handler
        
        Args:
            name: Command name
            handler: Command handler function
        """
        self.command_handlers[name] = handler
        logger.info(f"Registered command handler for /{name}")
    
    def register_component(self, custom_id_pattern: str, handler: Callable):
        """
        Register a component interaction handler
        
        Args:
            custom_id_pattern: Pattern to match in custom_id
            handler: Component handler function
        """
        self.component_handlers[custom_id_pattern] = handler
        logger.info(f"Registered component handler for pattern '{custom_id_pattern}'")
    
    def register_modal(self, custom_id_pattern: str, handler: Callable):
        """
        Register a modal submission handler
        
        Args:
            custom_id_pattern: Pattern to match in custom_id
            handler: Modal handler function
        """
        self.modal_handlers[custom_id_pattern] = handler
        logger.info(f"Registered modal handler for pattern '{custom_id_pattern}'")
    
    async def handle_interaction(self, interaction: discord.Interaction):
        """
        Handle a Discord interaction
        
        Args:
            interaction: Discord interaction
        """
        # Create context wrapper
        ctx = create_interaction_context(interaction, self.base_cache)
        
        try:
            if ctx.interaction_type == "slash_command":
                await self._handle_command(ctx)
            
            elif ctx.interaction_type == "component":
                await self._handle_component(ctx)
            
            elif ctx.interaction_type == "modal":
                await self._handle_modal(ctx)
            
            else:
                logger.warning(f"Unknown interaction type: {ctx.interaction_type}")
        
        except Exception as e:
            logger.error(f"Error handling interaction: {e}")
            logger.error(traceback.format_exc())
            
            # Try to respond with error if not already responded
            if not ctx.has_responded:
                try:
                    await ctx.respond("I encountered an error processing your request.", ephemeral=True)
                except Exception as respond_error:
                    logger.error(f"Failed to respond with error: {respond_error}")
    
    async def _handle_command(self, ctx: InteractionContext):
        """
        Handle slash command interaction
        
        Args:
            ctx: Interaction context
        """
        command_name = ctx.data.get("command_name")
        
        # Check if we have a registered handler
        if command_name in self.command_handlers:
            await self.command_handlers[command_name](ctx)
        else:
            # Default handling based on command name
            if command_name == "ask":
                await self._handle_ask_command(ctx)
            elif command_name == "clear":
                await self._handle_clear_command(ctx)
            else:
                await ctx.respond(f"I don't know how to handle the /{command_name} command yet.")
    
    async def _handle_component(self, ctx: InteractionContext):
        """
        Handle component interaction
        
        Args:
            ctx: Interaction context
        """
        custom_id = ctx.data.get("custom_id", "")
        
        # Find matching handler based on pattern
        for pattern, handler in self.component_handlers.items():
            if pattern in custom_id:
                await handler(ctx)
                return
        
        # Default handling if no handler found
        await ctx.respond("I don't know how to handle this component interaction.", ephemeral=True)
    
    async def _handle_modal(self, ctx: InteractionContext):
        """
        Handle modal interaction
        
        Args:
            ctx: Interaction context
        """
        custom_id = ctx.data.get("custom_id", "")
        
        # Find matching handler based on pattern
        for pattern, handler in self.modal_handlers.items():
            if pattern in custom_id:
                await handler(ctx)
                return
        
        # Default handling if no handler found
        await ctx.respond("I don't know how to handle this modal submission.", ephemeral=True)
    
    async def _handle_ask_command(self, ctx: InteractionContext):
        """
        Handle the /ask command
        
        Args:
            ctx: Interaction context
        """
        # Defer response since reasoning might take time
        await ctx.defer()
        
        # Extract query from options
        options = ctx.data.get("options", {})
        query = options.get("query", "")
        reasoning_type = options.get("reasoning", None)
        
        if not query:
            await ctx.respond("Please provide a query to ask me about!")
            return
        
        # Get conversation context
        context = await self._get_interaction_context(ctx.interaction)
        
        # Process query
        formatted_response = await self.process_interaction(
            interaction=ctx.interaction,
            query=query,
            reasoning_method=reasoning_type,
            ephemeral=False
        )
        
        # Send response
        if formatted_response:
            content = formatted_response.get("content", "")
            embed = formatted_response.get("embed")
            
            embeds = [embed] if embed else []
            await ctx.respond(content, embeds=embeds)
        else:
            await ctx.respond("I'm sorry, but I couldn't generate a response.")
    
    async def _handle_clear_command(self, ctx: InteractionContext):
        """
        Handle the /clear command to clear conversation history
        
        Args:
            ctx: Interaction context
        """
        await ctx.defer(ephemeral=True)
        
        # Clear conversation context
        cleared = await self.clear_user_context(
            user_id=ctx.user_id,
            guild_id=ctx.guild_id,
            channel_id=ctx.channel_id
        )
        
        await ctx.respond(f"Your conversation history has been cleared! ({cleared} entries removed)", ephemeral=True)
    
    async def process_message(self, 
                            message: discord.Message, 
                            reasoning_method: Optional[str] = None) -> Optional[str]:
        """
        Process a message from Discord
        
        Args:
            message: Discord message
            reasoning_method: Optional reasoning method to use
            
        Returns:
            Response message or None if no response
        """
        start_time = time.time()
        response = None
        
        try:
            # Skip bot messages
            if message.author.bot:
                return None
            
            self.metrics["total_messages"] += 1
            
            # Get conversation context
            context = await self._get_conversation_context(message)
            
            # Try to get cached response first
            if self.context_cache:
                cached_response = await self.context_cache.get_response_with_context(
                    query=message.content,
                    user_id=str(message.author.id),
                    guild_id=str(message.guild.id) if message.guild else None,
                    channel_id=str(message.channel.id),
                    context_messages=context.get("conversation_history", [])
                )
                
                if cached_response:
                    # Check if it's a context hit or semantic match
                    if cached_response.get("semantic_match"):
                        self.metrics["semantic_matches"] += 1
                    else:
                        self.metrics["context_hits"] += 1
                    
                    self.metrics["cached_responses"] += 1
                    
                    # Add cache indicator to response
                    if "answer" in cached_response:
                        response = cached_response["answer"]
                        
                        # Add any additional processing for cached responses
                        # like adding emoji indicators
                        if "method_emoji" in cached_response:
                            response = f"{cached_response['method_emoji']} {response} *(cached)*"
                        else:
                            response = f"{response} *(cached)*"
                        
                        # Add sources if available
                        if "sources" in cached_response and cached_response["sources"]:
                            sources_text = "\n\n**Sources:**"
                            for i, source in enumerate(cached_response["sources"][:3]):  # Limit to 3 sources
                                sources_text += f"\n{i+1}. {source.get('title', 'Unknown')} {source.get('url', '')}"
                            response += sources_text
                        
                        # Update conversation context with cached response
                        await self._update_conversation_context(
                            message, 
                            response,
                            is_cached=True
                        )
                        
                        return response
            
            # Use integrated reasoning if available
            response_data = None
            if self.reasoning_integration:
                try:
                    # Use integrated reasoning with context awareness
                    response_data = await self.reasoning_integration.process_query(
                        query=message.content,
                        user_id=str(message.author.id),
                        channel_id=str(message.channel.id),
                        context=context
                    )
                    self.metrics["integrated_reasoning_used"] += 1
                except Exception as e:
                    logger.error(f"Error from integrated reasoning: {e}")
                    logger.error(traceback.format_exc())
                    # Fall back to standard reasoning router
            
            # Fall back to standard reasoning router if integrated reasoning failed or not available
            if not response_data and self.reasoning_router:
                # Convert reasoning method string to enum if provided
                method_enum = None
                if reasoning_method and self.reasoning_router:
                    try:
                        from features.reasoning.reasoning_router import ReasoningMethod
                        method_enum = ReasoningMethod(reasoning_method)
                    except (ValueError, ImportError) as e:
                        logger.warning(f"Invalid reasoning method: {reasoning_method} - {e}")
                
                # Get response using reasoning router
                try:
                    response_data = await self.reasoning_router.route_query(
                        query=message.content,
                        user_id=str(message.author.id),
                        method=method_enum,
                        context=context
                    )
                except Exception as e:
                    logger.error(f"Error from reasoning router: {e}")
                    logger.error(traceback.format_exc())
                    
                    response = "I'm having trouble processing your request. Please try again later."
                    self.metrics["errors"] += 1
            
            if response_data:
                response = response_data.get("answer", "No answer generated.")
                
                # Add reasoning method emoji if available
                if "method_emoji" in response_data:
                    response = f"{response_data['method_emoji']} {response}"
                
                # Add sources if available
                if "sources" in response_data and response_data["sources"]:
                    sources_text = "\n\n**Sources:**"
                    for i, source in enumerate(response_data["sources"][:3]):  # Limit to 3 sources
                        sources_text += f"\n{i+1}. {source.get('title', 'Unknown')} {source.get('url', '')}"
                    response += sources_text
                
                # Add thinking process visualization for hybrid methods
                if "thinking_process" in response_data and isinstance(response_data["thinking_process"], dict):
                    if "sequential" in response_data["thinking_process"] and "graph" in response_data["thinking_process"]:
                        response += "\n\n*Solved using sequential thinking and graph analysis*"
                    elif "retrieved_documents" in response_data["thinking_process"] and "sequential" in response_data["thinking_process"]:
                        response += "\n\n*Information retrieved and analyzed sequentially*"
                    elif "planning" in response_data["thinking_process"] and "actions" in response_data["thinking_process"]:
                        response += "\n\n*Planned and executed actions to solve this*"
                
                # Cache the response if it's not from cache already
                if self.context_cache and not response_data.get("from_cache", False):
                    await self.context_cache.store_response_with_context(
                        query=message.content,
                        response=response_data,
                        user_id=str(message.author.id),
                        guild_id=str(message.guild.id) if message.guild else None,
                        channel_id=str(message.channel.id),
                        context_messages=context.get("conversation_history", [])
                    )
                    
                    # Also store in semantic cache if applicable
                    if self.semantic_cache and not self._is_context_dependent(
                        message.content, 
                        context.get("conversation_history", [])
                    ):
                        await self.semantic_cache.set(
                            query=message.content,
                            response=response_data,
                            user_id=str(message.author.id)
                        )
                        
            # If no response from reasoning methods
            if not response:
                # Fallback to simple response
                logger.info("No response from reasoning methods, using fallback response")
                response = "I couldn't generate a proper response. Please try again or rephrase your question."
            
            # Update conversation context
            await self._update_conversation_context(message, response)
            
            # Update metrics
            self.metrics["total_responses"] += 1
            processing_time = time.time() - start_time
            self.metrics["total_response_time"] += processing_time
            self.metrics["avg_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["total_responses"]
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            return "I encountered an error while processing your message. Please try again later."
    
    async def _get_conversation_context(self, message: discord.Message) -> Dict[str, Any]:
        """
        Get conversation context for a message
        
        Args:
            message: Discord message
            
        Returns:
            Conversation context
        """
        context = {
            "conversation_history": [],
            "channel_type": str(message.channel.type),
            "guild_info": None,
            "user_info": {
                "id": str(message.author.id),
                "name": message.author.name,
                "display_name": message.author.display_name
            }
        }
        
        # Add guild info if in a guild
        if message.guild:
            context["guild_info"] = {
                "id": str(message.guild.id),
                "name": message.guild.name
            }
        
        # Get conversation history from context-aware cache if available
        if self.context_cache:
            conversation_history = await self.context_cache.update_conversation_context(
                user_id=str(message.author.id),
                guild_id=str(message.guild.id) if message.guild else None,
                channel_id=str(message.channel.id)
            )
            
            if conversation_history:
                context["conversation_history"] = conversation_history
        # Fallback to base cache
        elif self.base_cache:
            cache_key = f"conversation:{message.channel.id}"
            conversation_history = await self.base_cache.get(
                cache_key,
                cache_type="conversation",
                user_id=str(message.author.id)
            )
            
            if conversation_history:
                context["conversation_history"] = conversation_history
        
        return context
    
    async def _update_conversation_context(self, 
                                         message: discord.Message, 
                                         response: str,
                                         is_cached: bool = False) -> None:
        """
        Update conversation context with a new message and response
        
        Args:
            message: Discord message
            response: Bot response
            is_cached: Whether the response is from cache
        """
        new_message = {
            "user": message.content,
            "bot": response,
            "timestamp": time.time(),
            "is_cached": is_cached
        }
        
        # Use context-aware cache if available
        if self.context_cache:
            await self.context_cache.update_conversation_context(
                user_id=str(message.author.id),
                guild_id=str(message.guild.id) if message.guild else None,
                channel_id=str(message.channel.id),
                new_message=new_message
            )
        # Fallback to base cache
        elif self.base_cache:
            # Get current conversation history
            cache_key = f"conversation:{message.channel.id}"
            conversation_history = await self.base_cache.get(
                cache_key,
                cache_type="conversation",
                user_id=str(message.author.id)
            ) or []
            
            # Add new message and response
            conversation_history.append(new_message)
            
            # Limit history length (keep last 10 exchanges)
            conversation_history = conversation_history[-10:]
            
            # Save updated history
            await self.base_cache.set(
                cache_key,
                conversation_history,
                cache_type="conversation",
                ttl=3600,  # 1 hour
                user_id=str(message.author.id)
            )
    
    def _is_context_dependent(self, query: str, context_messages: List[Dict[str, Any]]) -> bool:
        """
        Check if a query is context-dependent
        
        Args:
            query: User query
            context_messages: Context messages
            
        Returns:
            True if query is context-dependent, False otherwise
        """
        # If no context, it's not context-dependent
        if not context_messages:
            return False
        
        # Check for direct references to context in the query
        context_reference_terms = ["you said", "earlier", "previous", "before", "last time", 
                                 "you mentioned", "as mentioned", "above", "that"]
        
        for term in context_reference_terms:
            if term in query.lower():
                return True
        
        # Check for pronouns that might indicate context dependence
        pronoun_indicators = ["it", "this", "that", "they", "them", "these", "those"]
        
        # If query starts with a standalone pronoun, it's likely context-dependent
        query_words = query.lower().split()
        if query_words and query_words[0] in pronoun_indicators:
            return True
            
        return False
    
    async def process_interaction(self, 
                                interaction: discord.Interaction, 
                                query: str,
                                reasoning_method: Optional[str] = None,
                                ephemeral: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process an interaction from Discord
        
        Args:
            interaction: Discord interaction
            query: Query from interaction
            reasoning_method: Optional reasoning method to use
            ephemeral: Whether to send response as ephemeral
            
        Returns:
            Response data or None if no response
        """
        start_time = time.time()
        response_data = None
        
        try:
            self.metrics["total_messages"] += 1
            
            # Get interaction context
            context = await self._get_interaction_context(interaction)
            
            # Try to get cached response first
            if self.context_cache:
                cached_response = await self.context_cache.get_response_with_context(
                    query=query,
                    user_id=str(interaction.user.id),
                    guild_id=str(interaction.guild_id) if interaction.guild_id else None,
                    channel_id=str(interaction.channel_id) if interaction.channel_id else None,
                    context_messages=context.get("conversation_history", [])
                )
                
                if cached_response:
                    # Check if it's a context hit or semantic match
                    if cached_response.get("semantic_match"):
                        self.metrics["semantic_matches"] += 1
                    else:
                        self.metrics["context_hits"] += 1
                    
                    self.metrics["cached_responses"] += 1
                    
                    # Add information that this was a cached response
                    cached_response["from_cache"] = True
                    
                    # Update conversation context with cached response
                    await self._update_interaction_context(
                        interaction,
                        query,
                        cached_response.get("answer", "No answer found in cache"),
                        is_cached=True
                    )
                    
                    return cached_response
            
            # Use integrated reasoning if available
            if self.reasoning_integration:
                try:
                    # Use integrated reasoning with context awareness
                    response_data = await self.reasoning_integration.process_query(
                        query=query,
                        user_id=str(interaction.user.id),
                        channel_id=str(interaction.channel_id) if interaction.channel_id else None,
                        context=context
                    )
                    self.metrics["integrated_reasoning_used"] += 1
                except Exception as e:
                    logger.error(f"Error from integrated reasoning: {e}")
                    logger.error(traceback.format_exc())
                    # Fall back to standard reasoning router
                    
            # Fall back to standard reasoning router if integrated reasoning failed or not available
            if not response_data and self.reasoning_router:
                # Convert reasoning method string to enum if provided
                method_enum = None
                if reasoning_method:
                    try:
                        from features.reasoning.reasoning_router import ReasoningMethod
                        method_enum = ReasoningMethod(reasoning_method)
                    except (ValueError, ImportError) as e:
                        logger.warning(f"Invalid reasoning method: {reasoning_method} - {e}")
                
                # Get response using reasoning router
                try:
                    response_data = await self.reasoning_router.route_query(
                        query=query,
                        user_id=str(interaction.user.id),
                        method=method_enum,
                        context=context
                    )
                except Exception as e:
                    logger.error(f"Error from reasoning router: {e}")
                    logger.error(traceback.format_exc())
                    self.metrics["errors"] += 1
                    
                    # Create error response
                    response_data = {
                        "answer": "I'm having trouble processing your request. Please try again later.",
                        "method": "error",
                        "method_emoji": "❌",
                        "error": str(e)
                    }
            
            # If still no response data, create a fallback response
            if not response_data:
                response_data = {
                    "answer": "I couldn't generate a proper response. Please try again or rephrase your question.",
                    "method": "fallback",
                    "method_emoji": "⚠️"
                }
            
            # Format response for Discord
            formatted_response = self.format_response_for_discord(response_data)
            
            # Update conversation context with the response
            await self._update_interaction_context(
                interaction,
                query,
                response_data.get("answer", "No answer generated")
            )
            
            # Update metrics
            self.metrics["total_responses"] += 1
            processing_time = time.time() - start_time
            self.metrics["total_response_time"] += processing_time
            self.metrics["avg_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["total_responses"]
            )
            
            # Add processing time to response if it's too long
            if processing_time > 5.0 and "content" in formatted_response:
                formatted_response["content"] += f"\n\n*Processing time: {processing_time:.2f}s*"
            
            return formatted_response
        
        except Exception as e:
            logger.error(f"Error in process_interaction: {e}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            
            # Create and format error response
            error_response = {
                "answer": "I encountered an error while processing your request. Please try again later.",
                "method": "error",
                "method_emoji": "❌",
                "error": str(e)
            }
            
            return self.format_response_for_discord(error_response)
    
    async def _get_interaction_context(self, interaction: discord.Interaction) -> Dict[str, Any]:
        """
        Get conversation context for an interaction
        
        Args:
            interaction: Discord interaction
            
        Returns:
            Conversation context
        """
        context = {
            "conversation_history": [],
            "channel_type": "interaction",
            "guild_info": None,
            "user_info": {
                "id": str(interaction.user.id),
                "name": interaction.user.name,
                "display_name": getattr(interaction.user, "display_name", interaction.user.name)
            }
        }
        
        # Add guild info if in a guild
        if interaction.guild:
            context["guild_info"] = {
                "id": str(interaction.guild.id),
                "name": interaction.guild.name
            }
        
        # Get conversation history from context-aware cache if available
        if self.context_cache:
            conversation_history = await self.context_cache.update_conversation_context(
                user_id=str(interaction.user.id),
                guild_id=str(interaction.guild.id) if interaction.guild else None,
                channel_id="interaction"  # Special channel ID for interactions
            )
            
            if conversation_history:
                context["conversation_history"] = conversation_history
        # Fallback to base cache
        elif self.base_cache:
            # Use user ID for interaction context
            cache_key = f"conversation:interaction:{interaction.user.id}"
            conversation_history = await self.base_cache.get(
                cache_key,
                cache_type="conversation",
                user_id=str(interaction.user.id)
            )
            
            if conversation_history:
                context["conversation_history"] = conversation_history
        
        return context
    
    async def _update_interaction_context(self, 
                                        interaction: discord.Interaction, 
                                        query: str,
                                        response: str,
                                        is_cached: bool = False) -> None:
        """
        Update conversation context with a new interaction and response
        
        Args:
            interaction: Discord interaction
            query: User query
            response: Bot response
            is_cached: Whether the response is from cache
        """
        new_message = {
            "user": query,
            "bot": response,
            "timestamp": time.time(),
            "is_cached": is_cached
        }
        
        # Use context-aware cache if available
        if self.context_cache:
            await self.context_cache.update_conversation_context(
                user_id=str(interaction.user.id),
                guild_id=str(interaction.guild.id) if interaction.guild else None,
                channel_id="interaction",  # Special channel ID for interactions
                new_message=new_message
            )
        # Fallback to base cache
        elif self.base_cache:
            # Get current conversation history
            cache_key = f"conversation:interaction:{interaction.user.id}"
            conversation_history = await self.base_cache.get(
                cache_key,
                cache_type="conversation",
                user_id=str(interaction.user.id)
            ) or []
            
            # Add new interaction and response
            conversation_history.append(new_message)
            
            # Limit history length (keep last 10 exchanges)
            conversation_history = conversation_history[-10:]
            
            # Save updated history
            await self.base_cache.set(
                cache_key,
                conversation_history,
                cache_type="conversation",
                ttl=3600,  # 1 hour
                user_id=str(interaction.user.id)
            )
    
    async def clear_user_context(self, user_id: str, guild_id: Optional[str] = None, 
                               channel_id: Optional[str] = None) -> int:
        """
        Clear conversation context for a user
        
        Args:
            user_id: User ID
            guild_id: Optional guild ID
            channel_id: Optional channel ID
            
        Returns:
            Number of entries cleared
        """
        count = 0
        
        # Clear from context-aware cache if available
        if self.context_cache:
            count += await self.context_cache.invalidate_cache_for_user(user_id)
        
        # Clear from semantic cache if available
        if self.semantic_cache:
            count += await self.semantic_cache.clear(user_id)
        
        # Clear from base cache if available
        if self.base_cache:
            count += await self.base_cache.clear(
                cache_type="conversation",
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id
            )
        
        return count
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics"""
        metrics = dict(self.metrics)
        
        # Add cache metrics if available
        if self.base_cache and hasattr(self.base_cache, "get_metrics"):
            try:
                cache_metrics = await self.base_cache.get_metrics()
                metrics["base_cache"] = cache_metrics
            except Exception as e:
                logger.error(f"Error getting base cache metrics: {e}")
        
        # Add context cache metrics if available
        if self.context_cache and hasattr(self.context_cache, "get_metrics"):
            try:
                context_metrics = await self.context_cache.get_metrics()
                metrics["context_cache"] = context_metrics
            except Exception as e:
                logger.error(f"Error getting context cache metrics: {e}")
        
        # Add semantic cache metrics if available
        if self.semantic_cache and hasattr(self.semantic_cache, "get_metrics"):
            try:
                semantic_metrics = await self.semantic_cache.get_metrics()
                metrics["semantic_cache"] = semantic_metrics
            except Exception as e:
                logger.error(f"Error getting semantic cache metrics: {e}")
        
        # Add reasoning metrics if available
        if self.reasoning_router and hasattr(self.reasoning_router, "get_metrics"):
            try:
                reasoning_metrics = await self.reasoning_router.get_metrics()
                metrics["reasoning"] = reasoning_metrics
            except Exception as e:
                logger.error(f"Error getting reasoning metrics: {e}")
        
        # Add hallucination metrics if available
        if self.hallucination_handler and hasattr(self.hallucination_handler, "get_metrics"):
            try:
                hallucination_metrics = await self.hallucination_handler.get_metrics()
                metrics["hallucination"] = hallucination_metrics
            except Exception as e:
                logger.error(f"Error getting hallucination metrics: {e}")
        
        return metrics
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        # Cleanup base cache if available
        if self.base_cache and hasattr(self.base_cache, "cleanup"):
            try:
                await self.base_cache.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up base cache: {e}")
    
    def format_response_for_discord(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a response data object for Discord output
        
        Args:
            response_data: Response data
            
        Returns:
            Formatted response for Discord
        """
        # Basic response content
        content = response_data.get("answer", "No response generated.")
        
        # Add method emoji if available
        if "method_emoji" in response_data:
            content = f"{response_data['method_emoji']} {content}"
            
        # Add semantic match indicator if it came from semantic cache
        if response_data.get("semantic_match"):
            semantic_match = response_data["semantic_match"]
            similarity = semantic_match.get("similarity_score", 0)
            if similarity > 0.9:
                content += f" *(cached - similar question)*"
        
        # Create embed for additional information
        embed = None
        if any(key in response_data for key in ["sources", "thoughts", "reasoning_steps", "nodes"]):
            embed = discord.Embed(
                title="Additional Information",
                color=discord.Color.blue()
            )
            
            # Add sources if available
            if "sources" in response_data and response_data["sources"]:
                sources_value = ""
                for i, source in enumerate(response_data["sources"][:3]):  # Limit to 3 sources
                    sources_value += f"{i+1}. {source.get('title', 'Unknown')} {source.get('url', '')}\n"
                
                embed.add_field(
                    name="Sources",
                    value=sources_value if sources_value else "No sources",
                    inline=False
                )
            
            # Add reasoning info if available
            if "thoughts" in response_data and response_data["thoughts"]:
                thoughts_value = "\n".join([
                    f"{t.get('thought_number', i+1)}. {t.get('content', 'No content')}"
                    for i, t in enumerate(response_data["thoughts"][:3])
                ])
                
                embed.add_field(
                    name="Thought Process",
                    value=thoughts_value if thoughts_value else "No thoughts",
                    inline=False
                )
            
            # Add processing time
            if "processing_time" in response_data:
                embed.set_footer(text=f"Processed in {response_data['processing_time']:.2f}s")
        
        return {
            "content": content,
            "embed": embed
        }

def create_discord_integration(bot: commands.Bot, **kwargs) -> DiscordIntegration:
    """
    Factory function to create a Discord integration
    
    Args:
        bot: Discord bot instance
        **kwargs: Additional components (response_cache, reasoning_router, etc.)
        
    Returns:
        Configured DiscordIntegration instance
    """
    # For backward compatibility
    if "response_cache" in kwargs:
        kwargs["base_cache"] = kwargs.pop("response_cache")
    
    return DiscordIntegration(
        bot=bot,
        **kwargs
    )

def create_interaction_context(interaction, cache_provider=None) -> InteractionContext:
    """
    Factory function to create an interaction context
    
    Args:
        interaction: Discord interaction
        cache_provider: Optional cache provider
        
    Returns:
        Configured InteractionContext instance
    """
    return InteractionContext(
        interaction=interaction,
        cache_provider=cache_provider
    ) 