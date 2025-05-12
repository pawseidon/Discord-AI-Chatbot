import discord
from discord.ext import commands

from ..common import current_language, instructions, instruc_config, message_history
from bot_utilities.config_loader import load_active_channels
import json
from bot_utilities.memory_utils import UserPreferences
import os
from bot_utilities.fallback_utils import FALLBACK_DIR

class ChatConfigCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_channels = load_active_channels

    @commands.hybrid_command(name="toggleactive", description=current_language["toggleactive"])
    @discord.app_commands.choices(persona=[
        discord.app_commands.Choice(name=persona.capitalize(), value=persona)
        for persona in instructions
    ])
    @commands.has_permissions(administrator=True)
    async def toggleactive(self, ctx, persona: discord.app_commands.Choice[str] = instructions[instruc_config]):
        channel_id = f"{ctx.channel.id}"
        active_channels = self.active_channels()
        if channel_id in active_channels:
            del active_channels[channel_id]
            with open("channels.json", "w", encoding='utf-8') as f:
                json.dump(active_channels, f, indent=4)
            await ctx.send(f"{ctx.channel.mention} {current_language['toggleactive_msg_1']}", delete_after=3)
        else:
            active_channels[channel_id] = persona.value if persona.value else persona
            with open("channels.json", "w", encoding='utf-8') as f:
                json.dump(active_channels, f, indent=4)
            await ctx.send(f"{ctx.channel.mention} {current_language['toggleactive_msg_2']}", delete_after=3)

    @commands.hybrid_command(name="clear", description=current_language["bonk"])
    async def clear(self, ctx):
        try:
            key = f"{ctx.author.id}-{ctx.channel.id}"
            message_history[key].clear()
        except Exception as e:
            await ctx.send(f"⚠️ There is no message history to be cleared \n ```{e}```", delete_after=2)
            return
        
        await ctx.send("Message history has been cleared", delete_after=4)

    @commands.hybrid_command(name="preferences", description="View or update your personal bot preferences")
    async def preferences(self, ctx, response_length: str = None, voice_enabled: bool = None, use_embeds: bool = None):
        """
        View or update your preferences for the bot
        
        Parameters:
        response_length (str, optional): Set your preferred response length (short, medium, long)
        voice_enabled (bool, optional): Enable or disable voice responses
        use_embeds (bool, optional): Enable or disable rich embed responses
        """
        # If no parameters provided, just show current preferences
        if response_length is None and voice_enabled is None and use_embeds is None:
            prefs = await UserPreferences.get_user_preferences(ctx.author.id)
            
            # Format the preferences into a nice embed
            embed = discord.Embed(
                title="Your AI Assistant Preferences",
                description="Here are your current preferences for interacting with me:",
                color=discord.Color.blue()
            )
            
            # Add preference fields
            embed.add_field(
                name="Response Length", 
                value=prefs.get("preferred_response_length", "medium").capitalize(),
                inline=True
            )
            
            embed.add_field(
                name="Voice Enabled", 
                value="Yes" if prefs.get("use_voice", False) else "No",
                inline=True
            )
            
            embed.add_field(
                name="Rich Embeds", 
                value="Yes" if prefs.get("use_embeds", True) else "No",
                inline=True
            )
            
            # Add topics of interest if any
            topics = prefs.get("topics_of_interest", [])
            if topics:
                embed.add_field(
                    name="Topics You're Interested In",
                    value=", ".join(topics).capitalize(),
                    inline=False
                )
            
            embed.set_footer(text=f"Use /preferences to update these settings")
            await ctx.send(embed=embed)
            return
            
        # Update preferences if parameters provided
        if response_length:
            # Validate response length
            if response_length.lower() not in ["short", "medium", "long"]:
                await ctx.send("⚠️ Response length must be 'short', 'medium', or 'long'", delete_after=4)
                return
                
            await UserPreferences.update_user_preference(
                ctx.author.id, 
                "preferred_response_length", 
                response_length.lower()
            )
            
        if voice_enabled is not None:
            await UserPreferences.update_user_preference(
                ctx.author.id,
                "use_voice",
                voice_enabled
            )
            
        if use_embeds is not None:
            await UserPreferences.update_user_preference(
                ctx.author.id,
                "use_embeds",
                use_embeds
            )
            
        await ctx.send("✅ Your preferences have been updated!", delete_after=4)

    @commands.hybrid_command(name="toggleoffline", description="Toggle offline mode for the bot")
    @commands.has_permissions(administrator=True)
    async def toggle_offline(self, ctx):
        """Toggle offline mode (fallback mode) for the bot - Admin only"""
        offline_flag = os.path.join(FALLBACK_DIR, "offline_mode")
        
        if os.path.exists(offline_flag):
            try:
                os.remove(offline_flag)
                await ctx.send("✅ Online mode activated. The bot will now use the language model for responses.", ephemeral=True)
            except Exception as e:
                await ctx.send(f"❌ Error disabling offline mode: {e}", ephemeral=True)
        else:
            try:
                with open(offline_flag, 'w') as f:
                    f.write('1')  # Just a placeholder
                await ctx.send("⚠️ Offline mode activated. The bot will now use fallback responses without connecting to the language model.", ephemeral=True)
            except Exception as e:
                await ctx.send(f"❌ Error enabling offline mode: {e}", ephemeral=True)

async def setup(bot):
    await bot.add_cog(ChatConfigCog(bot))
