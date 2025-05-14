import discord
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# Define the bot
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    """Remove all slash commands when the bot is ready"""
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('-' * 50)
    print('Starting to remove all slash commands...')
    
    try:
        # Fetch all global commands
        global_commands = await bot.http.request(
            discord.http.Route('GET', '/applications/{app_id}/commands', 
                           app_id=bot.user.id)
        )
        
        print(f"Found {len(global_commands)} global commands")
        
        # Delete each global command
        for cmd in global_commands:
            cmd_id = cmd['id']
            cmd_name = cmd['name']
            print(f"Deleting global command: {cmd_name} ({cmd_id})")
            
            try:
                await bot.http.request(
                    discord.http.Route('DELETE', '/applications/{app_id}/commands/{cmd_id}', 
                                   app_id=bot.user.id, cmd_id=cmd_id)
                )
                print(f"✅ Successfully deleted global command: {cmd_name}")
            except Exception as e:
                print(f"❌ Failed to delete global command {cmd_name}: {e}")
        
        # Get a list of all guilds
        for guild in bot.guilds:
            print(f"\nProcessing guild: {guild.name} (ID: {guild.id})")
            
            # Fetch all guild commands
            guild_commands = await bot.http.request(
                discord.http.Route('GET', '/applications/{app_id}/guilds/{guild_id}/commands', 
                               app_id=bot.user.id, guild_id=guild.id)
            )
            
            print(f"Found {len(guild_commands)} commands in {guild.name}")
            
            # Delete each guild command
            for cmd in guild_commands:
                cmd_id = cmd['id']
                cmd_name = cmd['name']
                print(f"Deleting command in {guild.name}: {cmd_name} ({cmd_id})")
                
                try:
                    await bot.http.request(
                        discord.http.Route('DELETE', '/applications/{app_id}/guilds/{guild_id}/commands/{cmd_id}', 
                                       app_id=bot.user.id, guild_id=guild.id, cmd_id=cmd_id)
                    )
                    print(f"✅ Successfully deleted command {cmd_name} in {guild.name}")
                except Exception as e:
                    print(f"❌ Failed to delete command {cmd_name} in {guild.name}: {e}")
        
        print('\nCommand removal process completed!')
        print('-' * 50)
        print('All slash commands have been removed from the bot.')
        print('The bot will now exit. Restart your main bot application for the changes to take effect.')
        
    except Exception as e:
        print(f"An error occurred during the command removal process: {e}")
    
    # Close the bot session
    await bot.close()

# Run the bot
if __name__ == "__main__":
    print("Starting slash command removal script...")
    print("This script will remove ALL slash commands from your bot.")
    print("Press Ctrl+C now to cancel if you don't want to proceed.")
    print("Waiting 5 seconds before proceeding...")
    
    try:
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(5))
        print("Proceeding with command removal...")
        bot.run(TOKEN)
    except KeyboardInterrupt:
        print("Operation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}") 