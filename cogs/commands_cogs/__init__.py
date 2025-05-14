import os
import importlib

async def get_command_cogs():
    """
    Returns all command cog modules
    """
    command_cogs = []
    command_cogs_dir = os.path.dirname(__file__)
    
    # Only load the cogs we need
    target_cogs = ['HelpCog.py', 'ReasoningCog.py', 'ChatConfigCog.py', 'EmojiReactionCog.py']
    
    for file in os.listdir(command_cogs_dir):
        if file in target_cogs:
            # Import the file as a module
            module_name = file[:-3]  # Remove .py
            module = importlib.import_module(f"cogs.commands_cogs.{module_name}")
            command_cogs.append(module)
            
    return command_cogs

async def get_all_cogs():
    """
    Returns all cog classes from command cog modules
    """
    # These are the cogs we want to load - keeping only what's truly needed
    cog_modules = [
        "HelpCog",          # Help information
        "ReasoningCog",     # Context-aware reasoning - our primary interaction system
        "ChatConfigCog",    # Configuration commands for the chatbot
        "EmojiReactionCog", # Emoji reactions for reasoning types
    ]
    
    # In the future, this can be auto-detected
    return cog_modules
