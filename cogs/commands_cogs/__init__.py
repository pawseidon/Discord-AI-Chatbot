import os
import importlib

async def get_command_cogs():
    """
    Returns all command cog modules
    """
    command_cogs = []
    command_cogs_dir = os.path.dirname(__file__)
    for file in os.listdir(command_cogs_dir):
        if file.endswith(".py") and not file.startswith("__init__"):
            # Import the file as a module
            module_name = file[:-3]  # Remove .py
            module = importlib.import_module(f"cogs.commands_cogs.{module_name}")
            command_cogs.append(module)
        elif os.path.isdir(os.path.join(command_cogs_dir, file)) and not file.startswith("__"):
            # Check if there's an __init__.py file in the directory
            init_path = os.path.join(command_cogs_dir, file, "__init__.py")
            if os.path.exists(init_path):
                module = importlib.import_module(f"cogs.commands_cogs.{file}")
                command_cogs.append(module)
    return command_cogs

async def get_all_cogs():
    """
    Returns all cog classes from command cog modules
    """
    # These are the new cogs we added
    cog_modules = [
        "AgentCog", 
        "KnowledgeBaseCog",
        "ImageCog",
        "StatsCog", 
        "ReflectiveRAGCog",
        "MCPAgentCog",
        # List other cogs here
    ]
    
    # In the future, this can be auto-detected
    return cog_modules
