from bot_utilities.config_loader import load_current_language, load_instructions, config

# Chatbot and discord config

instructions = load_instructions()
allow_dm = config['ALLOW_DM']
active_channels = set()
smart_mention_enabled = config.get('SMART_MENTION', True)
presences = config["PRESENCES"]
presences_disabled = config["DISABLE_PRESENCE"]
internet_access = config['INTERNET_ACCESS']
instruc_config = config['DEFAULT_INSTRUCTION']

# Message history and config
message_history = {}
MAX_HISTORY = config['MAX_HISTORY']
replied_messages = {}
active_channels = {}
server_settings = {}  # Store per-server settings

# Imagine config
blacklisted_words = config['BLACKLIST_WORDS']
prevent_nsfw = config['AI_NSFW_CONTENT_FILTER']

# Accessability
current_language = load_current_language()

def smart_mention(msg_content: str, bot) -> bool:
    """
    Check if the message content contains a smart mention of the bot
    
    Args:
        msg_content: The message content
        bot: The Discord bot instance
        
    Returns:
        bool: True if the message contains a smart mention, False otherwise
    """
    if not smart_mention_enabled:
        return False
        
    msg_content_lower = msg_content.lower()
    
    # Get bot names and triggers
    bot_names = [bot.user.name.lower()]
    if hasattr(bot.user, 'display_name') and bot.user.display_name:
        bot_names.append(bot.user.display_name.lower())
    
    # Add any custom names from config
    if 'BOT_NAMES' in config and isinstance(config['BOT_NAMES'], list):
        for name in config['BOT_NAMES']:
            bot_names.append(name.lower())
    
    # Check for bot names in message
    for name in bot_names:
        if name and name in msg_content_lower:
            return True
            
    # Check for trigger words from config
    if 'TRIGGER' in config and isinstance(config['TRIGGER'], list):
        for trigger in config['TRIGGER']:
            # Handle special placeholders in triggers
            processed_trigger = trigger.lower()
            processed_trigger = processed_trigger.replace('%bot_name%', bot.user.name.lower() if bot.user.name else '')
            processed_trigger = processed_trigger.replace('%bot_nickname%', bot.user.display_name.lower() if bot.user.display_name else '')
            processed_trigger = processed_trigger.replace('%bot_username%', bot.user.name.lower() if bot.user.name else '')
            
            if processed_trigger in msg_content_lower:
                return True
    
    # Check for prefix usage
    if 'PREFIXES' in config and isinstance(config['PREFIXES'], list):
        for prefix in config['PREFIXES']:
            if msg_content_lower.startswith(prefix.lower()):
                return True
    
    return False