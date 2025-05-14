from bot_utilities.config_loader import load_current_language, load_instructions, config
from bot_utilities.services.memory_service import memory_service
from bot_utilities.services.message_service import message_service

# Chatbot and discord config
instructions = load_instructions()
allow_dm = config['ALLOW_DM']
active_channels = set()
smart_mention = config['SMART_MENTION']
presences = config["PRESENCES"]
presences_disabled = config["DISABLE_PRESENCE"]
internet_access = config['INTERNET_ACCESS']
instruc_config = config['DEFAULT_INSTRUCTION']

# Message history and config
# DEPRECATED: Use memory_service.add_to_history instead
# This dictionary is maintained for backward compatibility
message_history = {}
MAX_HISTORY = config['MAX_HISTORY']

# DEPRECATED: Use message_service for message handling
replied_messages = {}

# Active channels tracking is now provided by load_active_channels function
active_channels = {}

# Imagine config
blacklisted_words = config['BLACKLIST_WORDS']
prevent_nsfw = config['AI_NSFW_CONTENT_FILTER']

# Accessability
current_language = load_current_language()