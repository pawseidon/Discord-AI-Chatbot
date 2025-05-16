"""
Configuration Module

This module provides configuration settings for the Discord AI Chatbot.
"""

import os
import yaml
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('config')

# Default configuration values
DEFAULT_CONFIG = {
    # API configuration
    'API_BASE_URL': 'https://api.groq.com/openai/v1',
    'MODEL_ID': 'meta-llama/llama-4-maverick-17b-128e-instruct',
    
    # Bot configuration
    'DEFAULT_BOT_NAME': 'Assistant',
    'TRIGGER': ['assistant', 'ai'],
    'ALLOW_DM': True,
    'SMART_MENTION': True,
    
    # Language and Instruction configuration
    'LANGUAGE': 'en',
    'DEFAULT_INSTRUCTION': 'hand',
    
    # Feature toggles
    'INTERNET_ACCESS': True,
    'USE_LOCAL_MODEL': False,
    
    # Local model configuration
    'LOCAL_MODEL_HOST': '127.0.0.1',
    'LOCAL_MODEL_PORT': '1234',
    'LOCAL_MODEL_ID': 'local',
    
    # Workflow configuration
    'ENABLE_WORKFLOWS': True,
    'DEFAULT_WORKFLOW_TYPE': 'auto'
}

# Load configuration from YAML file
try:
    with open('config.yml', 'r', encoding='utf-8') as file:
        yaml_config = yaml.safe_load(file)
        if yaml_config:
            # Merge with default config
            config = {**DEFAULT_CONFIG, **yaml_config}
            logger.info("Loaded configuration from config.yml")
        else:
            config = DEFAULT_CONFIG
            logger.warning("config.yml was empty, using default configuration")
except FileNotFoundError:
    logger.warning("config.yml not found, using default configuration")
    config = DEFAULT_CONFIG
except Exception as e:
    logger.error(f"Error loading config.yml: {e}")
    config = DEFAULT_CONFIG

# Override with environment variables where available
for key in config.keys():
    if key in os.environ:
        # Handle special types
        if isinstance(config[key], bool):
            config[key] = os.environ[key].lower() in ('true', 'yes', '1', 'y')
        elif isinstance(config[key], int):
            config[key] = int(os.environ[key])
        elif isinstance(config[key], float):
            config[key] = float(os.environ[key])
        elif isinstance(config[key], list):
            # Try to parse as JSON list
            try:
                config[key] = json.loads(os.environ[key])
            except:
                config[key] = os.environ[key].split(',')
        else:
            config[key] = os.environ[key]

# Log configuration (excluding sensitive data)
safe_config = {k: v for k, v in config.items() if not any(sensitive in k.lower() for sensitive in ['key', 'token', 'password', 'secret'])}
logger.info(f"Configuration: {safe_config}")

# Export the config object
__all__ = ['config'] 