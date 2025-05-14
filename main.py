#!/usr/bin/env python
"""
Discord AI Chatbot - Main entry point
"""
import os
import sys
import asyncio
import logging
import json
from typing import Dict, Any
from pathlib import Path

from core.bot import create_bot, initialize_bot

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("discord_bot.log"),
        logging.StreamHandler(stream=sys.stdout)
    ]
)

logger = logging.getLogger("main")

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file {config_path}")
        return {}

def get_token() -> str:
    """
    Get Discord bot token from environment or config
    
    Returns:
        Discord bot token
    """
    # Check environment first
    token = os.getenv("DISCORD_TOKEN")
    if token:
        return token
    
    # Then check config
    config = load_config()
    token = config.get("token")
    if token:
        return token
    
    # Prompt user for token if not found
    logger.warning("Discord token not found in environment or config")
    print("Please enter your Discord bot token:")
    token = input().strip()
    return token

async def main():
    """Main entry point"""
    try:
        # Get token
        token = get_token()
        if not token:
            logger.error("No Discord token provided. Exiting.")
            return
        
        # Load config
        config = load_config()
        prefix = config.get("prefix", "!")
        
        # Create and initialize bot
        bot = create_bot(token=token, prefix=prefix)
        await initialize_bot(bot, config)
        
        # Start bot
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Ensure proper cleanup
        if 'bot' in locals():
            await bot.close()

if __name__ == "__main__":
    # Run the async main
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
