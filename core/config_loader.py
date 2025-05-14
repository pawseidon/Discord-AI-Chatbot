"""
Configuration loader module for Discord AI Chatbot.

This module provides configuration loading, language settings,
and instruction loading capabilities.
"""

import yaml
import json
import os
import logging
from typing import Dict, Any, Optional, List

# Set up logging
logger = logging.getLogger("config_loader")

class ConfigLoader:
    """Configuration loader for Discord AI Chatbot"""
    
    def __init__(self, config_file='config.yml', lang_directory='lang', instructions_directory='instructions'):
        """
        Initialize the configuration loader
        
        Args:
            config_file: Path to the YAML configuration file
            lang_directory: Path to the language files directory
            instructions_directory: Path to the instructions directory
        """
        self.config_file = config_file
        self.lang_directory = lang_directory
        self.instructions_directory = instructions_directory
        self.config = {}
        self.current_language = {}
        self.instructions = {}
        self.active_channels = {}
        self.valid_language_codes = []
        
        # Load configuration
        self.load_config()
        
        # Find valid language codes
        self._find_valid_languages()
        
        # Load current language
        self.load_current_language()
        
        # Load instructions
        self.load_instructions()
        
        # Load active channels
        self.load_active_channels()
        
        logger.info(f"Configuration loaded from {config_file}")
        logger.info(f"Current language: {self.get_config('LANGUAGE', 'en')}")
        logger.info(f"Found {len(self.instructions)} instruction files")
    
    def load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as config_file:
                self.config = yaml.safe_load(config_file)
            return self.config
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            self.config = {}
            return {}
    
    def _find_valid_languages(self) -> List[str]:
        """Find valid language codes in the language directory"""
        self.valid_language_codes = []
        try:
            for filename in os.listdir(self.lang_directory):
                if filename.startswith("lang.") and filename.endswith(".json") and os.path.isfile(
                        os.path.join(self.lang_directory, filename)):
                    language_code = filename.split(".")[1]
                    self.valid_language_codes.append(language_code)
            return self.valid_language_codes
        except Exception as e:
            logger.error(f"Error finding valid languages: {e}")
            return []
    
    def load_current_language(self) -> Dict[str, Any]:
        """Load the current language file"""
        try:
            current_language_code = self.get_config('LANGUAGE', 'en')
            lang_file_path = os.path.join(
                self.lang_directory, f"lang.{current_language_code}.json")
            
            with open(lang_file_path, encoding="utf-8") as lang_file:
                self.current_language = json.load(lang_file)
            
            return self.current_language
        except Exception as e:
            logger.error(f"Error loading language file: {e}")
            self.current_language = {}
            return {}
    
    def load_instructions(self) -> Dict[str, str]:
        """Load instruction files from the instructions directory"""
        try:
            self.instructions = {}
            for file_name in os.listdir(self.instructions_directory):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(self.instructions_directory, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                    # Use the file name without extension as the variable name
                    variable_name = file_name.split('.')[0]
                    self.instructions[variable_name] = file_content
            
            return self.instructions
        except Exception as e:
            logger.error(f"Error loading instructions: {e}")
            self.instructions = {}
            return {}
    
    def load_active_channels(self) -> Dict[str, Any]:
        """Load active channels from channels.json"""
        try:
            if os.path.exists("channels.json"):
                with open("channels.json", "r", encoding='utf-8') as f:
                    self.active_channels = json.load(f)
            
            return self.active_channels
        except Exception as e:
            logger.error(f"Error loading active channels: {e}")
            self.active_channels = {}
            return {}
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with an optional default
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> Any:
        """
        Update a configuration value in memory (not persisted to file)
        
        Args:
            key: Configuration key
            value: New value
            
        Returns:
            Updated value
        """
        self.config[key] = value
        return value
    
    def save_config(self) -> bool:
        """
        Save the current configuration to the YAML file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_language_string(self, key: str, default: str = "") -> str:
        """
        Get a string from the current language file
        
        Args:
            key: Language string key
            default: Default value if key not found
            
        Returns:
            Localized string or default
        """
        return self.current_language.get(key, default)
    
    def get_instruction(self, name: str, default: str = "") -> str:
        """
        Get an instruction by name
        
        Args:
            name: Instruction name
            default: Default value if instruction not found
            
        Returns:
            Instruction content or default
        """
        return self.instructions.get(name, default)

# Global instance for singleton pattern
_config_loader = None

def get_config_loader(config_file='config.yml', lang_directory='lang', instructions_directory='instructions') -> ConfigLoader:
    """
    Get or create the global configuration loader instance
    
    Args:
        config_file: Path to the YAML configuration file
        lang_directory: Path to the language files directory
        instructions_directory: Path to the instructions directory
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(
            config_file=config_file,
            lang_directory=lang_directory,
            instructions_directory=instructions_directory
        )
        
    return _config_loader

# Convenience functions for direct access
def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value with an optional default"""
    loader = get_config_loader()
    return loader.get_config(key, default)

def set_config(key: str, value: Any) -> Any:
    """Update a configuration value in memory"""
    loader = get_config_loader()
    return loader.set_config(key, value)

def get_language_string(key: str, default: str = "") -> str:
    """Get a string from the current language file"""
    loader = get_config_loader()
    return loader.get_language_string(key, default)

def get_instruction(name: str, default: str = "") -> str:
    """Get an instruction by name"""
    loader = get_config_loader()
    return loader.get_instruction(name, default)

def load_current_language() -> Dict[str, Any]:
    """Load the current language file"""
    loader = get_config_loader()
    return loader.load_current_language()

def load_instructions() -> Dict[str, str]:
    """Load instruction files"""
    loader = get_config_loader()
    return loader.load_instructions()

def load_active_channels() -> Dict[str, Any]:
    """Load active channels"""
    loader = get_config_loader()
    return loader.load_active_channels() 