import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the functions we want to test
from bot_utilities.ai_utils import get_bot_names_and_triggers

class TestBotNames(unittest.TestCase):
    
    @patch('bot_utilities.ai_utils.config')
    def test_get_bot_names_and_triggers(self, mock_config):
        # Set up the mock configuration
        mock_config.get.return_value = "TestBot"
        mock_config.__getitem__.side_effect = lambda key: {
            "TRIGGER": ["%BOT_NAME%", "%BOT_NICKNAME%", "%BOT_USERNAME%"]
        }.get(key)
        
        # Call the function
        result = get_bot_names_and_triggers()
        
        # Verify the results
        self.assertEqual(result["name"], "TestBot")
        self.assertIn("testbot", result["names"])
        self.assertEqual(result["triggers"], [])  # Should be empty as these are dynamic placeholders
    
    def test_print_config(self):
        """Print the current configuration for debugging"""
        from bot_utilities.config_loader import config
        print("\nCurrent TRIGGER configuration:")
        print(config.get("TRIGGER", []))

if __name__ == "__main__":
    unittest.main() 