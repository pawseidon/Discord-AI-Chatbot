"""
Test Sequential Thinking Formatting

This script tests the formatting of sequential thinking output for Discord UI.
"""

import asyncio
import os
import sys
import json
import re
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import required services
from bot_utilities.services.sequential_thinking_service import sequential_thinking_service
from bot_utilities.ai_utils import get_ai_provider

# Discord message character limit
DISCORD_CHAR_LIMIT = 2000

class MockDiscord:
    """
    A simple class to simulate Discord message constraints
    """
    @staticmethod
    def format_message(content: str) -> list:
        """Format a message to respect Discord's character limit"""
        if len(content) <= DISCORD_CHAR_LIMIT:
            return [content]
            
        # Split long messages
        chunks = []
        current_chunk = ""
        
        for line in content.split('\n'):
            # If adding this line would exceed the limit, start a new chunk
            if len(current_chunk) + len(line) + 1 > DISCORD_CHAR_LIMIT:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If the line itself is too long, split it further
                if len(line) > DISCORD_CHAR_LIMIT:
                    words = line.split(' ')
                    current_line = ""
                    
                    for word in words:
                        if len(current_line) + len(word) + 1 > DISCORD_CHAR_LIMIT:
                            chunks.append(current_line)
                            current_line = word
                        else:
                            current_line += " " + word if current_line else word
                    
                    if current_line:
                        current_chunk = current_line
                else:
                    current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    @staticmethod
    def simulate_rendering(content: str) -> str:
        """Simulate how Discord would render certain Markdown elements"""
        # Discord doesn't support nested lists well, convert to simpler format
        content = re.sub(r'(\s+)[-*]\s+', r'\1• ', content)
        
        # Replace Markdown headers with bolded text for testing
        content = re.sub(r'^#{3,6}\s+(.+)$', r'**\1**', content, flags=re.MULTILINE)
        content = re.sub(r'^#{1,2}\s+(.+)$', r'**\1**', content, flags=re.MULTILINE)
        
        # Ensure code blocks are formatted
        content = re.sub(r'```(\w+)?\n(.*?)```', r'```\n\2```', content, flags=re.DOTALL)
        
        return content

async def test_sequential_thinking_format():
    """Test the formatting of sequential thinking output"""
    print("Testing sequential thinking formatting...")
    
    # Test queries
    queries = [
        "Explain the process of photosynthesis step by step",
        "Break down how to solve a Rubik's cube",
        "Analyze the causes of climate change",
        "Describe how the Internet works"
    ]
    
    # Initialize services
    llm_provider = await get_ai_provider()
    await sequential_thinking_service.initialize(llm_provider)
    
    # Create output directory
    os.makedirs("test_results", exist_ok=True)
    
    # Test different configurations
    results = {}
    
    for query in queries:
        print(f"\nTesting query: {query}")
        session_id = f"format_test_{hash(query)}"
        
        results[query] = {}
        
        # Test basic sequential thinking
        print("  Testing basic sequential thinking...")
        success, basic_response = await sequential_thinking_service.process_sequential_thinking(
            problem=query,
            context={"testing_formatting": True},
            prompt_style="sequential",
            num_thoughts=3,
            temperature=0.3,
            enable_revision=False,
            enable_reflection=False,
            session_id=session_id
        )
        
        # Test with revisions
        print("  Testing sequential thinking with revisions...")
        success, revised_response = await sequential_thinking_service.process_sequential_thinking(
            problem=query,
            context={"testing_formatting": True},
            prompt_style="sequential",
            num_thoughts=3,
            temperature=0.3,
            enable_revision=True,
            enable_reflection=False,
            session_id=f"{session_id}_revision"
        )
        
        # Test with reflection
        print("  Testing sequential thinking with reflection...")
        success, reflection_response = await sequential_thinking_service.process_sequential_thinking(
            problem=query,
            context={"testing_formatting": True},
            prompt_style="sequential",
            num_thoughts=3,
            temperature=0.3,
            enable_revision=False,
            enable_reflection=True,
            session_id=f"{session_id}_reflection"
        )
        
        # Simulate Discord rendering
        discord_basic = MockDiscord.simulate_rendering(basic_response)
        discord_revised = MockDiscord.simulate_rendering(revised_response)
        discord_reflection = MockDiscord.simulate_rendering(reflection_response)
        
        # Check for Discord formatting issues
        basic_chunks = MockDiscord.format_message(discord_basic)
        revised_chunks = MockDiscord.format_message(discord_revised)
        reflection_chunks = MockDiscord.format_message(discord_reflection)
        
        # Check for proper step formatting
        basic_has_steps = len(re.findall(r'(Step|Thought) \d+', discord_basic)) > 0
        steps_formatted = len(re.findall(r'🔄.*?(Step|Thought)', discord_basic)) > 0
        conclusion_formatted = '✅' in discord_basic
        
        format_success = basic_has_steps and steps_formatted and conclusion_formatted
        
        # Look for revision formatting
        if 'revision' in discord_revised.lower():
            revision_formatted = '🔁' in discord_revised
        else:
            revision_formatted = True  # Not applicable
        
        # Save results
        results[query] = {
            "basic": {
                "response": basic_response,
                "discord_rendered": discord_basic,
                "num_chunks": len(basic_chunks),
                "has_steps": basic_has_steps,
                "steps_formatted": steps_formatted,
                "conclusion_formatted": conclusion_formatted,
                "overall_success": format_success
            },
            "revision": {
                "response": revised_response,
                "discord_rendered": discord_revised,
                "num_chunks": len(revised_chunks),
                "revision_formatted": revision_formatted
            },
            "reflection": {
                "response": reflection_response,
                "discord_rendered": discord_reflection,
                "num_chunks": len(reflection_chunks)
            }
        }
        
        # Save example outputs to files for visual inspection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f"test_results/sequential_basic_{timestamp}.md", "w") as f:
            f.write(f"# Original Response\n\n{basic_response}\n\n# Discord Rendered\n\n{discord_basic}")
            
        with open(f"test_results/sequential_revised_{timestamp}.md", "w") as f:
            f.write(f"# Original Response\n\n{revised_response}\n\n# Discord Rendered\n\n{discord_revised}")
            
        with open(f"test_results/sequential_reflection_{timestamp}.md", "w") as f:
            f.write(f"# Original Response\n\n{reflection_response}\n\n# Discord Rendered\n\n{discord_reflection}")
        
        print(f"  Basic format success: {'✅' if format_success else '❌'}")
        print(f"  Revision format success: {'✅' if revision_formatted else '❌'}")
        print(f"  Basic format chunks: {len(basic_chunks)}")
        print(f"  Revision format chunks: {len(revised_chunks)}")
        print(f"  Reflection format chunks: {len(reflection_chunks)}")
    
    # Save overall results
    with open(f"test_results/sequential_format_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        # Convert results to a serializable format by removing full responses
        serializable_results = {}
        for query, result in results.items():
            serializable_results[query] = {
                "basic": {
                    "num_chunks": result["basic"]["num_chunks"],
                    "has_steps": result["basic"]["has_steps"],
                    "steps_formatted": result["basic"]["steps_formatted"],
                    "conclusion_formatted": result["basic"]["conclusion_formatted"],
                    "overall_success": result["basic"]["overall_success"]
                },
                "revision": {
                    "num_chunks": result["revision"]["num_chunks"],
                    "revision_formatted": result["revision"]["revision_formatted"]
                },
                "reflection": {
                    "num_chunks": result["reflection"]["num_chunks"]
                }
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nTest results saved to test_results/ directory")

async def main():
    """Run the script"""
    await test_sequential_thinking_format()

if __name__ == "__main__":
    asyncio.run(main()) 