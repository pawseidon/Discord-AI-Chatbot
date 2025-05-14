#!/usr/bin/env python3
"""
Redundant Code Cleanup Script

This script scans the codebase for redundant code that could be replaced with service calls.
It identifies common patterns that should be migrated to the service architecture.

Usage:
    python clean_redundant_code.py             # Scan and report redundancies
    python clean_redundant_code.py --fix       # Scan and add deprecation comments
    python clean_redundant_code.py --help      # Show help message
"""

import os
import re
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Set

# Define patterns to look for and their replacements
REDUNDANCY_PATTERNS = [
    {
        "name": "UserPreferences.get_user_preferences",
        "pattern": r"UserPreferences\.get_user_preferences\s*\(\s*([^)]+)\s*\)",
        "replacement": "memory_service.get_user_preferences(\\1)",
        "import": "from bot_utilities.services.memory_service import memory_service",
        "files": [r".*\.py$"],
        "exclude_files": [r"bot_utilities/services/memory_service\.py$"]
    },
    {
        "name": "save_user_preferences",
        "pattern": r"(UserPreferences\.)?save_user_preferences\s*\(\s*([^,]+),\s*([^)]+)\s*\)",
        "replacement": "memory_service.set_user_preferences(\\2, \\3)",
        "import": "from bot_utilities.services.memory_service import memory_service",
        "files": [r".*\.py$"],
        "exclude_files": [r"bot_utilities/services/memory_service\.py$"]
    },
    {
        "name": "update_user_preference",
        "pattern": r"(UserPreferences\.)?update_user_preference\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\s*\)",
        "replacement": "memory_service.set_user_preference(\\2, \\3, \\4)",
        "import": "from bot_utilities.services.memory_service import memory_service",
        "files": [r".*\.py$"],
        "exclude_files": [r"bot_utilities/services/memory_service\.py$"]
    },
    {
        "name": "process_conversation_history",
        "pattern": r"process_conversation_history\s*\(\s*([^,]+),\s*([^,)]+)(?:,\s*([^)]+))?\s*\)",
        "replacement": "memory_service.get_conversation_history(\\1, \\2\\3)",
        "import": "from bot_utilities.services.memory_service import memory_service",
        "files": [r".*\.py$"],
        "exclude_files": [r"bot_utilities/services/memory_service\.py$"]
    },
    {
        "name": "split_response",
        "pattern": r"split_response\s*\(\s*([^)]+)\s*\)",
        "replacement": "message_service.split_message(\\1)",
        "import": "from bot_utilities.services.message_service import message_service",
        "files": [r".*\.py$"],
        "exclude_files": [r"bot_utilities/services/message_service\.py$"]
    },
    {
        "name": "run_agent direct call",
        "pattern": r"run_agent\s*\(\s*([^)]*)\s*\)",
        "replacement": "agent_service.process_query(\\1)",
        "import": "from bot_utilities.services.agent_service import agent_service",
        "files": [r".*\.py$"],
        "exclude_files": [r"bot_utilities/services/agent_service\.py$"]
    },
    {
        "name": "Direct LLM call",
        "pattern": r"generate_response\s*\(\s*([^)]*)\s*\)",
        "replacement": "agent_service.process_query(\\1)",
        "import": "from bot_utilities.services.agent_service import agent_service",
        "files": [r".*\.py$"],
        "exclude_files": [r"bot_utilities/services/agent_service\.py$"]
    },
    {
        "name": "format_response_for_discord",
        "pattern": r"format_response_for_discord\s*\(\s*([^)]*)\s*\)",
        "replacement": "message_service.format_response(\\1)",
        "import": "from bot_utilities.services.message_service import message_service",
        "files": [r".*\.py$"],
        "exclude_files": [r"bot_utilities/services/message_service\.py$"]
    }
]

# List of files that have been deleted or fully migrated
# This helps the script avoid looking for patterns in files that no longer exist
DELETED_FILES = [
    "bot_utilities/agent_utils.py",
    "sequential_thinking.py",  # Moved to bot_utilities/services/sequential_thinking_service.py
    "bot_utilities/response_utils.py",
    "bot_utilities/memory_utils.py",
    "bot_utilities/agent_memory.py",
    "bot_utilities/agent_tools_manager.py",
    "bot_utilities/agent_orchestrator.py",
    "bot_utilities/agent_workflow_manager.py",
    "bot_utilities/symbolic_reasoning_registry.py",
    "bot_utilities/symbolic_reasoning.py",
    "bot_utilities/reasoning_utils.py",
    "bot_utilities/reasoning_cache.py",
    "bot_utilities/formatting_utils.py",
    "cogs/commands_cogs/NekoCog.py",
]

def is_match_for_file(file_path: str, pattern_info: Dict) -> bool:
    """Check if a file matches the include/exclude patterns"""
    # Skip files that have been deleted or fully migrated
    for deleted_file in DELETED_FILES:
        if deleted_file in file_path:
            return False
    
    # Check if file matches include pattern
    include_match = False
    for file_pattern in pattern_info.get("files", [r".*\.py$"]):
        if re.match(file_pattern, file_path):
            include_match = True
            break
    
    if not include_match:
        return False
    
    # Check if file matches exclude pattern
    for exclude_pattern in pattern_info.get("exclude_files", []):
        if re.match(exclude_pattern, file_path):
            return False
    
    return True

def scan_file(file_path: str, pattern_info: Dict) -> List[Tuple[int, str, str]]:
    """Scan a file for a redundancy pattern and return matches with line numbers"""
    matches = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print(f"Skipping file due to encoding issues: {file_path}")
        return matches
        
    pattern = pattern_info["pattern"]
    replacement = pattern_info["replacement"]
    
    for i, line in enumerate(lines):
        for match in re.finditer(pattern, line):
            # Create replacement by substituting captured groups
            replaced_text = re.sub(pattern, replacement, match.group(0))
            matches.append((i + 1, match.group(0), replaced_text))
    
    return matches

def fix_file(file_path: str, pattern_info: Dict, add_imports: bool = True) -> int:
    """
    Fix a file by adding deprecation comments to redundant code
    Returns the number of fixes made
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print(f"Skipping file due to encoding issues: {file_path}")
        return 0
    
    pattern = pattern_info["pattern"]
    replacement = pattern_info["replacement"]
    fixes_made = 0
    
    # Keep track of lines that need imports
    needs_import = False
    
    # Process each line
    for i in range(len(lines)):
        # Check if the line contains the pattern
        if re.search(pattern, lines[i]):
            # Don't modify if it's inside a string literal or comment
            stripped = lines[i].lstrip()
            if stripped.startswith('#') or lines[i].count('"') % 2 == 1 or lines[i].count("'") % 2 == 1:
                continue
                
            # Check if the line is already commented with a deprecation notice
            if "DEPRECATED" in lines[i] or "Use service instead" in lines[i]:
                continue
                
            # Add deprecation notice
            indent = re.match(r'^\s*', lines[i]).group(0)
            fixed_code = re.sub(pattern, replacement, lines[i].rstrip())
            
            lines[i] = f"{lines[i].rstrip()} # DEPRECATED: Use {replacement} instead\n"
            fixes_made += 1
            needs_import = True
    
    # Add import if needed and requested
    if needs_import and add_imports:
        import_statement = pattern_info["import"] + "\n"
        
        # Check if the import already exists
        import_exists = False
        for line in lines:
            if pattern_info["import"] in line:
                import_exists = True
                break
        
        if not import_exists:
            # Find a suitable place to add the import
            import_section_end = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    import_section_end = i + 1
                elif line.strip() and import_section_end > 0 and not line.startswith("#"):
                    break
            
            # Insert the import statement
            if import_section_end > 0:
                lines.insert(import_section_end, import_statement)
                fixes_made += 1
            else:
                # No import section found, add at the top after any comments/docstrings
                first_code_line = 0
                in_docstring = False
                for i, line in enumerate(lines):
                    if '"""' in line or "'''" in line:
                        in_docstring = not in_docstring
                    elif not line.strip() or line.startswith("#"):
                        continue
                    elif not in_docstring:
                        first_code_line = i
                        break
                
                lines.insert(first_code_line, import_statement + "\n")
                fixes_made += 1
    
    # Write the changes back to the file
    if fixes_made > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return fixes_made

def scan_directory(directory: str, pattern_info: Dict) -> List[Tuple[str, int, str, str]]:
    """Scan a directory recursively for files matching the pattern"""
    all_matches = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            
            if is_match_for_file(relative_path, pattern_info):
                file_matches = scan_file(file_path, pattern_info)
                for line_number, matched_text, replacement in file_matches:
                    all_matches.append((relative_path, line_number, matched_text, replacement))
    
    return all_matches

def fix_directory(directory: str, pattern_info: Dict, add_imports: bool = True) -> int:
    """Fix redundant code in a directory recursively"""
    fixes_made = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            
            if is_match_for_file(relative_path, pattern_info):
                fixes_made += fix_file(file_path, pattern_info, add_imports)
    
    return fixes_made

def main():
    parser = argparse.ArgumentParser(description='Scan for redundant code that could be replaced with service calls.')
    parser.add_argument('--fix', action='store_true', help='Add deprecation comments to redundant code')
    parser.add_argument('--directory', '-d', default='.', help='Directory to scan (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--pattern', '-p', help='Only check for specific pattern by name')
    
    args = parser.parse_args()
    
    directory = args.directory
    fix_mode = args.fix
    verbose = args.verbose
    specific_pattern = args.pattern
    
    total_matches = 0
    total_fixes = 0
    
    # Filter patterns if specific pattern is requested
    patterns_to_check = REDUNDANCY_PATTERNS
    if specific_pattern:
        patterns_to_check = [p for p in REDUNDANCY_PATTERNS if p["name"] == specific_pattern]
        if not patterns_to_check:
            print(f"No pattern found with name '{specific_pattern}'")
            print("Available patterns:")
            for p in REDUNDANCY_PATTERNS:
                print(f"  - {p['name']}")
            sys.exit(1)
    
    # Check each pattern
    for pattern_info in patterns_to_check:
        pattern_name = pattern_info["name"]
        
        print(f"\nChecking for '{pattern_name}'...")
        
        if fix_mode:
            # Fix files
            fixes = fix_directory(directory, pattern_info)
            total_fixes += fixes
            if fixes > 0:
                print(f"  Added {fixes} deprecation comments")
            else:
                print("  No matches found")
        else:
            # Scan files
            matches = scan_directory(directory, pattern_info)
            total_matches += len(matches)
            
            if matches:
                print(f"  Found {len(matches)} occurrences:")
                for file_path, line_number, matched_text, replacement in matches:
                    if verbose:
                        print(f"    {file_path}:{line_number}: {matched_text.strip()} -> {replacement.strip()}")
                    else:
                        print(f"    {file_path}:{line_number}")
            else:
                print("  No matches found")
    
    # Print summary
    print("\nSummary:")
    if fix_mode:
        print(f"  Added {total_fixes} deprecation comments")
        if total_fixes > 0:
            print("  Run script without --fix to see what remains")
    else:
        print(f"  Found {total_matches} occurrences of redundant code")
        if total_matches > 0:
            print("  Run with --fix to add deprecation comments")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 