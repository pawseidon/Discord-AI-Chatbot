#!/usr/bin/env python3
"""
Workflow Analyzer Script

This script analyzes the workflow_service.py file to identify which 
workflows are referenced but don't have corresponding implementation files.
"""

import os
import re
import sys

# Configuration
WORKFLOWS_DIR = "bot_utilities/services/workflows"
WORKFLOW_SERVICE_PATH = "bot_utilities/services/workflow_service.py"

def extract_workflow_methods(file_path):
    """Extract workflow method names from workflow_service.py"""
    workflow_methods = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Pattern to match workflow method definitions
    pattern = r"async def ([a-zA-Z_]+_workflow)\s*\("
    matches = re.findall(pattern, content)
    
    for match in matches:
        if match != "process_with_workflow":  # Skip the general processor
            workflow_methods.append(match)
            
    return workflow_methods

def get_existing_implementation_files(dir_path):
    """Get the list of workflow implementation files"""
    implementation_files = []
    
    for file in os.listdir(dir_path):
        if file.endswith("_workflow.py"):
            implementation_files.append(file[:-3])  # Remove .py extension
            
    return implementation_files

def analyze_workflows():
    """Analyze which workflows need implementation files"""
    print("Analyzing workflow implementations...")
    
    # Get methods from workflow_service.py
    service_methods = extract_workflow_methods(WORKFLOW_SERVICE_PATH)
    print(f"Found {len(service_methods)} workflow methods in workflow_service.py")
    
    # Get existing implementation files
    existing_implementations = get_existing_implementation_files(WORKFLOWS_DIR)
    print(f"Found {len(existing_implementations)} workflow implementation files")
    
    # Find missing implementations
    missing_implementations = [method for method in service_methods 
                              if method not in existing_implementations]
    
    # Find implementation files not referenced in workflow_service.py
    unreferenced_implementations = [impl for impl in existing_implementations 
                                   if impl not in service_methods]
    
    # Print results
    print("\n======== ANALYSIS RESULTS ========")
    
    if missing_implementations:
        print("\n🔴 Missing Implementation Files:")
        for method in missing_implementations:
            print(f"  - {method}.py")
            print(f"    Implementation needed for: bot_utilities/services/workflows/{method}.py")
    else:
        print("\n✅ All workflow methods have implementation files!")
        
    if unreferenced_implementations:
        print("\n⚠️ Unreferenced Implementation Files:")
        for impl in unreferenced_implementations:
            print(f"  - {impl}.py (exists but not referenced in workflow_service.py)")
    else:
        print("\n✅ All implementation files are properly referenced!")
        
    # Print summary
    print("\n======== SUMMARY ========")
    print(f"Total workflow methods:  {len(service_methods)}")
    print(f"Existing implementations: {len(existing_implementations)}")
    print(f"Missing implementations:  {len(missing_implementations)}")
    print(f"Unreferenced files:       {len(unreferenced_implementations)}")
    
    return missing_implementations, unreferenced_implementations

def main():
    """Main entry point"""
    missing, unreferenced = analyze_workflows()
    
    if missing:
        print("\nYou need to create the following implementation files:")
        for method in missing:
            print(f"- {method}.py")
        sys.exit(1)
    
    if unreferenced:
        print("\nConsider cleaning up these unreferenced implementation files:")
        for impl in unreferenced:
            print(f"- {impl}.py")
            
    if not missing and not unreferenced:
        print("\nEverything looks good! All workflows are properly implemented.")
        sys.exit(0)

if __name__ == "__main__":
    main() 