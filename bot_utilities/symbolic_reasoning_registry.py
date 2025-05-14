"""
Symbolic Reasoning Registry

This module provides a registry for symbolic reasoning components.
Each registered reasoner provides deterministic reasoning capabilities
for a specific domain (math, logic, etc.)
"""

from typing import Dict, Any, Optional, List, Callable, Protocol, runtime_checkable
import logging

@runtime_checkable
class SymbolicReasoner(Protocol):
    """Protocol for symbolic reasoning components"""
    
    async def solve_math_problem(self, expression: str) -> Dict[str, Any]:
        """Solve a mathematical problem"""
        ...
        
    async def verify_logical_statement(self, statement: str) -> Dict[str, Any]:
        """Verify a logical statement"""
        ...

class SymbolicReasoningRegistry:
    """Registry for symbolic reasoning components"""
    
    def __init__(self):
        """Initialize an empty registry"""
        self.reasoners: Dict[str, SymbolicReasoner] = {}
        
    def register(self, name: str, reasoner: SymbolicReasoner) -> None:
        """
        Register a new reasoning component
        
        Args:
            name: Unique identifier for the reasoner
            reasoner: The reasoning component to register
        """
        self.reasoners[name] = reasoner
        
    def get_reasoner(self, name: str) -> Optional[SymbolicReasoner]:
        """
        Get a reasoner by name
        
        Args:
            name: The name of the reasoner to retrieve
            
        Returns:
            The requested reasoner, or None if not found
        """
        return self.reasoners.get(name)
        
    def get_all_reasoners(self) -> Dict[str, SymbolicReasoner]:
        """
        Get all registered reasoners
        
        Returns:
            Dictionary of registered reasoners
        """
        return self.reasoners.copy()
        
    def unregister(self, name: str) -> bool:
        """
        Remove a reasoner from the registry
        
        Args:
            name: The name of the reasoner to remove
            
        Returns:
            True if the reasoner was removed, False otherwise
        """
        if name in self.reasoners:
            del self.reasoners[name]
            return True
        return False

# Create a singleton registry instance
symbolic_reasoning_registry = SymbolicReasoningRegistry() 