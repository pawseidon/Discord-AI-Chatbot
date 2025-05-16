"""
Symbolic Reasoning Service

This service provides deterministic symbolic reasoning capabilities including
mathematical problem solving, logical verification, and structured knowledge processing.
It serves as a bridge between neural reasoning in LLMs and precise rule-based computation.
"""

import logging
import re
import asyncio
import traceback
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
import json

# Try to import symbolic reasoning libraries
try:
    import sympy
    from sympy import symbols, sympify, solve, simplify, expand
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    from sympy.logic.boolalg import to_dnf, And, Or, Not, Implies, Equivalent
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    logging.warning("sympy not installed. Symbolic reasoning capabilities will be limited.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("networkx not installed. Graph-based reasoning capabilities will be limited.")

# Import from the symbolic reasoning registry
from ..symbolic_reasoning_registry import SymbolicReasoner, symbolic_reasoning_registry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('symbolic_reasoning_service')

class MathSolver(SymbolicReasoner):
    """
    A symbolic reasoner that solves mathematical problems using SymPy.
    """
    
    def __init__(self):
        """Initialize the math solver"""
        super().__init__()
        self.variable_pattern = re.compile(r'[a-zA-Z]+')
        
    async def solve_math_problem(self, expression: str) -> Dict[str, Any]:
        """
        Solve a mathematical expression or equation symbolically
        
        Args:
            expression: The math expression or equation to solve
            
        Returns:
            Dictionary with results including steps
        """
        if not HAS_SYMPY:
            return {
                "success": False,
                "result": None,
                "steps": [],
                "error": "SymPy is not available. Install with 'pip install sympy'"
            }
            
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Check if this is an equation (contains =)
            if "=" in expression:
                return await self._solve_equation(expression)
            else:
                return await self._evaluate_expression(expression)
                
        except Exception as e:
            logger.error(f"Error solving math problem: {str(e)}")
            return {
                "success": False,
                "result": None,
                "steps": [],
                "error": f"Error: {str(e)}"
            }
            
    async def _evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """Evaluate a mathematical expression"""
        steps = []
        
        try:
            # Parse the expression
            steps.append(f"Parsing expression: {expression}")
            expr = parse_expr(expression, transformations=standard_transformations)
            
            # First try direct evaluation
            try:
                result = float(expr)
                steps.append(f"Direct evaluation: {result}")
                return {
                    "success": True,
                    "result": result,
                    "steps": steps,
                    "error": None
                }
            except:
                # If direct evaluation fails, try to simplify
                steps.append("Cannot directly evaluate, attempting to simplify...")
            
            # Try simplification
            simplified = simplify(expr)
            steps.append(f"Simplified expression: {simplified}")
            
            # Try expansion
            expanded = expand(simplified)
            if expanded != simplified:
                steps.append(f"Expanded form: {expanded}")
            
            # Try numeric evaluation again
            try:
                result = float(expanded)
                steps.append(f"Final evaluation: {result}")
                return {
                    "success": True,
                    "result": result,
                    "steps": steps,
                    "error": None
                }
            except:
                # Return the simplified symbolic result if numeric evaluation isn't possible
                steps.append(f"Final result (symbolic): {expanded}")
                return {
                    "success": True,
                    "result": str(expanded),
                    "steps": steps,
                    "error": None
                }
                
        except Exception as e:
            logger.error(f"Error evaluating expression: {str(e)}")
            return {
                "success": False,
                "result": None,
                "steps": steps,
                "error": f"Error: {str(e)}"
            }
            
    async def _solve_equation(self, equation: str) -> Dict[str, Any]:
        """Solve a mathematical equation"""
        steps = []
        
        try:
            # Split the equation into left and right sides
            steps.append(f"Parsing equation: {equation}")
            left, right = equation.split("=", 1)
            left = left.strip()
            right = right.strip()
            
            # Move everything to the left side
            steps.append(f"Standard form: {left} - ({right}) = 0")
            expr = parse_expr(f"({left}) - ({right})", transformations=standard_transformations)
            
            # Find all variables
            var_matches = self.variable_pattern.findall(equation)
            variables = list(set(var_matches))
            
            if not variables:
                steps.append("No variables found in equation.")
                simplified = simplify(expr)
                steps.append(f"Simplified equation: {simplified} = 0")
                
                # Check if the equation is true (evaluates to 0)
                if simplified == 0:
                    steps.append("Equation is true for all values.")
                    return {
                        "success": True,
                        "result": "Identity (true for all values)",
                        "steps": steps,
                        "error": None
                    }
                else:
                    steps.append("Equation has no solution.")
                    return {
                        "success": True,
                        "result": "No solution",
                        "steps": steps,
                        "error": None
                    }
            
            # For multiple variables, solve for the first one
            primary_var = variables[0]
            steps.append(f"Solving for variable: {primary_var}")
            
            # Create symbolic variables
            sym_vars = [symbols(var) for var in variables]
            var_dict = dict(zip(variables, sym_vars))
            
            # Replace variables in the expression
            symbolic_expr = expr
            
            # Solve the equation
            solutions = solve(symbolic_expr, var_dict[primary_var])
            
            if not solutions:
                steps.append(f"No solutions found for {primary_var}.")
                return {
                    "success": True,
                    "result": "No solution",
                    "steps": steps,
                    "error": None
                }
                
            # Format the solutions
            steps.append(f"Solutions for {primary_var}:")
            for sol in solutions:
                steps.append(f"  {primary_var} = {sol}")
                
            # Return the solutions
            result = {primary_var: [str(sol) for sol in solutions]}
            
            return {
                "success": True,
                "result": result,
                "steps": steps,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error solving equation: {str(e)}")
            return {
                "success": False,
                "result": None,
                "steps": steps,
                "error": f"Error: {str(e)}"
            }
            
    async def verify_logical_statement(self, statement: str) -> Dict[str, Any]:
        """Pass the request to LogicalVerifier"""
        logical_verifier = LogicalVerifier()
        return await logical_verifier.verify_logical_statement(statement)

class LogicalVerifier(SymbolicReasoner):
    """
    A symbolic reasoner that verifies logical statements using rule-based analysis.
    """
    
    def __init__(self):
        """Initialize the logical verifier"""
        super().__init__()
        self.knowledge_base = {}
        
    def add_fact(self, fact: str, truth_value: bool = True) -> None:
        """Add a fact to the knowledge base"""
        self.knowledge_base[fact.strip().lower()] = truth_value
        
    def add_rule(self, condition: str, conclusion: str) -> None:
        """Add a rule (if-then) to the knowledge base"""
        # Rules are stored as a special type of fact
        rule_key = f"RULE:{condition.strip().lower()}->{conclusion.strip().lower()}"
        self.knowledge_base[rule_key] = True
        
    async def verify_logical_statement(self, statement: str) -> Dict[str, Any]:
        """
        Verify a logical statement against the knowledge base
        
        Args:
            statement: The logical statement to verify
            
        Returns:
            Dictionary with verification results
        """
        statement = statement.strip().lower()
        steps = []
        
        try:
            # Direct fact lookup
            if statement in self.knowledge_base:
                truth_value = self.knowledge_base[statement]
                steps.append(f"Direct fact lookup: '{statement}' is {'true' if truth_value else 'false'}")
                return {
                    "success": True,
                    "result": truth_value,
                    "steps": steps,
                    "error": None
                }
                
            # Check for rule-based inferences
            for key, value in self.knowledge_base.items():
                if key.startswith("RULE:") and value:
                    rule_parts = key[5:].split("->")
                    condition = rule_parts[0]
                    conclusion = rule_parts[1]
                    
                    # If the condition is in the knowledge base and true,
                    # and the conclusion matches our statement
                    if (condition in self.knowledge_base and 
                        self.knowledge_base[condition] and
                        conclusion == statement):
                        
                        steps.append(f"Rule application: '{condition}' implies '{conclusion}'")
                        steps.append(f"'{condition}' is known to be true")
                        steps.append(f"Therefore, '{statement}' is true")
                        
                        return {
                            "success": True,
                            "result": True,
                            "steps": steps,
                            "error": None
                        }
            
            # No definitive conclusion
            steps.append(f"No facts or rules found for '{statement}'")
            return {
                "success": True,
                "result": None,
                "steps": steps,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error verifying logical statement: {str(e)}")
            return {
                "success": False,
                "result": None,
                "steps": steps,
                "error": f"Error: {str(e)}"
            }
            
    async def solve_math_problem(self, expression: str) -> Dict[str, Any]:
        """
        Logical verifier doesn't handle math problems directly
        """
        return {
            "success": False,
            "result": None,
            "steps": [],
            "error": "Math problem solving not implemented for the logical verifier"
        }

class GraphAnalyzer(SymbolicReasoner):
    """
    A symbolic reasoner that analyzes graph structures and relationships.
    """
    
    def __init__(self):
        """Initialize the graph analyzer"""
        super().__init__()
        self.graphs = {}
        
    def create_graph(self, graph_id: str) -> None:
        """Create a new graph"""
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available, graph operations will fail")
            return
            
        self.graphs[graph_id] = nx.DiGraph()
        
    def add_node(self, graph_id: str, node_id: str, properties: Dict[str, Any] = None) -> None:
        """Add a node to the graph"""
        if not HAS_NETWORKX or graph_id not in self.graphs:
            return
            
        self.graphs[graph_id].add_node(node_id, **(properties or {}))
        
    def add_edge(self, graph_id: str, from_node: str, to_node: str, 
                 relation: str = None, properties: Dict[str, Any] = None) -> None:
        """Add an edge to the graph"""
        if not HAS_NETWORKX or graph_id not in self.graphs:
            return
            
        edge_attrs = properties or {}
        if relation:
            edge_attrs['relation'] = relation
            
        self.graphs[graph_id].add_edge(from_node, to_node, **edge_attrs)
        
    async def analyze_graph(self, graph_id: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze a graph based on the requested analysis type
        
        Args:
            graph_id: ID of the graph to analyze
            analysis_type: Type of analysis (centrality, paths, etc.)
            
        Returns:
            Dictionary with analysis results
        """
        if not HAS_NETWORKX:
            return {
                "success": False,
                "result": None,
                "error": "NetworkX is not available. Install with 'pip install networkx'"
            }
            
        if graph_id not in self.graphs:
            return {
                "success": False,
                "result": None,
                "error": f"Graph '{graph_id}' not found"
            }
            
        graph = self.graphs[graph_id]
        
        try:
            if analysis_type == "centrality":
                # Calculate centrality measures
                degree_cent = nx.degree_centrality(graph)
                betweenness_cent = nx.betweenness_centrality(graph)
                
                return {
                    "success": True,
                    "result": {
                        "degree_centrality": degree_cent,
                        "betweenness_centrality": betweenness_cent
                    },
                    "error": None
                }
                
            elif analysis_type == "paths":
                # Find shortest paths between all nodes
                paths = {}
                for source in graph.nodes():
                    paths[source] = {}
                    for target in graph.nodes():
                        if source != target:
                            try:
                                path = nx.shortest_path(graph, source, target)
                                paths[source][target] = path
                            except nx.NetworkXNoPath:
                                paths[source][target] = None
                
                return {
                    "success": True,
                    "result": {"paths": paths},
                    "error": None
                }
                
            elif analysis_type == "communities":
                # Convert to undirected for community detection
                undirected = graph.to_undirected()
                
                try:
                    # Try to detect communities
                    from networkx.algorithms import community
                    communities = list(community.greedy_modularity_communities(undirected))
                    result = [list(c) for c in communities]
                    
                    return {
                        "success": True,
                        "result": {"communities": result},
                        "error": None
                    }
                except ImportError:
                    return {
                        "success": False,
                        "result": None,
                        "error": "Community detection requires additional NetworkX packages"
                    }
                    
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Unknown analysis type: {analysis_type}"
                }
                
        except Exception as e:
            logger.error(f"Error analyzing graph: {str(e)}")
            return {
                "success": False,
                "result": None,
                "error": f"Error: {str(e)}"
            }

class SymbolicReasoningService:
    """
    Central service for symbolic reasoning capabilities.
    This service provides methods for solving math problems, verifying logical statements,
    and other symbolic reasoning tasks.
    """
    
    def __init__(self):
        """Initialize the symbolic reasoning service"""
        self.math_solver = None
        self.logical_verifier = None
        self.graph_analyzer = None
        self._initialized = False
        
    async def ensure_initialized(self) -> bool:
        """Initialize required components if not already initialized"""
        if not self._initialized:
            try:
                # Create the reasoners
                self.math_solver = MathSolver()
                self.logical_verifier = LogicalVerifier()
                self.graph_analyzer = GraphAnalyzer() if HAS_NETWORKX else None
                
                # Register the reasoners
                symbolic_reasoning_registry.register("math", self.math_solver)
                symbolic_reasoning_registry.register("logic", self.logical_verifier)
                
                if self.graph_analyzer:
                    symbolic_reasoning_registry.register("graph", self.graph_analyzer)
                
                self._initialized = True
                logger.info("Symbolic reasoning service initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Error initializing symbolic reasoning service: {str(e)}")
                return False
        return True
        
    async def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression or logical statement
        
        Args:
            expression: The expression to evaluate
            
        Returns:
            Dictionary with results including steps
        """
        await self.ensure_initialized()
        
        # Clean the expression
        expression = expression.strip()
        
        # Determine if this is a mathematical or logical expression
        is_logical = any(term in expression.lower() for term in 
                           ["and", "or", "not", "if", "then", "implies", 
                            "equivalent", "true", "false", "→", "∧", "∨", "¬", "↔", "⇔"])
        
        has_math_chars = any(char in expression for char in "+-*/^0123456789()[]{}=<>")
        
        if is_logical and not has_math_chars:
            # Handle as logical expression
            result = await self.verify_logical_statement(expression)
            return result
        else:
            # Handle as mathematical expression
            result = await self.solve_math_problem(expression)
            return result
        
    async def solve_math_problem(self, expression: str) -> Dict[str, Any]:
        """
        Solve a mathematical expression or equation
        
        Args:
            expression: The math expression or equation to solve
            
        Returns:
            Dictionary with results including steps
        """
        await self.ensure_initialized()
        return await self.math_solver.solve_math_problem(expression)
        
    async def verify_logical_statement(self, statement: str) -> Dict[str, Any]:
        """
        Verify a logical statement
        
        Args:
            statement: The logical statement to verify
            
        Returns:
            Dictionary with results including steps
        """
        await self.ensure_initialized()
        return await self.logical_verifier.verify_logical_statement(statement)
        
    async def add_logical_fact(self, fact: str, truth_value: bool = True) -> None:
        """
        Add a logical fact to the knowledge base
        
        Args:
            fact: The fact to add
            truth_value: Whether the fact is true or false
        """
        await self.ensure_initialized()
        self.logical_verifier.add_fact(fact, truth_value)
        
    async def add_logical_rule(self, condition: str, conclusion: str) -> None:
        """
        Add a logical rule to the knowledge base
        
        Args:
            condition: The condition part of the rule
            conclusion: The conclusion part of the rule
        """
        await self.ensure_initialized()
        self.logical_verifier.add_rule(condition, conclusion)
        
    async def create_graph(self, graph_id: str) -> None:
        """
        Create a new graph for analysis
        
        Args:
            graph_id: Identifier for the graph
        """
        await self.ensure_initialized()
        if not self.graph_analyzer:
            raise ValueError("Graph analysis functionality is not available. Install networkx.")
            
        self.graph_analyzer.create_graph(graph_id)
        
    async def add_graph_node(self, graph_id: str, node_id: str, 
                            properties: Dict[str, Any] = None) -> None:
        """
        Add a node to an existing graph
        
        Args:
            graph_id: The graph identifier
            node_id: The node identifier
            properties: Optional node properties
        """
        await self.ensure_initialized()
        if not self.graph_analyzer:
            raise ValueError("Graph analysis functionality is not available. Install networkx.")
            
        self.graph_analyzer.add_node(graph_id, node_id, properties)
        
    async def add_graph_edge(self, graph_id: str, from_node: str, to_node: str,
                            relation: str = None, properties: Dict[str, Any] = None) -> None:
        """
        Add an edge between nodes in a graph
        
        Args:
            graph_id: The graph identifier
            from_node: The source node
            to_node: The target node
            relation: Optional relationship type
            properties: Optional edge properties
        """
        await self.ensure_initialized()
        if not self.graph_analyzer:
            raise ValueError("Graph analysis functionality is not available. Install networkx.")
            
        self.graph_analyzer.add_edge(graph_id, from_node, to_node, relation, properties)
        
    async def analyze_graph(self, graph_id: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze a graph using the specified analysis type
        
        Args:
            graph_id: The graph identifier
            analysis_type: The type of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        await self.ensure_initialized()
        if not self.graph_analyzer:
            raise ValueError("Graph analysis functionality is not available. Install networkx.")
            
        return await self.graph_analyzer.analyze_graph(graph_id, analysis_type)
        
    def get_available_reasoners(self) -> List[str]:
        """
        Get the list of available symbolic reasoners
        
        Returns:
            List of available reasoner names
        """
        return list(symbolic_reasoning_registry.get_all_reasoners().keys())

    async def process_calculation(self, query: str, user_id: str = None, conversation_id: str = None, update_callback: Callable = None) -> Dict[str, Any]:
        """
        Process a calculation request from the agent service
        
        Args:
            query: The mathematical expression or equation to solve
            user_id: The user ID (optional)
            conversation_id: Conversation ID for context (optional)
            update_callback: Optional callback for streaming updates
            
        Returns:
            Dict containing the result and explanation
        """
        try:
            # Ensure the service is initialized
            await self.ensure_initialized()
            
            # Log the calculation request
            logger.info(f"Processing calculation: {query}")
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Performing symbolic calculation..."
                })
            
            # First try to solve as a math problem using solve_math_problem
            math_result = await self.solve_math_problem(query)
            
            if math_result.get("success", False):
                # Format the result for the agent
                steps_text = "\n".join(math_result.get("steps", []))
                result_text = math_result.get("result", "Unknown")
                
                # Notify about completion if callback provided
                if update_callback:
                    await update_callback("thinking", {
                        "thinking": f"Calculation complete: {result_text}"
                    })
                
                return {
                    "response": f"Result: {result_text}",
                    "result": result_text,
                    "thinking": steps_text,
                    "success": True,
                    "reasoning_type": "symbolic"
                }
            else:
                # If solving as a math problem failed, try direct evaluation
                eval_result = await self.evaluate_expression(query)
                
                if eval_result.get("success", False):
                    # Format the result for the agent
                    steps_text = "\n".join(eval_result.get("steps", []))
                    result_text = eval_result.get("result", "Unknown")
                    
                    # Notify about completion if callback provided
                    if update_callback:
                        await update_callback("thinking", {
                            "thinking": f"Calculation complete: {result_text}"
                        })
                    
                    return {
                        "response": f"Result: {result_text}",
                        "result": result_text,
                        "thinking": steps_text,
                        "success": True,
                        "reasoning_type": "symbolic"
                    }
                else:
                    # Both methods failed
                    error = eval_result.get("error", "Unknown error")
                    
                    # Notify about error if callback provided
                    if update_callback:
                        await update_callback("thinking", {
                            "thinking": f"Calculation failed: {error}"
                        })
                    
                    return {
                        "response": f"I couldn't solve this mathematical expression. Error: {error}",
                        "thinking": f"Failed to solve expression: {query}\nError: {error}",
                        "success": False,
                        "reasoning_type": "symbolic"
                    }
                    
        except Exception as e:
            logger.error(f"Error in process_calculation: {str(e)}")
            
            # Notify about error if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": f"Error in calculation: {str(e)}"
                })
            
            return {
                "response": f"I encountered an error while calculating: {str(e)}",
                "thinking": f"Exception in process_calculation: {str(e)}",
                "success": False,
                "reasoning_type": "symbolic"
            }

# Create a singleton instance for global access
symbolic_reasoning_service = SymbolicReasoningService() 