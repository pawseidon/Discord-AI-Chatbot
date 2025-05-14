"""
Symbolic Reasoning Service

This service provides deterministic symbolic reasoning capabilities including
mathematical problem solving, logical verification, and structured knowledge processing.
It serves as a bridge between neural reasoning in LLMs and precise rule-based computation.
"""

import logging
import re
import asyncio
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
            solutions = solve(symbolic_expr, sym_vars[0])
            
            if not solutions:
                steps.append("No solutions found.")
                return {
                    "success": True,
                    "result": "No solution",
                    "steps": steps,
                    "error": None
                }
                
            # Format the solutions
            solution_strs = [f"{primary_var} = {sol}" for sol in solutions]
            steps.append(f"Solutions: {', '.join(solution_strs)}")
            
            return {
                "success": True,
                "result": solution_strs,
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
        """
        Verify a logical statement
        
        This is not directly handled by the math solver but could be
        implemented for certain types of mathematical statements.
        """
        return {
            "success": False,
            "result": None,
            "error": "Logical statement verification not implemented for the math solver"
        }

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
    Service that provides access to various symbolic reasoning capabilities.
    """
    
    def __init__(self):
        """Initialize the service"""
        self.initialized = False
        self.solvers = {}
        self.graph_cache = {}
        
    async def ensure_initialized(self) -> bool:
        """
        Ensure the service is initialized
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        if self.initialized:
            return True
            
        try:
            if HAS_SYMPY:
                # Register standard solvers
                self.solvers["algebra"] = self._solve_algebraic
                self.solvers["calculus"] = self._solve_calculus
                self.solvers["equation"] = self._solve_equation
                self.solvers["logic"] = self._solve_logic
                
                # Register the available reasoners
                math_solver = MathSolver()
                symbolic_reasoning_registry.register_reasoner("math", math_solver)
                
                logical_verifier = LogicalVerifier()
                symbolic_reasoning_registry.register_reasoner("logic", logical_verifier)
                
                graph_analyzer = GraphAnalyzer()
                symbolic_reasoning_registry.register_reasoner("graph", graph_analyzer)
                
                self.initialized = True
                logger.info("Symbolic reasoning service initialized")
                return True
            else:
                logging.warning("Symbolic reasoning service initialization failed: sympy not available")
                return False
        except Exception as e:
            logging.error(f"Error initializing SymbolicReasoningService: {e}")
            return False
        
    async def solve_math_problem(self, expression: str) -> Dict[str, Any]:
        """
        Solve a mathematical problem
        
        Args:
            expression: The mathematical expression or equation to solve
            
        Returns:
            Dictionary with results and steps
        """
        await self.ensure_initialized()
        math_solver = symbolic_reasoning_registry.get_reasoner("math")
        return await math_solver.solve_math_problem(expression)
        
    async def verify_logical_statement(self, statement: str) -> Dict[str, Any]:
        """
        Verify a logical statement
        
        Args:
            statement: The logical statement to verify
            
        Returns:
            Dictionary with verification results
        """
        await self.ensure_initialized()
        logical_verifier = symbolic_reasoning_registry.get_reasoner("logic")
        return await logical_verifier.verify_logical_statement(statement)
        
    async def add_logical_fact(self, fact: str, truth_value: bool = True) -> None:
        """
        Add a fact to the logical knowledge base
        
        Args:
            fact: The fact to add
            truth_value: Whether the fact is true or false
        """
        await self.ensure_initialized()
        logical_verifier = symbolic_reasoning_registry.get_reasoner("logic")
        logical_verifier.add_fact(fact, truth_value)
        
    async def add_logical_rule(self, condition: str, conclusion: str) -> None:
        """
        Add a rule to the logical knowledge base
        
        Args:
            condition: The condition part of the rule
            conclusion: The conclusion part of the rule
        """
        await self.ensure_initialized()
        logical_verifier = symbolic_reasoning_registry.get_reasoner("logic")
        logical_verifier.add_rule(condition, conclusion)
        
    async def create_graph(self, graph_id: str) -> None:
        """
        Create a new graph for analysis
        
        Args:
            graph_id: Identifier for the graph
        """
        await self.ensure_initialized()
        graph_analyzer = symbolic_reasoning_registry.get_reasoner("graph")
        graph_analyzer.create_graph(graph_id)
        
    async def add_graph_node(self, graph_id: str, node_id: str, 
                             properties: Dict[str, Any] = None) -> None:
        """
        Add a node to a graph
        
        Args:
            graph_id: Identifier for the graph
            node_id: Identifier for the node
            properties: Optional node properties
        """
        await self.ensure_initialized()
        graph_analyzer = symbolic_reasoning_registry.get_reasoner("graph")
        graph_analyzer.add_node(graph_id, node_id, properties)
        
    async def add_graph_edge(self, graph_id: str, from_node: str, to_node: str,
                             relation: str = None, properties: Dict[str, Any] = None) -> None:
        """
        Add an edge to a graph
        
        Args:
            graph_id: Identifier for the graph
            from_node: Source node
            to_node: Target node
            relation: Optional relation type
            properties: Optional edge properties
        """
        await self.ensure_initialized()
        graph_analyzer = symbolic_reasoning_registry.get_reasoner("graph")
        graph_analyzer.add_edge(graph_id, from_node, to_node, relation, properties)
        
    async def analyze_graph(self, graph_id: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze a graph
        
        Args:
            graph_id: Identifier for the graph
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        await self.ensure_initialized()
        graph_analyzer = symbolic_reasoning_registry.get_reasoner("graph")
        return await graph_analyzer.analyze_graph(graph_id, analysis_type)
        
    def get_available_reasoners(self) -> List[str]:
        """
        Get a list of available symbolic reasoners
        
        Returns:
            List of reasoner names
        """
        return symbolic_reasoning_registry.list_reasoners()

    async def _solve_algebraic(self, problem: str) -> Tuple[str, List[str]]:
        """Solve algebraic problems"""
        steps = []
        steps.append(f"Starting with the algebraic expression: {problem}")
        
        try:
            # Try to parse the expression
            expr = parse_expr(problem, transformations=standard_transformations)
            steps.append(f"Parsed expression: {expr}")
            
            # Simplify the expression
            simplified = simplify(expr)
            if simplified != expr:
                steps.append(f"Simplified: {simplified}")
            
            # Expand the expression
            expanded = expand(simplified)
            if expanded != simplified:
                steps.append(f"Expanded: {expanded}")
                
            return str(expanded), steps
        except Exception as e:
            steps.append(f"Error: {str(e)}")
            return "Could not solve algebraic problem", steps
    
    async def _solve_equation(self, problem: str) -> Tuple[str, List[str]]:
        """Solve equations"""
        steps = []
        steps.append(f"Starting with the equation: {problem}")
        
        try:
            # Check if this is directly a "solve for x" type problem
            if "solve" in problem.lower():
                # Extract the equation part
                equation_match = re.search(r"solve\s+(.+?)\s+for\s+(.+)", problem, re.IGNORECASE)
                if equation_match:
                    equation_str = equation_match.group(1)
                    variable_str = equation_match.group(2)
                    steps.append(f"Extracted equation: {equation_str}")
                    steps.append(f"Solving for variable: {variable_str}")
                    
                    # Check if the equation contains =
                    if "=" in equation_str:
                        left, right = equation_str.split("=", 1)
                        left_expr = parse_expr(left.strip(), transformations=standard_transformations)
                        right_expr = parse_expr(right.strip(), transformations=standard_transformations)
                        equation = left_expr - right_expr
                    else:
                        # Assume it's set to 0
                        equation = parse_expr(equation_str, transformations=standard_transformations)
                    
                    # Solve the equation
                    variable = symbols(variable_str.strip())
                    solution = solve(equation, variable)
                    steps.append(f"Solution: {variable} = {solution}")
                    
                    return str(solution), steps
            
            # Otherwise, try to find the = sign
            if "=" in problem:
                left, right = problem.split("=", 1)
                left_expr = parse_expr(left.strip(), transformations=standard_transformations)
                right_expr = parse_expr(right.strip(), transformations=standard_transformations)
                steps.append(f"Left side: {left_expr}")
                steps.append(f"Right side: {right_expr}")
                
                # Move everything to one side
                equation = left_expr - right_expr
                steps.append(f"Equation in standard form: {equation} = 0")
                
                # Try to identify the variable
                symbols_used = list(equation.free_symbols)
                if len(symbols_used) == 0:
                    # No variables, just calculate
                    result = equation == 0
                    steps.append(f"This is a constant equation: {result}")
                    return str(result), steps
                elif len(symbols_used) == 1:
                    # One variable, solve for it
                    variable = symbols_used[0]
                    steps.append(f"Solving for variable: {variable}")
                    solution = solve(equation, variable)
                    steps.append(f"Solution: {variable} = {solution}")
                    return str(solution), steps
                else:
                    # Multiple variables, we need context
                    steps.append(f"Multiple variables found: {symbols_used}")
                    steps.append("Assuming we should solve for the first variable")
                    variable = symbols_used[0]
                    solution = solve(equation, variable)
                    steps.append(f"Solution: {variable} = {solution}")
                    return str(solution), steps
            
            # If we can't identify an equation or solve directive
            return "Could not identify equation structure", steps
            
        except Exception as e:
            steps.append(f"Error: {str(e)}")
            return "Could not solve equation", steps
    
    async def _solve_calculus(self, problem: str) -> Tuple[str, List[str]]:
        """Solve calculus problems"""
        steps = []
        steps.append(f"Starting with the calculus problem: {problem}")
        
        try:
            from sympy import Symbol, Derivative, Integral
            
            # Check if it's a derivative problem
            if any(word in problem.lower() for word in ["derivative", "differentiate"]):
                # Try to extract the function and variable
                derivative_match = re.search(r"derivative\s+of\s+(.+?)\s+with\s+respect\s+to\s+(.+)", problem, re.IGNORECASE)
                if not derivative_match:
                    derivative_match = re.search(r"differentiate\s+(.+?)\s+with\s+respect\s+to\s+(.+)", problem, re.IGNORECASE)
                
                if derivative_match:
                    function_str = derivative_match.group(1)
                    variable_str = derivative_match.group(2)
                    steps.append(f"Extracting function: {function_str}")
                    steps.append(f"Differentiating with respect to: {variable_str}")
                    
                    # Parse the function and variable
                    function = parse_expr(function_str, transformations=standard_transformations)
                    variable = Symbol(variable_str.strip())
                    
                    # Compute the derivative
                    derivative = Derivative(function, variable).doit()
                    steps.append(f"Computed derivative: {derivative}")
                    
                    # Simplify the result
                    simplified = simplify(derivative)
                    if simplified != derivative:
                        steps.append(f"Simplified: {simplified}")
                        
                    return str(simplified), steps
            
            # Check if it's an integral problem
            if any(word in problem.lower() for word in ["integral", "integrate"]):
                # Try to extract the function and variable
                integral_match = re.search(r"integral\s+of\s+(.+?)\s+with\s+respect\s+to\s+(.+)", problem, re.IGNORECASE)
                if not integral_match:
                    integral_match = re.search(r"integrate\s+(.+?)\s+with\s+respect\s+to\s+(.+)", problem, re.IGNORECASE)
                
                if integral_match:
                    function_str = integral_match.group(1)
                    variable_str = integral_match.group(2)
                    steps.append(f"Extracting function: {function_str}")
                    steps.append(f"Integrating with respect to: {variable_str}")
                    
                    # Parse the function and variable
                    function = parse_expr(function_str, transformations=standard_transformations)
                    variable = Symbol(variable_str.strip())
                    
                    # Compute the integral
                    integral = Integral(function, variable).doit()
                    steps.append(f"Computed integral: {integral}")
                    
                    # Simplify the result
                    simplified = simplify(integral)
                    if simplified != integral:
                        steps.append(f"Simplified: {simplified}")
                        
                    return str(simplified), steps
            
            return "Could not identify calculus problem structure", steps
            
        except Exception as e:
            steps.append(f"Error: {str(e)}")
            return "Could not solve calculus problem", steps
    
    async def _solve_logic(self, problem: str) -> Tuple[bool, List[str]]:
        """Solve logical problems"""
        # This is a simple implementation - can be expanded later
        return await self.verify_logical_statement(problem)
    
    async def _solve_generic(self, problem: str) -> Tuple[str, List[str]]:
        """Generic fallback solver"""
        steps = []
        steps.append(f"Analyzing problem: {problem}")
        
        try:
            # Try to parse as a direct expression
            expr = parse_expr(problem, transformations=standard_transformations)
            steps.append(f"Parsed as mathematical expression: {expr}")
            
            # Simplify the expression
            simplified = simplify(expr)
            steps.append(f"Simplified: {simplified}")
            
            # Try to evaluate if it's a numerical expression
            try:
                evald = float(simplified)
                steps.append(f"Evaluated to: {evald}")
                return str(evald), steps
            except:
                # Not directly evaluable, return simplified form
                return str(simplified), steps
                
        except Exception as e:
            steps.append(f"Could not parse as direct expression: {str(e)}")
            steps.append("Trying alternate approaches...")
            
            # Try different solvers
            for solver_name, solver in self.solvers.items():
                if solver_name != "generic":  # Avoid recursion
                    steps.append(f"Trying {solver_name} solver...")
                    try:
                        result, solver_steps = await solver(problem)
                        steps.extend(solver_steps)
                        return result, steps
                    except Exception as solver_e:
                        steps.append(f"Solver failed: {str(solver_e)}")
            
            # If all solvers fail
            return "Could not solve problem", steps
    
    def _normalize_logical_statement(self, statement: str) -> str:
        """Normalize a logical statement for parsing"""
        # Replace common text with symbols
        replacements = {
            "and": "&",
            "or": "|",
            "not": "~",
            "implies": ">>",
            "if and only if": "<->",
            "iff": "<->",
            "equivalent to": "<->",
            "if": ">>",
            "then": ""  # Remove "then" as it's implied by "if"
        }
        
        normalized = statement.lower()
        for word, symbol in replacements.items():
            normalized = re.sub(r'\b' + word + r'\b', symbol, normalized)
        
        # Handle other replacements
        normalized = normalized.replace("=>", ">>")
        normalized = normalized.replace("->", ">>")
        normalized = normalized.replace("<=>", "<->")
        
        return normalized
    
    def _parse_logical_expression(self, statement: str) -> Any:
        """Parse a logical expression into a sympy form"""
        from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent
        from sympy import Symbol
        
        # Extract variable names
        var_names = set(re.findall(r'\b([a-zA-Z])\b', statement))
        variables = {name: Symbol(name) for name in var_names}
        
        # Replace operators with Python operators that sympy can parse
        statement = statement.replace("&", " and ")
        statement = statement.replace("|", " or ")
        statement = statement.replace("~", " not ")
        statement = statement.replace(">>", " >> ")
        statement = statement.replace("<->", " <-> ")
        
        # Create local namespace with variables and operators
        namespace = {
            "Symbol": Symbol,
            "And": And,
            "Or": Or,
            "Not": Not,
            "Implies": Implies,
            "Equivalent": Equivalent
        }
        namespace.update(variables)
        
        # Convert operators to sympy functions
        statement = statement.replace(" and ", " & ")
        statement = statement.replace(" or ", " | ")
        statement = statement.replace(" not ", "~")
        statement = statement.replace(" >> ", " >> ")
        statement = statement.replace(" <-> ", " <-> ")
        
        # Parse with a custom approach or using sympify for simple cases
        try:
            return sympify(statement, locals=namespace)
        except:
            # More complex parsing would be implemented here
            raise ValueError(f"Could not parse logical expression: {statement}")
    
    async def _verify_logical(self, expr, statement: str) -> Tuple[bool, List[str]]:
        """Verify a logical expression"""
        from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent, to_cnf, is_cnf
        from sympy.logic.inference import satisfiable
        
        steps = []
        steps.append(f"Analyzing logical statement: {statement}")
        
        try:
            # Try to simplify to canonical form
            steps.append(f"Converting to canonical form...")
            cnf_form = to_cnf(expr)
            steps.append(f"Canonical form: {cnf_form}")
            
            # Check satisfiability
            is_sat = satisfiable(expr)
            if is_sat:
                steps.append(f"Statement is satisfiable with values: {is_sat}")
                
                # Check if it's a tautology (always true)
                not_expr = Not(expr)
                is_not_sat = satisfiable(not_expr)
                if not is_not_sat:
                    steps.append("Statement is a tautology (always true)")
                    return True, steps
                else:
                    steps.append(f"Statement is not a tautology. It is false when: {is_not_sat}")
                    # This is a contingency (sometimes true, sometimes false)
                    return None, steps
            else:
                steps.append("Statement is not satisfiable (always false)")
                return False, steps
                
        except Exception as e:
            steps.append(f"Error in verification: {str(e)}")
            return None, steps

# Create a singleton instance
symbolic_reasoning_service = SymbolicReasoningService() 