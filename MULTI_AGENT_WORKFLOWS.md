# Multi-Agent Workflow Implementation

This document outlines the implementation of different workflow combinations for the Discord AI Bot's multi-agent reasoning system.

## Workflow Types

The bot supports the following specialized workflow combinations:

### 1. Sequential + RAG Workflow
- **Use Case**: Educational content, complex explanations, "how" and "why" questions
- **Pattern**: `sequential_rag`
- **Process**:
  1. RAG retrieves relevant factual information
  2. Sequential thinking organizes and explains the information step-by-step
  3. Results are combined into a coherent, structured response

### 2. Verification + RAG + Sequential Workflow
- **Use Case**: Fact-checking, current events, contentious topics
- **Pattern**: `verification_rag`
- **Process**:
  1. RAG retrieves information from multiple sources
  2. Verification checks source reliability and cross-references facts
  3. Sequential thinking synthesizes verified information
  4. Results are presented with confidence ratings

### 3. Calculation + Sequential Workflow
- **Use Case**: Mathematical problems, statistical analysis, numerical reasoning
- **Pattern**: `calculation_sequential`
- **Process**:
  1. Calculation agent performs precise mathematical operations
  2. Sequential thinking explains steps and reasoning
  3. Results are presented with the calculation steps

### 4. Creative + Sequential Workflow
- **Use Case**: Storytelling, content creation, structured creative output
- **Pattern**: `creative_sequential`
- **Process**:
  1. Creative agent generates imaginative content
  2. Sequential thinking provides structure and logical flow
  3. Results combine creativity with coherent organization

### 5. Graph + RAG + Verification Workflow
- **Use Case**: Relationship mapping, network analysis, complex systems understanding
- **Pattern**: `graph_rag_verification`
- **Process**:
  1. RAG retrieves information about entities and relationships
  2. Graph agent builds and analyzes relationship model
  3. Verification ensures accuracy of connections
  4. Results visualize relationships with confidence ratings

### 6. Multi-Agent General Workflow
- **Use Case**: Complex queries that don't fit other patterns
- **Pattern**: `multi_agent`
- **Process**:
  1. Orchestrator plans the approach using appropriate reasoning types
  2. Multiple agents collaborate on different aspects of the query
  3. Results are synthesized into a cohesive response

## Workflow Detection

The system uses pattern recognition to detect which workflow is most appropriate for a given query:

```python
async def detect_workflow_type(self, query: str, conversation_id: str = None) -> str:
    # First check for calculation + sequential workflow
    if re.search(r'(calculate|compute|solve|equation|formula|math problem)', query, re.IGNORECASE):
        has_explanation_request = re.search(r'(explain|show steps|show work|why|how)', query, re.IGNORECASE)
        if has_explanation_request:
            return "calculation_sequential"
    
    # Check for verification + RAG workflow
    factual_verification_pattern = r'(verify|fact check|is it true|confirm|evidence for|sources for|research on|current events)'
    if re.search(factual_verification_pattern, query, re.IGNORECASE):
        return "verification_rag"
        
    # Check for creative + sequential workflow
    creative_pattern = r'(write|create|generate|story|poem|creative|imagine|fiction)'
    explanation_pattern = r'(explain|analyze|outline|structure|organize|plan)'
    if re.search(creative_pattern, query, re.IGNORECASE) and re.search(explanation_pattern, query, re.IGNORECASE):
        return "creative_sequential"
        
    # Check for graph + RAG + verification workflow
    graph_pattern = r'(relationship|network|connect|graph|diagram|map the|connections between)'
    if re.search(graph_pattern, query, re.IGNORECASE):
        return "graph_rag_verification"
        
    # Default to sequential + RAG for educational/explanatory content
    educational_pattern = r'(what is|how does|explain|describe|why does|how do|what are|definition of)'
    if re.search(educational_pattern, query, re.IGNORECASE):
        return "sequential_rag"
        
    # Multi-agent is the most general workflow
    return "multi_agent"
```

## Reasoning Type Detection

The system identifies which reasoning types apply to a query using pattern matching:

```python
async def detect_multiple_reasoning_types(self, query: str, conversation_id: str = None, max_types: int = 3) -> List[str]:
    # Define patterns for different reasoning types
    patterns = {
        "sequential": r"(step[s]?[ -]by[ -]step|logical|explain why|explain how|think through|break down|walkthrough|reasoning|analysis)",
        "rag": r"(information|research|look up|find out|search for|latest|recent|news|article|data)",
        "verification": r"(verify|fact check|is it true|confirm|evidence|proof|reliable|ensure|validate)",
        "calculation": r"(calculate|compute|solve|equation|formula|math|add|multiply|divide|subtract|percentage|formula)",
        "creative": r"(creative|story|poem|imagine|pretend|fiction|narrative|write a|generate a)",
        "graph": r"(relationship|network|connect|graph|diagram|map the|connections between|linked|association)"
    }
    
    # Check each pattern against the query and rank by relevance
    matches = {}
    for reasoning_type, pattern in patterns.items():
        match_count = len(re.findall(pattern, query, re.IGNORECASE))
        if match_count > 0:
            matches[reasoning_type] = match_count
            
    # Sort by relevance and return top results
    sorted_types = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    result = [reasoning_type for reasoning_type, _ in sorted_types[:max_types]]
    
    return result or ["conversational"]
```

## Combining Reasoning Types

The system decides whether to combine reasoning types based on query complexity and reasoning compatibility:

```python
async def should_combine_reasoning(self, query: str, conversation_id: str = None) -> bool:
    # Detect multiple reasoning types
    reasoning_types = await self.detect_multiple_reasoning_types(query, conversation_id, max_types=2)
    
    # Only consider combining if we have at least 2 types
    if len(reasoning_types) < 2:
        return False
    
    # Define effective combinations
    effective_combinations = [
        {"sequential", "rag"},
        {"verification", "rag"},
        {"calculation", "sequential"},
        {"creative", "sequential"},
        {"graph", "rag"},
        {"graph", "verification"}
    ]
    
    # Check if detected types form an effective combination
    detected_set = set(reasoning_types[:2])
    for combination in effective_combinations:
        if detected_set.issubset(combination) or combination.issubset(detected_set):
            return True
            
    # For complex queries with multiple keywords, prefer combination
    complex_patterns = [
        r"(why.*how)",
        r"(calculate.*explain)",
        r"(verify.*explain)",
        r"(creative.*structure)",
        r"(search.*analyze)",
        r"(relationship.*data)"
    ]
    
    for pattern in complex_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
            
    return False
```

## Workflow Processing

When processing a query, the system:

1. Detects the most appropriate workflow type
2. Uses the workflow_service to handle the query with the selected workflow
3. Applies the appropriate combination of reasoning types
4. Manages transitions between reasoning stages with callback updates
5. Formats the final response with appropriate styling

Example workflow entry point:

```python
async def process_with_workflow(self, 
                               query: str,
                               user_id: str,
                               conversation_id: str = None,
                               workflow_type: str = None,
                               update_callback: Callable = None) -> str:
    # Auto-detect workflow type if not specified
    if not workflow_type:
        workflow_type = await self.detect_workflow_type(query, conversation_id)
        
    # Call the appropriate workflow handler
    if workflow_type in self.workflows:
        workflow_handler = self.workflows[workflow_type]
        return await workflow_handler(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            update_callback=update_callback
        )
    else:
        # Default to multi-agent workflow
        return await self.multi_agent_workflow(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            update_callback=update_callback
        )
```

## Testing

To test the multi-agent workflow implementation, run:

```bash
python test_multi_agent.py
```

This will test:
- Workflow detection
- Reasoning type detection
- Combining reasoning types
- Processing with different workflows

## Example Queries

Here are examples of queries that trigger each workflow:

- **Sequential + RAG**: "How does nuclear fusion work?"
- **Verification + RAG**: "Verify if the Earth is 4.5 billion years old."
- **Calculation + Sequential**: "Calculate the area of a circle with radius 5cm and explain the steps."
- **Creative + Sequential**: "Write a story about an enchanted forest with a clear beginning, middle, and end."
- **Graph + RAG + Verification**: "Map the relationship between tech companies in Silicon Valley."
- **Multi-Agent**: "What's the best approach to solve climate change considering economic, social, and technological factors?" 