# HMAS Reasoning Module

## Overview
The Reasoning Module provides knowledge representation and reasoning capabilities for the Hierarchical Multi-Agent System, including symbolic reasoning, causal reasoning, and common-sense reasoning.

## Features
- Knowledge Graph Management
- Symbolic Reasoning
- Causal Reasoning
- Common-Sense Reasoning
- Rule-Based Inference
- Knowledge Representation

## Configuration
The module can be configured through environment variables and the `config.py` file:

```bash
INFERENCE_DEPTH=3
SYMBOLIC_REASONING_ENABLED=true
CAUSAL_REASONING_ENABLED=true
```

## API Endpoints
- `POST /knowledge/add_node`: Add a node to the knowledge graph
- `POST /knowledge/add_edge`: Add an edge to the knowledge graph
- `POST /symbolic/add_rule`: Add a symbolic reasoning rule
- `POST /causal/add_relation`: Add a causal relation
- `POST /common_sense/add_rule`: Add a common-sense rule
- `POST /reason/query`: Query the knowledge base

## Dependencies
- NetworkX
- RDFLib
- FastAPI
- NumPy
- Pandas

## Usage
```python
from reasoning import KnowledgeGraphManager, SymbolicReasoner

# Create knowledge graph
kg_manager = KnowledgeGraphManager()
kg_manager.add_node("concept1", {"type": "entity", "properties": {...}})
kg_manager.add_edge("concept1", "concept2", "related_to")

# Add reasoning rules
symbolic_reasoner = SymbolicReasoner()
symbolic_reasoner.add_rule({
    "premise": ["A", "B"],
    "conclusion": "C",
    "confidence": 0.9
})
```

## Reasoning Strategies
1. Symbolic Reasoning: Rule-based inference
2. Causal Reasoning: Cause-effect relationships
3. Common-Sense Reasoning: Domain-specific knowledge
4. Knowledge Graph: Structured representation

## Contributing
See the main project's contribution guidelines.

## License
Same as the main project. 