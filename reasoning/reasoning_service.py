from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Set
import networkx as nx
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDF, RDFS, OWL
import json
from datetime import datetime
import os
import logging
from functools import lru_cache
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
from contextlib import asynccontextmanager
from enum import Enum
import asyncio
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
import spacy
from sympy import Symbol, solve, Eq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with lifespan
app = FastAPI(title="HMAS Reasoning Service")

# Define custom namespace for our ontology
HMAS = Namespace("http://hmas.org/ontology/")

class KnowledgeGraph(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    properties: Dict[str, Any]

class ReasoningRule(BaseModel):
    premise: List[str]
    conclusion: str
    confidence: float
    context: Optional[Dict[str, Any]] = None

class CausalRelation(BaseModel):
    cause: str
    effect: str
    strength: float
    context: Dict[str, Any]
    temporal_delay: Optional[float] = None

class CommonSenseRule(BaseModel):
    domain: str
    rule: str
    exceptions: List[str]
    confidence: float
    priority: Optional[int] = 1

class ReasoningConfig(BaseModel):
    symbolic_reasoning_enabled: bool = True
    causal_reasoning_enabled: bool = True
    common_sense_enabled: bool = True
    inference_depth: int = 3
    confidence_threshold: float = 0.7
    max_branching_factor: int = 10

class ReasoningType(str, Enum):
    SYMBOLIC = "symbolic"
    CAUSAL = "causal"
    COMMON_SENSE = "common_sense"
    PROBABILISTIC = "probabilistic"

class KnowledgeType(str, Enum):
    FACT = "fact"
    RULE = "rule"
    RELATION = "relation"
    CONCEPT = "concept"

class ReasoningRequest(BaseModel):
    reasoning_type: ReasoningType
    query: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[List[Dict[str, Any]]] = None

class ReasoningResponse(BaseModel):
    result: Any
    confidence: float
    explanation: Optional[str] = None
    supporting_facts: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Knowledge Graph Management
class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.rdf_graph = Graph()
        self._load_ontology()
        
    def _load_ontology(self):
        """Load the base ontology."""
        # Add core concepts
        self.rdf_graph.bind("hmas", HMAS)
        self.rdf_graph.add((HMAS.Agent, RDF.type, OWL.Class))
        self.rdf_graph.add((HMAS.Task, RDF.type, OWL.Class))
        self.rdf_graph.add((HMAS.Resource, RDF.type, OWL.Class))
        
    def add_node(self, node_id: str, properties: Dict[str, Any]):
        """Add a node with properties to both graphs."""
        # Add to NetworkX graph
        self.graph.add_node(node_id, **properties)
        
        # Add to RDF graph
        node_uri = URIRef(HMAS[node_id])
        for key, value in properties.items():
            pred_uri = URIRef(HMAS[key])
            self.rdf_graph.add((node_uri, pred_uri, Literal(value)))
            
    def add_edge(self, source: str, target: str, relation: str):
        """Add an edge to both graphs."""
        # Add to NetworkX graph
        self.graph.add_edge(source, target, relation=relation)
        
        # Add to RDF graph
        source_uri = URIRef(HMAS[source])
        target_uri = URIRef(HMAS[target])
        relation_uri = URIRef(HMAS[relation])
        self.rdf_graph.add((source_uri, relation_uri, target_uri))
        
    def query_graph(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the knowledge graph using both graph representations."""
        results = []
        
        # Process query parameters
        node_type = query.get("type")
        properties = query.get("properties", {})
        relations = query.get("relations", [])
        
        # Query NetworkX graph for basic pattern matching
        candidates = set(self.graph.nodes())
        
        # Filter by properties
        for key, value in properties.items():
            candidates = {
                node for node in candidates
                if self.graph.nodes[node].get(key) == value
            }
            
        # Filter by relations
        for relation in relations:
            rel_type = relation["type"]
            target_type = relation.get("target_type")
            
            if relation.get("direction") == "incoming":
                neighbors_func = self.graph.predecessors
            else:
                neighbors_func = self.graph.successors
                
            filtered = set()
            for node in candidates:
                for neighbor in neighbors_func(node):
                    edge_data = self.graph.edges[node, neighbor]
                    if edge_data["relation"] == rel_type:
                        if not target_type or self.graph.nodes[neighbor].get("type") == target_type:
                            filtered.add(node)
                            break
            candidates = filtered
            
        # Convert results
        for node in candidates:
            node_data = self.graph.nodes[node]
            results.append({
                "id": node,
                "properties": node_data,
                "relations": [
                    {
                        "target": target,
                        "type": self.graph.edges[node, target]["relation"]
                    }
                    for target in self.graph.successors(node)
                ]
            })
            
        return results

# Symbolic Reasoning
class SymbolicReasoner:
    def __init__(self):
        self.rules = []
        self.inference_cache = {}
        
    def add_rule(self, rule: ReasoningRule):
        """Add a reasoning rule with validation."""
        # Validate rule structure
        if not rule.premise or not rule.conclusion:
            raise ValueError("Rule must have both premise and conclusion")
            
        if rule.confidence < 0 or rule.confidence > 1:
            raise ValueError("Rule confidence must be between 0 and 1")
            
        self.rules.append(rule)
        # Clear cache when rules change
        self.inference_cache.clear()
        
    @lru_cache(maxsize=1000)
    def apply_rules(self, facts: List[str]) -> List[str]:
        """Apply reasoning rules to derive new facts."""
        facts_set = set(facts)
        new_facts = set()
        changed = True
        depth = 0
        max_depth = 10  # Prevent infinite loops
        
        while changed and depth < max_depth:
            changed = False
            depth += 1
            
            for rule in self.rules:
                # Check if all premises are satisfied
                if all(p in facts_set for p in rule.premise):
                    if rule.conclusion not in facts_set:
                        new_facts.add((rule.conclusion, rule.confidence))
                        changed = True
                        
        # Convert fact-confidence pairs to sorted list
        return sorted(
            [(fact, conf) for fact, conf in new_facts],
            key=lambda x: x[1],
            reverse=True
        )

# Causal Reasoning
class CausalReasoner:
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.temporal_cache = {}
        
    def add_relation(self, relation: CausalRelation):
        """Add a causal relation to the graph."""
        self.causal_graph.add_edge(
            relation.cause,
            relation.effect,
            weight=relation.strength,
            context=relation.context,
            temporal_delay=relation.temporal_delay
        )
        
    def find_causal_chain(
        self,
        start: str,
        end: str,
        min_strength: float = 0.5
    ) -> List[List[str]]:
        """Find causal chains between events."""
        paths = []
        
        try:
            # Find all simple paths
            all_paths = nx.all_simple_paths(
                self.causal_graph,
                start,
                end,
                cutoff=5  # Limit path length
            )
            
            for path in all_paths:
                # Calculate path strength
                path_strength = 1.0
                for i in range(len(path) - 1):
                    edge_data = self.causal_graph.edges[path[i], path[i + 1]]
                    path_strength *= edge_data["weight"]
                    
                if path_strength >= min_strength:
                    paths.append({
                        "path": path,
                        "strength": path_strength,
                        "temporal_sequence": [
                            self.causal_graph.edges[path[i], path[i + 1]].get(
                                "temporal_delay", 0
                            )
                            for i in range(len(path) - 1)
                        ]
                    })
                    
        except nx.NetworkXNoPath:
            logger.warning(f"No causal path found between {start} and {end}")
            
        return sorted(paths, key=lambda x: x["strength"], reverse=True)
        
    def predict_effects(
        self,
        cause: str,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Predict potential effects of a cause."""
        if cause not in self.causal_graph:
            return []
            
        effects = []
        for _, effect in self.causal_graph.edges(cause):
            edge_data = self.causal_graph.edges[cause, effect]
            if edge_data["weight"] >= min_confidence:
                effects.append({
                    "effect": effect,
                    "confidence": edge_data["weight"],
                    "context": edge_data["context"],
                    "temporal_delay": edge_data.get("temporal_delay", 0)
                })
                
        return sorted(effects, key=lambda x: x["confidence"], reverse=True)
        
    def find_common_causes(
        self,
        effects: List[str],
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find common causes for multiple effects."""
        if not effects or len(effects) < 2:
            return []
            
        # Get predecessors for each effect
        predecessors = [
            set(self.causal_graph.predecessors(effect))
            for effect in effects
        ]
        
        # Find common causes
        common_causes = set.intersection(*predecessors)
        
        results = []
        for cause in common_causes:
            # Calculate minimum confidence across all effect links
            min_conf = min(
                self.causal_graph.edges[cause, effect]["weight"]
                for effect in effects
            )
            
            if min_conf >= min_confidence:
                results.append({
                    "cause": cause,
                    "confidence": min_conf,
                    "effects": effects,
                    "contexts": [
                        self.causal_graph.edges[cause, effect]["context"]
                        for effect in effects
                    ]
                })
                
        return sorted(results, key=lambda x: x["confidence"], reverse=True)

# Common Sense Reasoning
class CommonSenseReasoner:
    def __init__(self):
        self.rules = defaultdict(list)
        self.rule_priorities = {}
        self.nlp = spacy.load("en_core_web_sm")
        
    def add_rule(self, rule: CommonSenseRule):
        """Add a common sense rule."""
        self.rules[rule.domain].append(rule)
        self.rule_priorities[(rule.domain, rule.rule)] = rule.priority
        
    def get_applicable_rules(
        self,
        domain: str,
        context: Dict[str, Any]
    ) -> List[CommonSenseRule]:
        """Get applicable rules for a domain and context."""
        if domain not in self.rules:
            return []
            
        applicable = []
        for rule in self.rules[domain]:
            # Check if any exceptions apply
            exceptions_apply = any(
                self._check_exception(exc, context)
                for exc in rule.exceptions
            )
            
            if not exceptions_apply:
                applicable.append(rule)
                
        # Sort by priority and confidence
        return sorted(
            applicable,
            key=lambda x: (
                self.rule_priorities[(x.domain, x.rule)],
                x.confidence
            ),
            reverse=True
        )
        
    def _check_exception(
        self,
        exception: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if an exception applies to the context."""
        # Implement exception checking logic
        # This is a simplified version
        try:
            return eval(exception, {"context": context})
        except:
            logger.warning(f"Failed to evaluate exception: {exception}")
            return False
            
    def apply_common_sense(
        self,
        domain: str,
        context: Dict[str, Any],
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Apply common sense reasoning to a situation."""
        applicable_rules = self.get_applicable_rules(domain, context)
        
        results = []
        for rule in applicable_rules:
            if rule.confidence >= min_confidence:
                try:
                    # Evaluate rule in context
                    rule_result = eval(
                        rule.rule,
                        {"context": context}
                    )
                    
                    results.append({
                        "rule": rule.rule,
                        "result": rule_result,
                        "confidence": rule.confidence,
                        "priority": rule.priority
                    })
                except Exception as e:
                    logger.error(f"Error applying rule {rule.rule}: {str(e)}")
                    
        return sorted(
            results,
            key=lambda x: (x["priority"], x["confidence"]),
            reverse=True
        )

# Dependency injection
def get_kg_manager():
    return KnowledgeGraphManager()

def get_symbolic_reasoner():
    return SymbolicReasoner()

def get_causal_reasoner():
    return CausalReasoner()

def get_common_sense_reasoner():
    return CommonSenseReasoner()

# API endpoints
@app.post("/knowledge/add_node")
async def add_knowledge_node(
    node_id: str,
    properties: Dict[str, Any],
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    try:
        kg_manager.add_node(node_id, properties)
        return {"status": "success", "node_id": node_id}
    except Exception as e:
        logger.error(f"Error adding node: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/add_edge")
async def add_knowledge_edge(
    source: str,
    target: str,
    relation: str,
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    try:
        kg_manager.add_edge(source, target, relation)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding edge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/symbolic/add_rule")
async def add_symbolic_rule(
    rule: ReasoningRule,
    symbolic_reasoner: SymbolicReasoner = Depends(get_symbolic_reasoner)
):
    try:
        symbolic_reasoner.add_rule(rule)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding symbolic rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/causal/add_relation")
async def add_causal_relation(
    relation: CausalRelation,
    causal_reasoner: CausalReasoner = Depends(get_causal_reasoner)
):
    try:
        causal_reasoner.add_relation(relation)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding causal relation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/common_sense/add_rule")
async def add_common_sense_rule(
    rule: CommonSenseRule,
    common_sense_reasoner: CommonSenseReasoner = Depends(get_common_sense_reasoner)
):
    try:
        common_sense_reasoner.add_rule(rule)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding common sense rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason/query")
async def query_knowledge(
    query: Dict[str, Any],
    kg_manager: KnowledgeGraphManager = Depends(get_kg_manager)
):
    try:
        results = kg_manager.query_graph(query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error querying knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason/infer")
async def infer_knowledge(
    facts: List[str],
    symbolic_reasoner: SymbolicReasoner = Depends(get_symbolic_reasoner)
):
    try:
        inferred_facts = symbolic_reasoner.apply_rules(tuple(facts))  # Convert to tuple for caching
        return {"inferred_facts": inferred_facts}
    except Exception as e:
        logger.error(f"Error inferring knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason/causes")
async def infer_causes(
    effect: str,
    causal_reasoner: CausalReasoner = Depends(get_causal_reasoner)
):
    try:
        causes = causal_reasoner.predict_effects(effect)
        return {"causes": causes}
    except Exception as e:
        logger.error(f"Error inferring causes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason/common_sense")
async def apply_common_sense(
    situation: Dict[str, Any],
    common_sense_reasoner: CommonSenseReasoner = Depends(get_common_sense_reasoner)
):
    try:
        result = common_sense_reasoner.apply_common_sense(
            situation["domain"],
            situation,
            situation.get("confidence", 0.5)
        )
        return result
    except Exception as e:
        logger.error(f"Error applying common sense reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason", response_model=ReasoningResponse)
async def reason(request: ReasoningRequest):
    """Main reasoning endpoint combining multiple reasoning types."""
    try:
        # Initialize reasoners
        symbolic = SymbolicReasoner()
        causal = CausalReasoner()
        common_sense = CommonSenseReasoner()
        
        config = request.config or ReasoningConfig()
        
        conclusions = []
        evidence = []
        reasoning_path = []
        
        # Apply symbolic reasoning
        if config.symbolic_reasoning_enabled:
            symbolic_results = symbolic.apply_rules(request.facts)
            for fact, conf in symbolic_results:
                conclusions.append({
                    "type": "symbolic",
                    "conclusion": fact,
                    "confidence": conf
                })
                reasoning_path.append(f"Symbolic: {fact}")
                
        # Apply causal reasoning
        if config.causal_reasoning_enabled:
            for fact in request.facts:
                effects = causal.predict_effects(
                    fact,
                    config.confidence_threshold
                )
                for effect in effects:
                    conclusions.append({
                        "type": "causal",
                        "conclusion": effect["effect"],
                        "confidence": effect["confidence"],
                        "temporal_delay": effect["temporal_delay"]
                    })
                    reasoning_path.append(
                        f"Causal: {fact} -> {effect['effect']}"
                    )
                    
        # Apply common sense reasoning
        if config.common_sense_enabled:
            # Extract domain from query
            domain = request.query.split(".")[0]
            context = {"facts": request.facts, "query": request.query}
            
            common_sense_results = common_sense.apply_common_sense(
                domain,
                context,
                config.confidence_threshold
            )
            
            for result in common_sense_results:
                conclusions.append({
                    "type": "common_sense",
                    "conclusion": str(result["result"]),
                    "confidence": result["confidence"],
                    "priority": result["priority"]
                })
                reasoning_path.append(
                    f"Common Sense: {result['rule']} -> {result['result']}"
                )
                
        # Calculate overall confidence
        if conclusions:
            overall_confidence = np.mean([
                c["confidence"] for c in conclusions
            ])
        else:
            overall_confidence = 0.0
            
        return ReasoningResponse(
            conclusions=conclusions,
            confidence=overall_confidence,
            supporting_evidence=evidence,
            reasoning_path=reasoning_path
        )
        
    except Exception as e:
        logger.error(f"Reasoning error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning failed: {str(e)}"
        )

@app.post("/knowledge/add")
async def add_knowledge(knowledge: KnowledgeGraph):
    """Add knowledge to the graph."""
    try:
        kg_manager = KnowledgeGraphManager()
        
        # Add nodes
        for node in knowledge.nodes:
            kg_manager.add_node(node["id"], node["properties"])
            
        # Add edges
        for edge in knowledge.edges:
            kg_manager.add_edge(
                edge["source"],
                edge["target"],
                edge["relation"]
            )
            
        return {"status": "success", "message": "Knowledge added"}
        
    except Exception as e:
        logger.error(f"Failed to add knowledge: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add knowledge: {str(e)}"
        )

@app.post("/rules/add")
async def add_rule(rule: ReasoningRule):
    """Add a reasoning rule."""
    try:
        symbolic = SymbolicReasoner()
        symbolic.add_rule(rule)
        return {"status": "success", "message": "Rule added"}
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to add rule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add rule: {str(e)}"
        )

@app.post("/causal/add")
async def add_causal_relation(relation: CausalRelation):
    """Add a causal relation."""
    try:
        causal = CausalReasoner()
        causal.add_relation(relation)
        return {"status": "success", "message": "Causal relation added"}
        
    except Exception as e:
        logger.error(f"Failed to add causal relation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add causal relation: {str(e)}"
        )

@app.post("/common-sense/add")
async def add_common_sense_rule(rule: CommonSenseRule):
    """Add a common sense rule."""
    try:
        common_sense = CommonSenseReasoner()
        common_sense.add_rule(rule)
        return {"status": "success", "message": "Common sense rule added"}
        
    except Exception as e:
        logger.error(f"Failed to add common sense rule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add common sense rule: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing reasoning service...")
    try:
        reasoning_manager = ReasoningManager()
        app.state.reasoning_manager = reasoning_manager
        logger.info("Reasoning service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize reasoning service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down reasoning service...")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/reason")
@limiter.limit("50/minute")
async def process_reasoning_request(
    request: Request,
    reasoning_request: ReasoningRequest,
    background_tasks: BackgroundTasks
):
    """Process reasoning request."""
    try:
        result = await request.app.state.reasoning_manager.process_query(
            reasoning_request,
            background_tasks
        )
        return result.dict()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge")
@limiter.limit("30/minute")
async def add_knowledge(
    request: Request,
    knowledge_type: KnowledgeType,
    content: Dict[str, Any]
):
    """Add knowledge to the system."""
    try:
        if knowledge_type == KnowledgeType.FACT:
            request.app.state.reasoning_manager.knowledge_graph.add_fact(
                content["subject"],
                content["predicate"],
                content["object"]
            )
        elif knowledge_type == KnowledgeType.CONCEPT:
            request.app.state.reasoning_manager.knowledge_graph.add_concept(
                content["concept"],
                content["properties"]
            )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rules")
@limiter.limit("20/minute")
async def add_rule(
    request: Request,
    rule_type: ReasoningType,
    content: Dict[str, Any]
):
    """Add reasoning rule."""
    try:
        if rule_type == ReasoningType.SYMBOLIC:
            request.app.state.reasoning_manager.symbolic_reasoner.add_rule(
                content["rule_id"],
                content["premises"],
                content["conclusion"]
            )
        elif rule_type == ReasoningType.COMMON_SENSE:
            request.app.state.reasoning_manager.common_sense_reasoner.add_rule(
                content["context"],
                content["condition"],
                content["consequence"],
                content.get("exceptions")
            )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/causal")
@limiter.limit("20/minute")
async def add_causal_relation(
    request: Request,
    content: Dict[str, Any]
):
    """Add causal relation."""
    try:
        request.app.state.reasoning_manager.causal_reasoner.add_relation(
            content["cause"],
            content["effect"],
            content["strength"],
            content.get("temporal_delay")
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding causal relation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8400) 