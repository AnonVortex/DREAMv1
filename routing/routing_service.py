import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from datetime import datetime
import json
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class RouteType(str, Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    ANYCAST = "anycast"
    PRIORITY = "priority"

class MessageType(str, Enum):
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"
    RESPONSE = "response"
    NOTIFICATION = "notification"

class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOAD = "least_load"
    FASTEST_RESPONSE = "fastest_response"
    CONTENT_BASED = "content_based"
    PRIORITY_BASED = "priority_based"

class RoutingConfig(BaseModel):
    route_type: RouteType
    strategy: RoutingStrategy
    filters: Optional[Dict[str, Any]] = None
    priorities: Optional[Dict[str, int]] = None
    timeout_seconds: int = 30
    retry_count: int = 3
    backoff_factor: float = 1.5

class Message(BaseModel):
    message_id: str
    sender: str
    recipients: List[str]
    message_type: MessageType
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class RouteResult(BaseModel):
    route_id: str
    message_id: str
    status: str
    recipients: List[str]
    delivery_info: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class LoadBalancer:
    def __init__(self):
        self.node_stats: Dict[str, Dict[str, Any]] = {}
        self.last_used: Dict[str, datetime] = {}
        
    def update_stats(self, node_id: str, stats: Dict[str, Any]):
        """Update node statistics."""
        self.node_stats[node_id] = {
            **stats,
            "last_updated": datetime.now()
        }
        
    def get_next_node(self, strategy: RoutingStrategy, nodes: List[str]) -> str:
        """Get next node based on strategy."""
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(nodes)
        elif strategy == RoutingStrategy.LEAST_LOAD:
            return self._least_load_select(nodes)
        elif strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(nodes)
        else:
            return nodes[0]  # Default to first node
            
    def _round_robin_select(self, nodes: List[str]) -> str:
        """Select node using round-robin strategy."""
        # Find least recently used node
        sorted_nodes = sorted(
            nodes,
            key=lambda x: self.last_used.get(x, datetime.min)
        )
        selected = sorted_nodes[0]
        self.last_used[selected] = datetime.now()
        return selected
        
    def _least_load_select(self, nodes: List[str]) -> str:
        """Select node with least load."""
        return min(
            nodes,
            key=lambda x: self.node_stats.get(x, {}).get("load", float("inf"))
        )
        
    def _fastest_response_select(self, nodes: List[str]) -> str:
        """Select node with fastest response time."""
        return min(
            nodes,
            key=lambda x: self.node_stats.get(x, {}).get("response_time", float("inf"))
        )

class ContentRouter:
    def __init__(self):
        self.content_rules: Dict[str, Dict[str, Any]] = {}
        
    def add_rule(self, rule_id: str, rule: Dict[str, Any]):
        """Add content-based routing rule."""
        self.content_rules[rule_id] = rule
        
    def match_rules(self, content: Dict[str, Any]) -> List[str]:
        """Match content against routing rules."""
        matched_recipients = set()
        
        for rule in self.content_rules.values():
            if self._evaluate_rule(content, rule):
                matched_recipients.update(rule.get("recipients", []))
                
        return list(matched_recipients)
        
    def _evaluate_rule(self, content: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Evaluate content against a single rule."""
        conditions = rule.get("conditions", {})
        
        for key, condition in conditions.items():
            if key not in content:
                return False
                
            value = content[key]
            operator = condition.get("operator", "eq")
            target = condition.get("value")
            
            if not self._compare_values(value, operator, target):
                return False
                
        return True
        
    def _compare_values(self, value: Any, operator: str, target: Any) -> bool:
        """Compare values using specified operator."""
        if operator == "eq":
            return value == target
        elif operator == "ne":
            return value != target
        elif operator == "gt":
            return value > target
        elif operator == "lt":
            return value < target
        elif operator == "contains":
            return target in value
        elif operator == "startswith":
            return value.startswith(target)
        else:
            return False

class RoutingManager:
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.content_router = ContentRouter()
        self.configs: Dict[str, RoutingConfig] = {}
        self.active_routes: Dict[str, str] = {}
        
    def register_config(self, config_id: str, config: RoutingConfig):
        """Register a routing configuration."""
        self.configs[config_id] = config
        
    async def route_message(
        self,
        message: Message,
        config: RoutingConfig,
        background_tasks: BackgroundTasks
    ) -> RouteResult:
        """Route a message according to configuration."""
        route_id = f"route_{datetime.now().isoformat()}"
        recipients = []
        
        try:
            if config.route_type == RouteType.DIRECT:
                recipients = message.recipients
            elif config.route_type == RouteType.BROADCAST:
                # Send to all available nodes
                recipients = list(self.load_balancer.node_stats.keys())
            elif config.route_type == RouteType.MULTICAST:
                # Use content-based routing
                recipients = self.content_router.match_rules(message.content)
            elif config.route_type == RouteType.ANYCAST:
                # Select single best node
                recipients = [self.load_balancer.get_next_node(
                    config.strategy,
                    list(self.load_balancer.node_stats.keys())
                )]
                
            delivery_info = await self._deliver_message(
                message,
                recipients,
                config
            )
            
            return RouteResult(
                route_id=route_id,
                message_id=message.message_id,
                status="completed",
                recipients=recipients,
                delivery_info=delivery_info
            )
            
        except Exception as e:
            logger.error(f"Error routing message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def _deliver_message(
        self,
        message: Message,
        recipients: List[str],
        config: RoutingConfig
    ) -> Dict[str, Any]:
        """Deliver message to recipients."""
        delivery_info = {
            "successful": [],
            "failed": [],
            "retry_counts": {}
        }
        
        for recipient in recipients:
            try:
                # Implement actual message delivery logic here
                # This is a placeholder for demonstration
                await asyncio.sleep(0.1)
                delivery_info["successful"].append(recipient)
            except Exception as e:
                delivery_info["failed"].append({
                    "recipient": recipient,
                    "error": str(e)
                })
                
        return delivery_info

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing routing service...")
    try:
        routing_manager = RoutingManager()
        app.state.routing_manager = routing_manager
        logger.info("Routing service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize routing service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down routing service...")

app = FastAPI(title="HMAS Routing Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/config/{config_id}")
@limiter.limit("20/minute")
async def register_config(
    request: Request,
    config_id: str,
    config: RoutingConfig
):
    """Register a routing configuration."""
    try:
        request.app.state.routing_manager.register_config(config_id, config)
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Error registering configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/route")
@limiter.limit("100/minute")
async def route_message(
    request: Request,
    message: Message,
    config_id: str,
    background_tasks: BackgroundTasks
):
    """Route a message."""
    try:
        if config_id not in request.app.state.routing_manager.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        config = request.app.state.routing_manager.configs[config_id]
        result = await request.app.state.routing_manager.route_message(
            message,
            config,
            background_tasks
        )
        return result.dict()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error routing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/node/{node_id}/stats")
@limiter.limit("50/minute")
async def update_node_stats(
    request: Request,
    node_id: str,
    stats: Dict[str, Any]
):
    """Update node statistics."""
    try:
        request.app.state.routing_manager.load_balancer.update_stats(node_id, stats)
        return {"status": "success", "node_id": node_id}
    except Exception as e:
        logger.error(f"Error updating node stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rules/{rule_id}")
@limiter.limit("20/minute")
async def add_routing_rule(
    request: Request,
    rule_id: str,
    rule: Dict[str, Any]
):
    """Add content-based routing rule."""
    try:
        request.app.state.routing_manager.content_router.add_rule(rule_id, rule)
        return {"status": "success", "rule_id": rule_id}
    except Exception as e:
        logger.error(f"Error adding routing rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8600) 