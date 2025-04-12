import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from datetime import datetime
import json
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from prometheus_client import Counter, Histogram, start_http_server
import networkx as nx

# Configure logging
logging.config.fileConfig('pipelines/logging.conf')
logger = logging.getLogger(__name__)

class PipelineStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class StepType(str, Enum):
    PERCEPTION = "perception"
    MEMORY = "memory"
    LEARNING = "learning"
    REASONING = "reasoning"
    COMMUNICATION = "communication"
    FEEDBACK = "feedback"
    SPECIALIZED = "specialized"
    CUSTOM = "custom"

class PipelineStep(BaseModel):
    step_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    name: str
    step_type: StepType
    config: Dict[str, Any]
    dependencies: List[str] = []
    timeout: Optional[float] = 60.0
    retry_count: int = 3
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Pipeline(BaseModel):
    pipeline_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    name: str
    description: Optional[str] = None
    steps: List[PipelineStep]
    status: PipelineStatus = PipelineStatus.CREATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

class PipelineManager:
    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.step_results: Dict[str, Dict[str, Any]] = {}
        self.service_urls = {
            StepType.PERCEPTION: "http://localhost:8100",
            StepType.MEMORY: "http://localhost:8200",
            StepType.LEARNING: "http://localhost:8300",
            StepType.REASONING: "http://localhost:8400",
            StepType.COMMUNICATION: "http://localhost:8500",
            StepType.FEEDBACK: "http://localhost:8600",
            StepType.SPECIALIZED: "http://localhost:8700"
        }
    
    def validate_pipeline(self, pipeline: Pipeline) -> bool:
        """Validate pipeline structure and dependencies."""
        try:
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for step in pipeline.steps:
                G.add_node(step.step_id)
                for dep in step.dependencies:
                    G.add_edge(dep, step.step_id)
            
            # Check for cycles
            if not nx.is_directed_acyclic_graph(G):
                raise ValueError("Pipeline contains cyclic dependencies")
            
            # Validate step types and configurations
            for step in pipeline.steps:
                if step.step_type != StepType.CUSTOM and step.step_type not in self.service_urls:
                    raise ValueError(f"Invalid step type: {step.step_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline validation error: {str(e)}")
            raise ValueError(f"Pipeline validation failed: {str(e)}")
    
    async def execute_step(self, step: PipelineStep, pipeline_id: str) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        try:
            # Update step status
            step.status = StepStatus.RUNNING
            
            # Get service URL
            service_url = self.service_urls.get(step.step_type)
            if not service_url and step.step_type != StepType.CUSTOM:
                raise ValueError(f"Service URL not found for step type: {step.step_type}")
            
            # Prepare step input
            step_input = {
                "config": step.config,
                "dependencies": {
                    dep: self.step_results[pipeline_id].get(dep)
                    for dep in step.dependencies
                }
            }
            
            if step.step_type == StepType.CUSTOM:
                # Execute custom step logic
                result = await self._execute_custom_step(step, step_input)
            else:
                # Execute service step
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{service_url}/process",
                        json=step_input,
                        timeout=step.timeout
                    ) as response:
                        if response.status != 200:
                            raise HTTPException(
                                status_code=response.status,
                                detail=await response.text()
                            )
                        result = await response.json()
            
            # Update step status and result
            step.status = StepStatus.COMPLETED
            step.result = result
            
            # Store result
            if pipeline_id not in self.step_results:
                self.step_results[pipeline_id] = {}
            self.step_results[pipeline_id][step.step_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Step execution error: {str(e)}")
            step.status = StepStatus.FAILED
            step.error = str(e)
            raise
    
    async def _execute_custom_step(
        self,
        step: PipelineStep,
        step_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute custom step logic."""
        # Implement custom step execution logic here
        # This is a placeholder implementation
        return {
            "custom_step_result": True,
            "input_processed": step_input
        }
    
    async def execute_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """Execute pipeline steps in dependency order."""
        try:
            # Validate pipeline
            self.validate_pipeline(pipeline)
            
            # Update pipeline status
            pipeline.status = PipelineStatus.RUNNING
            pipeline.updated_at = datetime.utcnow()
            
            # Create directed graph
            G = nx.DiGraph()
            for step in pipeline.steps:
                G.add_node(step.step_id)
                for dep in step.dependencies:
                    G.add_edge(dep, step.step_id)
            
            # Get execution order
            execution_order = list(nx.topological_sort(G))
            
            # Execute steps in order
            for step_id in execution_order:
                step = next(s for s in pipeline.steps if s.step_id == step_id)
                
                # Check dependencies
                dependencies_met = all(
                    d.status == StepStatus.COMPLETED
                    for d in pipeline.steps
                    if d.step_id in step.dependencies
                )
                
                if not dependencies_met:
                    step.status = StepStatus.SKIPPED
                    continue
                
                # Execute step with retries
                for attempt in range(step.retry_count):
                    try:
                        await self.execute_step(step, pipeline.pipeline_id)
                        break
                    except Exception as e:
                        if attempt == step.retry_count - 1:
                            logger.error(
                                f"Step {step.step_id} failed after {step.retry_count} attempts"
                            )
                            pipeline.status = PipelineStatus.FAILED
                            raise
                        logger.warning(
                            f"Step {step.step_id} failed, attempt {attempt + 1}/{step.retry_count}"
                        )
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
            
            # Update pipeline status
            pipeline.status = PipelineStatus.COMPLETED
            pipeline.updated_at = datetime.utcnow()
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {str(e)}")
            pipeline.status = PipelineStatus.FAILED
            pipeline.updated_at = datetime.utcnow()
            raise

# Initialize FastAPI app
app = FastAPI(title="HMAS Pipeline Service")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# Initialize metrics
REQUESTS = Counter('pipeline_requests_total', 'Total requests to pipeline service')
LATENCY = Histogram('pipeline_request_latency_seconds', 'Request latency in seconds')

# Initialize pipeline manager
pipeline_manager = PipelineManager()

@app.post("/pipeline")
@limiter.limit("50/minute")
async def create_pipeline(pipeline: Pipeline, request: Request):
    """Create and execute a new pipeline."""
    REQUESTS.inc()
    with LATENCY.time():
        try:
            # Store pipeline
            pipeline_manager.pipelines[pipeline.pipeline_id] = pipeline
            
            # Execute pipeline
            result = await pipeline_manager.execute_pipeline(pipeline)
            return result
            
        except Exception as e:
            logger.error(f"Pipeline creation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline creation failed: {str(e)}"
            )

@app.get("/pipeline/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get pipeline status and results."""
    if pipeline_id not in pipeline_manager.pipelines:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline {pipeline_id} not found"
        )
    
    pipeline = pipeline_manager.pipelines[pipeline_id]
    return {
        "pipeline": pipeline,
        "results": pipeline_manager.step_results.get(pipeline_id, {})
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "active_pipelines": len(pipeline_manager.pipelines)
    }

# Start Prometheus metrics server
start_http_server(8801)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
