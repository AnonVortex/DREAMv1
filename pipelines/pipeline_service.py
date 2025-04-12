import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
import uuid
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

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class StepType(str, Enum):
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    RETRY = "retry"

class RetryStrategy(BaseModel):
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retry_on_errors: List[str] = []

class PipelineStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    step_type: StepType
    service: str
    endpoint: str
    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    condition: Optional[str] = None
    retry_strategy: Optional[RetryStrategy] = None
    timeout: Optional[float] = None
    dependencies: List[str] = []
    metadata: Optional[Dict[str, Any]] = None

class Pipeline(BaseModel):
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    version: str
    steps: List[PipelineStep]
    variables: Dict[str, Any] = {}
    status: PipelineStatus = PipelineStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class PipelineExecution(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    variables: Dict[str, Any] = {}
    step_results: Dict[str, Any] = {}
    current_step: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = {
        "total_duration": 0,
        "step_durations": {},
        "retries": {},
        "failures": {}
    }

class PipelineManager:
    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.active_executions: Set[str] = set()
        
    async def register_pipeline(self, pipeline: Pipeline) -> str:
        """Register a new pipeline."""
        if pipeline.pipeline_id in self.pipelines:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline {pipeline.pipeline_id} already exists"
            )
            
        # Validate pipeline structure
        await self._validate_pipeline(pipeline)
        
        self.pipelines[pipeline.pipeline_id] = pipeline
        logger.info(f"Registered pipeline: {pipeline.name} ({pipeline.pipeline_id})")
        return pipeline.pipeline_id
        
    async def execute_pipeline(
        self,
        pipeline_id: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start pipeline execution."""
        if pipeline_id not in self.pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline {pipeline_id} not found"
            )
            
        pipeline = self.pipelines[pipeline_id]
        execution = PipelineExecution(
            pipeline_id=pipeline_id,
            variables=variables or {}
        )
        
        self.executions[execution.execution_id] = execution
        self.active_executions.add(execution.execution_id)
        
        # Start execution in background
        asyncio.create_task(self._run_pipeline(execution))
        
        logger.info(f"Started pipeline execution: {execution.execution_id}")
        return execution.execution_id
        
    async def get_execution_status(self, execution_id: str) -> PipelineExecution:
        """Get status of pipeline execution."""
        if execution_id not in self.executions:
            raise HTTPException(
                status_code=404,
                detail=f"Execution {execution_id} not found"
            )
            
        return self.executions[execution_id]
        
    async def cancel_execution(self, execution_id: str) -> PipelineExecution:
        """Cancel pipeline execution."""
        if execution_id not in self.executions:
            raise HTTPException(
                status_code=404,
                detail=f"Execution {execution_id} not found"
            )
            
        execution = self.executions[execution_id]
        if execution.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
            raise HTTPException(
                status_code=400,
                detail="Cannot cancel completed or failed execution"
            )
            
        execution.status = PipelineStatus.CANCELLED
        self.active_executions.discard(execution_id)
        
        logger.info(f"Cancelled pipeline execution: {execution_id}")
        return execution
        
    async def _validate_pipeline(self, pipeline: Pipeline):
        """Validate pipeline structure and dependencies."""
        step_ids = {step.step_id for step in pipeline.steps}
        
        for step in pipeline.steps:
            # Validate dependencies
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid dependency {dep} in step {step.step_id}"
                    )
                    
            # Validate condition syntax if present
            if step.condition:
                try:
                    # Simple validation - could be more sophisticated
                    compile(step.condition, "<string>", "eval")
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid condition in step {step.step_id}: {str(e)}"
                    )
                    
    async def _run_pipeline(self, execution: PipelineExecution):
        """Execute pipeline steps."""
        try:
            pipeline = self.pipelines[execution.pipeline_id]
            execution.status = PipelineStatus.RUNNING
            execution.started_at = datetime.utcnow()
            
            # Create execution graph
            graph = self._create_execution_graph(pipeline)
            
            # Execute steps
            for step in self._get_execution_order(graph):
                if execution.status == PipelineStatus.CANCELLED:
                    break
                    
                execution.current_step = step.step_id
                start_time = datetime.utcnow()
                
                try:
                    # Check conditions
                    if step.condition and not self._evaluate_condition(
                        step.condition,
                        execution
                    ):
                        execution.step_results[step.step_id] = {
                            "status": StepStatus.SKIPPED,
                            "skipped_reason": "condition_not_met"
                        }
                        continue
                        
                    # Execute step
                    result = await self._execute_step(step, execution)
                    execution.step_results[step.step_id] = result
                    
                    # Update metrics
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    execution.metrics["step_durations"][step.step_id] = duration
                    
                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {str(e)}")
                    execution.step_results[step.step_id] = {
                        "status": StepStatus.FAILED,
                        "error": str(e)
                    }
                    execution.metrics["failures"][step.step_id] = str(e)
                    
                    if not step.retry_strategy:
                        execution.status = PipelineStatus.FAILED
                        execution.error = f"Step {step.step_id} failed: {str(e)}"
                        break
                        
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.COMPLETED
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            execution.status = PipelineStatus.FAILED
            execution.error = str(e)
            
        finally:
            execution.completed_at = datetime.utcnow()
            execution.metrics["total_duration"] = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            self.active_executions.discard(execution.execution_id)
            
    async def _execute_step(
        self,
        step: PipelineStep,
        execution: PipelineExecution
    ) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        if step.step_type == StepType.TASK:
            return await self._execute_task(step, execution)
        elif step.step_type == StepType.PARALLEL:
            return await self._execute_parallel(step, execution)
        elif step.step_type == StepType.LOOP:
            return await self._execute_loop(step, execution)
        elif step.step_type == StepType.RETRY:
            return await self._execute_with_retry(step, execution)
        else:
            raise ValueError(f"Unsupported step type: {step.step_type}")
            
    async def _execute_task(
        self,
        step: PipelineStep,
        execution: PipelineExecution
    ) -> Dict[str, Any]:
        """Execute a task step."""
        # Prepare input data
        input_data = {}
        for target, source in step.input_mapping.items():
            input_data[target] = self._get_variable(source, execution)
            
        # Execute task
        try:
            # Here you would make an HTTP request to the service endpoint
            # This is a placeholder for the actual service call
            result = {"status": "success", "data": input_data}
            
            # Map outputs
            for target, source in step.output_mapping.items():
                execution.variables[target] = result["data"].get(source)
                
            return {
                "status": StepStatus.COMPLETED,
                "result": result
            }
            
        except Exception as e:
            raise Exception(f"Task execution failed: {str(e)}")
            
    async def _execute_parallel(
        self,
        step: PipelineStep,
        execution: PipelineExecution
    ) -> Dict[str, Any]:
        """Execute parallel steps."""
        # Create tasks for parallel execution
        tasks = []
        for substep in step.metadata.get("steps", []):
            task = asyncio.create_task(self._execute_step(substep, execution))
            tasks.append(task)
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for failures
        failures = [
            (i, r) for i, r in enumerate(results)
            if isinstance(r, Exception)
        ]
        
        if failures:
            raise Exception(f"Parallel execution failed: {failures}")
            
        return {
            "status": StepStatus.COMPLETED,
            "results": results
        }
        
    async def _execute_loop(
        self,
        step: PipelineStep,
        execution: PipelineExecution
    ) -> Dict[str, Any]:
        """Execute a loop step."""
        results = []
        iteration = 0
        max_iterations = step.metadata.get("max_iterations", 10)
        
        while (
            iteration < max_iterations
            and self._evaluate_condition(step.condition, execution)
        ):
            result = await self._execute_step(
                step.metadata["body"],
                execution
            )
            results.append(result)
            iteration += 1
            
        return {
            "status": StepStatus.COMPLETED,
            "iterations": iteration,
            "results": results
        }
        
    async def _execute_with_retry(
        self,
        step: PipelineStep,
        execution: PipelineExecution
    ) -> Dict[str, Any]:
        """Execute a step with retry strategy."""
        if not step.retry_strategy:
            return await self._execute_step(step, execution)
            
        attempt = 0
        delay = step.retry_strategy.initial_delay
        
        while attempt < step.retry_strategy.max_attempts:
            try:
                result = await self._execute_step(step, execution)
                return result
            except Exception as e:
                attempt += 1
                execution.metrics["retries"][step.step_id] = attempt
                
                if attempt >= step.retry_strategy.max_attempts:
                    raise
                    
                # Check if error is retryable
                error_type = type(e).__name__
                if (
                    step.retry_strategy.retry_on_errors
                    and error_type not in step.retry_strategy.retry_on_errors
                ):
                    raise
                    
                # Wait before retrying
                await asyncio.sleep(delay)
                delay = min(
                    delay * step.retry_strategy.backoff_factor,
                    step.retry_strategy.max_delay
                )
                
    def _create_execution_graph(self, pipeline: Pipeline) -> Dict[str, Set[str]]:
        """Create execution graph from pipeline steps."""
        graph = {}
        for step in pipeline.steps:
            graph[step.step_id] = set(step.dependencies)
        return graph
        
    def _get_execution_order(self, graph: Dict[str, Set[str]]) -> List[PipelineStep]:
        """Get step execution order based on dependencies."""
        visited = set()
        temp = set()
        order = []
        
        def visit(step_id):
            if step_id in temp:
                raise HTTPException(
                    status_code=400,
                    detail="Circular dependency detected"
                )
            if step_id in visited:
                return
                
            temp.add(step_id)
            
            for dep in graph[step_id]:
                visit(dep)
                
            temp.remove(step_id)
            visited.add(step_id)
            order.append(step_id)
            
        for step_id in graph:
            if step_id not in visited:
                visit(step_id)
                
        return [
            step for step in self.pipelines[pipeline_id].steps
            if step.step_id in order
        ]
        
    def _get_variable(self, path: str, execution: PipelineExecution) -> Any:
        """Get variable value from execution context."""
        parts = path.split(".")
        value = execution.variables
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                raise ValueError(f"Invalid variable path: {path}")
                
        return value
        
    def _evaluate_condition(self, condition: str, execution: PipelineExecution) -> bool:
        """Evaluate condition in execution context."""
        try:
            return eval(
                condition,
                {},
                {"variables": execution.variables, "results": execution.step_results}
            )
        except Exception as e:
            raise ValueError(f"Error evaluating condition: {str(e)}")

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing pipeline service...")
    try:
        pipeline_manager = PipelineManager()
        app.state.pipeline_manager = pipeline_manager
        logger.info("Pipeline service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down pipeline service...")

app = FastAPI(title="HMAS Pipeline Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/pipelines")
@limiter.limit("50/minute")
async def register_pipeline(request: Request, pipeline: Pipeline):
    """Register a new pipeline."""
    try:
        pipeline_id = await request.app.state.pipeline_manager.register_pipeline(pipeline)
        return {"status": "success", "pipeline_id": pipeline_id}
    except Exception as e:
        logger.error(f"Error registering pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipelines/{pipeline_id}/execute")
@limiter.limit("30/minute")
async def execute_pipeline(
    request: Request,
    pipeline_id: str,
    variables: Optional[Dict[str, Any]] = None
):
    """Execute a pipeline."""
    try:
        execution_id = await request.app.state.pipeline_manager.execute_pipeline(
            pipeline_id,
            variables
        )
        return {"status": "success", "execution_id": execution_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error executing pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/executions/{execution_id}")
@limiter.limit("100/minute")
async def get_execution_status(request: Request, execution_id: str):
    """Get pipeline execution status."""
    try:
        execution = await request.app.state.pipeline_manager.get_execution_status(
            execution_id
        )
        return execution
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting execution status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/executions/{execution_id}/cancel")
@limiter.limit("30/minute")
async def cancel_execution(request: Request, execution_id: str):
    """Cancel pipeline execution."""
    try:
        execution = await request.app.state.pipeline_manager.cancel_execution(
            execution_id
        )
        return execution
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error cancelling execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8600) 