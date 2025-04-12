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

class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    OBSERVER = "observer"

class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Agent(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: AgentRole
    capabilities: List[str]
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    required_capabilities: List[str]
    assigned_agents: List[str] = []
    dependencies: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class TeamFormation(BaseModel):
    team_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    members: List[str]
    coordinator: str
    formed_at: datetime = Field(default_factory=datetime.utcnow)
    dissolved_at: Optional[datetime] = None
    status: str = "active"

class CoordinationEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_agent: str
    target_agents: List[str]
    task_id: Optional[str] = None
    team_id: Optional[str] = None
    details: Dict[str, Any]

class CoordinationManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.teams: Dict[str, TeamFormation] = {}
        self.events: List[CoordinationEvent] = []
        
    async def register_agent(self, agent: Agent) -> str:
        """Register a new agent in the system."""
        if agent.agent_id in self.agents:
            raise HTTPException(
                status_code=400,
                detail=f"Agent {agent.agent_id} already registered"
            )
            
        self.agents[agent.agent_id] = agent
        
        await self._log_event(
            "agent_registered",
            agent.agent_id,
            [],
            details={"agent_role": agent.role, "capabilities": agent.capabilities}
        )
        
        return agent.agent_id
        
    async def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        current_task: Optional[str] = None
    ) -> Agent:
        """Update an agent's status."""
        if agent_id not in self.agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
            
        agent = self.agents[agent_id]
        agent.status = status
        agent.current_task = current_task
        agent.last_heartbeat = datetime.utcnow()
        
        await self._log_event(
            "agent_status_updated",
            agent_id,
            [],
            details={"new_status": status, "current_task": current_task}
        )
        
        return agent
        
    async def create_task(self, task: Task) -> str:
        """Create a new task."""
        if task.task_id in self.tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Task {task.task_id} already exists"
            )
            
        self.tasks[task.task_id] = task
        
        # Find suitable agents for the task
        suitable_agents = await self._find_suitable_agents(task)
        
        if suitable_agents and task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
            await self._auto_assign_task(task, suitable_agents)
            
        await self._log_event(
            "task_created",
            "",
            [agent.agent_id for agent in suitable_agents],
            task_id=task.task_id,
            details={"priority": task.priority, "capabilities": task.required_capabilities}
        )
        
        return task.task_id
        
    async def assign_task(self, task_id: str, agent_ids: List[str]) -> Task:
        """Assign a task to specific agents."""
        if task_id not in self.tasks:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
            
        task = self.tasks[task_id]
        
        # Verify all agents exist and are available
        for agent_id in agent_ids:
            if agent_id not in self.agents:
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent {agent_id} not found"
                )
                
            agent = self.agents[agent_id]
            if agent.status != AgentStatus.IDLE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent {agent_id} is not available"
                )
                
            # Verify agent capabilities
            if not all(cap in agent.capabilities for cap in task.required_capabilities):
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent {agent_id} lacks required capabilities"
                )
                
        # Update task and agent status
        task.status = TaskStatus.ASSIGNED
        task.assigned_agents = agent_ids
        task.started_at = datetime.utcnow()
        
        for agent_id in agent_ids:
            await self.update_agent_status(
                agent_id,
                AgentStatus.BUSY,
                task_id
            )
            
        await self._log_event(
            "task_assigned",
            "",
            agent_ids,
            task_id=task_id,
            details={"assigned_agents": agent_ids}
        )
        
        return task
        
    async def update_task_progress(
        self,
        task_id: str,
        progress: float,
        status: Optional[TaskStatus] = None,
        result: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Update task progress and optionally its status and result."""
        if task_id not in self.tasks:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
            
        task = self.tasks[task_id]
        task.progress = progress
        
        if status:
            task.status = status
            
        if result:
            task.result = result
            
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.utcnow()
            # Free up assigned agents
            for agent_id in task.assigned_agents:
                await self.update_agent_status(
                    agent_id,
                    AgentStatus.IDLE,
                    None
                )
                
        await self._log_event(
            "task_progress_updated",
            "",
            task.assigned_agents,
            task_id=task_id,
            details={
                "progress": progress,
                "status": status,
                "has_result": bool(result)
            }
        )
        
        return task
        
    async def form_team(self, task_id: str, coordinator_id: str) -> TeamFormation:
        """Form a team for a specific task."""
        if task_id not in self.tasks:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
            
        if coordinator_id not in self.agents:
            raise HTTPException(
                status_code=404,
                detail=f"Coordinator {coordinator_id} not found"
            )
            
        coordinator = self.agents[coordinator_id]
        if coordinator.role != AgentRole.COORDINATOR:
            raise HTTPException(
                status_code=400,
                detail=f"Agent {coordinator_id} is not a coordinator"
            )
            
        task = self.tasks[task_id]
        suitable_agents = await self._find_suitable_agents(task)
        
        if not suitable_agents:
            raise HTTPException(
                status_code=400,
                detail="No suitable agents found for team formation"
            )
            
        team = TeamFormation(
            task_id=task_id,
            members=[agent.agent_id for agent in suitable_agents],
            coordinator=coordinator_id
        )
        
        self.teams[team.team_id] = team
        
        await self._log_event(
            "team_formed",
            coordinator_id,
            team.members,
            task_id=task_id,
            team_id=team.team_id,
            details={"team_size": len(team.members)}
        )
        
        return team
        
    async def dissolve_team(self, team_id: str) -> TeamFormation:
        """Dissolve an existing team."""
        if team_id not in self.teams:
            raise HTTPException(
                status_code=404,
                detail=f"Team {team_id} not found"
            )
            
        team = self.teams[team_id]
        team.dissolved_at = datetime.utcnow()
        team.status = "dissolved"
        
        await self._log_event(
            "team_dissolved",
            team.coordinator,
            team.members,
            team_id=team_id,
            details={"reason": "task_completed"}
        )
        
        return team
        
    async def _find_suitable_agents(self, task: Task) -> List[Agent]:
        """Find agents suitable for a task based on capabilities and availability."""
        suitable_agents = []
        
        for agent in self.agents.values():
            if (
                agent.status == AgentStatus.IDLE
                and all(cap in agent.capabilities for cap in task.required_capabilities)
            ):
                suitable_agents.append(agent)
                
        return suitable_agents
        
    async def _auto_assign_task(self, task: Task, suitable_agents: List[Agent]):
        """Automatically assign high-priority tasks to suitable agents."""
        if suitable_agents:
            # Sort agents by least recently used
            sorted_agents = sorted(
                suitable_agents,
                key=lambda x: x.last_heartbeat
            )
            
            # Assign to the most available agent
            selected_agent = sorted_agents[0]
            await self.assign_task(task.task_id, [selected_agent.agent_id])
            
    async def _log_event(
        self,
        event_type: str,
        source_agent: str,
        target_agents: List[str],
        task_id: Optional[str] = None,
        team_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log a coordination event."""
        event = CoordinationEvent(
            event_type=event_type,
            source_agent=source_agent,
            target_agents=target_agents,
            task_id=task_id,
            team_id=team_id,
            details=details or {}
        )
        self.events.append(event)
        logger.info(f"Coordination event: {event.dict()}")
        return event

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing coordination service...")
    try:
        coordination_manager = CoordinationManager()
        app.state.coordination_manager = coordination_manager
        logger.info("Coordination service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize coordination service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down coordination service...")

app = FastAPI(title="HMAS Coordination Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/agents")
@limiter.limit("50/minute")
async def register_agent(request: Request, agent: Agent):
    """Register a new agent."""
    try:
        agent_id = await request.app.state.coordination_manager.register_agent(agent)
        return {"status": "success", "agent_id": agent_id}
    except Exception as e:
        logger.error(f"Error registering agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/agents/{agent_id}/status")
@limiter.limit("100/minute")
async def update_agent_status(
    request: Request,
    agent_id: str,
    status: AgentStatus,
    current_task: Optional[str] = None
):
    """Update an agent's status."""
    try:
        agent = await request.app.state.coordination_manager.update_agent_status(
            agent_id,
            status,
            current_task
        )
        return agent
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks")
@limiter.limit("50/minute")
async def create_task(request: Request, task: Task):
    """Create a new task."""
    try:
        task_id = await request.app.state.coordination_manager.create_task(task)
        return {"status": "success", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/{task_id}/assign")
@limiter.limit("50/minute")
async def assign_task(
    request: Request,
    task_id: str,
    agent_ids: List[str]
):
    """Assign a task to specific agents."""
    try:
        task = await request.app.state.coordination_manager.assign_task(
            task_id,
            agent_ids
        )
        return task
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error assigning task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/tasks/{task_id}/progress")
@limiter.limit("100/minute")
async def update_task_progress(
    request: Request,
    task_id: str,
    progress: float,
    status: Optional[TaskStatus] = None,
    result: Optional[Dict[str, Any]] = None
):
    """Update task progress."""
    try:
        task = await request.app.state.coordination_manager.update_task_progress(
            task_id,
            progress,
            status,
            result
        )
        return task
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating task progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/teams")
@limiter.limit("30/minute")
async def form_team(
    request: Request,
    task_id: str,
    coordinator_id: str
):
    """Form a team for a task."""
    try:
        team = await request.app.state.coordination_manager.form_team(
            task_id,
            coordinator_id
        )
        return team
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error forming team: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/teams/{team_id}")
@limiter.limit("30/minute")
async def dissolve_team(request: Request, team_id: str):
    """Dissolve a team."""
    try:
        team = await request.app.state.coordination_manager.dissolve_team(team_id)
        return team
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error dissolving team: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8700) 