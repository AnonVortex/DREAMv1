from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from .agent import Agent

class Team:
    """Manages a collaborative group of agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        org_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.id = uuid4()
        self.name = name
        self.description = description
        self.org_id = org_id
        self.config = config or {}
        self.agents: Dict[UUID, Agent] = {}
        self.active = False
        self.state = {
            "task_queue": [],
            "completed_tasks": [],
            "performance_metrics": {}
        }
        
    async def add_agent(self, agent: Agent) -> bool:
        """Add an agent to the team."""
        if agent.id not in self.agents:
            agent.team_id = self.id
            self.agents[agent.id] = agent
            return True
        return False
        
    async def remove_agent(self, agent_id: UUID) -> bool:
        """Remove an agent from the team."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.team_id = None
            del self.agents[agent_id]
            return True
        return False
        
    async def start(self) -> bool:
        """Activate all agents in the team."""
        try:
            for agent in self.agents.values():
                await agent.start()
            self.active = True
            return True
        except Exception as e:
            print(f"Error starting team {self.name}: {str(e)}")
            return False
            
    async def stop(self) -> bool:
        """Deactivate all agents in the team."""
        try:
            for agent in self.agents.values():
                await agent.stop()
            self.active = False
            return True
        except Exception as e:
            print(f"Error stopping team {self.name}: {str(e)}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Return current team status."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "active": self.active,
            "org_id": str(self.org_id) if self.org_id else None,
            "agent_count": len(self.agents),
            "agents": [agent.get_status() for agent in self.agents.values()],
            "state": self.state
        }
        
    async def assign_task(self, task: Dict[str, Any]) -> bool:
        """Assign a task to the team."""
        self.state["task_queue"].append(task)
        return True
        
    async def get_capabilities(self) -> List[str]:
        """Get combined capabilities of all agents."""
        capabilities = set()
        for agent in self.agents.values():
            capabilities.update(agent.capabilities)
        return list(capabilities)
        
    async def broadcast(self, message: Dict[str, Any]) -> bool:
        """Send a message to all agents in the team."""
        try:
            for agent in self.agents.values():
                for target in self.agents.values():
                    if agent.id != target.id:
                        await agent.communicate(message, target.id)
            return True
        except Exception as e:
            print(f"Error broadcasting in team {self.name}: {str(e)}")
            return False 