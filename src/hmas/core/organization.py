from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from .team import Team

class Organization:
    """Manages teams and functional modules within the AGI system."""
    
    def __init__(
        self,
        name: str,
        description: str,
        org_type: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.id = uuid4()
        self.name = name
        self.description = description
        self.org_type = org_type
        self.config = config or {}
        self.teams: Dict[UUID, Team] = {}
        self.active = False
        self.state = {
            "resources": {},
            "metrics": {},
            "goals": [],
            "achievements": []
        }
        
    async def add_team(self, team: Team) -> bool:
        """Add a team to the organization."""
        if team.id not in self.teams:
            team.org_id = self.id
            self.teams[team.id] = team
            return True
        return False
        
    async def remove_team(self, team_id: UUID) -> bool:
        """Remove a team from the organization."""
        if team_id in self.teams:
            team = self.teams[team_id]
            team.org_id = None
            del self.teams[team_id]
            return True
        return False
        
    async def start(self) -> bool:
        """Activate all teams in the organization."""
        try:
            for team in self.teams.values():
                await team.start()
            self.active = True
            return True
        except Exception as e:
            print(f"Error starting organization {self.name}: {str(e)}")
            return False
            
    async def stop(self) -> bool:
        """Deactivate all teams in the organization."""
        try:
            for team in self.teams.values():
                await team.stop()
            self.active = False
            return True
        except Exception as e:
            print(f"Error stopping organization {self.name}: {str(e)}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Return current organization status."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "org_type": self.org_type,
            "active": self.active,
            "team_count": len(self.teams),
            "teams": [team.get_status() for team in self.teams.values()],
            "state": self.state
        }
        
    async def set_goals(self, goals: List[Dict[str, Any]]) -> bool:
        """Set organizational goals."""
        self.state["goals"] = goals
        return True
        
    async def get_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all teams."""
        capabilities = {}
        for team in self.teams.values():
            capabilities[str(team.id)] = await team.get_capabilities()
        return capabilities
        
    async def allocate_resources(self, resources: Dict[str, Any]) -> bool:
        """Allocate resources to teams."""
        self.state["resources"].update(resources)
        return True
        
    async def broadcast(self, message: Dict[str, Any]) -> bool:
        """Send a message to all teams in the organization."""
        try:
            for team in self.teams.values():
                await team.broadcast(message)
            return True
        except Exception as e:
            print(f"Error broadcasting in organization {self.name}: {str(e)}")
            return False
            
    async def evaluate_performance(self) -> Dict[str, Any]:
        """Evaluate performance of the organization."""
        metrics = {
            "team_metrics": {},
            "overall_metrics": {
                "active_teams": len([t for t in self.teams.values() if t.active]),
                "total_agents": sum(len(t.agents) for t in self.teams.values()),
                "goals_achieved": len(self.state["achievements"]),
                "pending_goals": len(self.state["goals"])
            }
        }
        
        for team in self.teams.values():
            metrics["team_metrics"][str(team.id)] = team.state["performance_metrics"]
            
        self.state["metrics"] = metrics
        return metrics 