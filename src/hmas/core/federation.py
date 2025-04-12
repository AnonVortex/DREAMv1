from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from .organization import Organization
import asyncio
import logging

class Federation:
    """Manages the complete AGI pipeline and coordinates all organizations."""
    
    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.id = uuid4()
        self.name = name
        self.description = description
        self.config = config or {}
        self.organizations: Dict[UUID, Organization] = {}
        self.active = False
        self.state = {
            "global_resources": {},
            "global_metrics": {},
            "system_goals": [],
            "achievements": [],
            "health_status": "initializing"
        }
        self.logger = logging.getLogger(__name__)
        
    async def add_organization(self, org: Organization) -> bool:
        """Add an organization to the federation."""
        if org.id not in self.organizations:
            self.organizations[org.id] = org
            return True
        return False
        
    async def remove_organization(self, org_id: UUID) -> bool:
        """Remove an organization from the federation."""
        if org_id in self.organizations:
            del self.organizations[org_id]
            return True
        return False
        
    async def start(self) -> bool:
        """Activate all organizations in the federation."""
        try:
            for org in self.organizations.values():
                await org.start()
            self.active = True
            self.state["health_status"] = "running"
            return True
        except Exception as e:
            self.logger.error(f"Error starting federation {self.name}: {str(e)}")
            self.state["health_status"] = "error"
            return False
            
    async def stop(self) -> bool:
        """Deactivate all organizations in the federation."""
        try:
            for org in self.organizations.values():
                await org.stop()
            self.active = False
            self.state["health_status"] = "stopped"
            return True
        except Exception as e:
            self.logger.error(f"Error stopping federation {self.name}: {str(e)}")
            self.state["health_status"] = "error"
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Return current federation status."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "active": self.active,
            "org_count": len(self.organizations),
            "organizations": [org.get_status() for org in self.organizations.values()],
            "state": self.state
        }
        
    async def set_system_goals(self, goals: List[Dict[str, Any]]) -> bool:
        """Set system-wide goals."""
        self.state["system_goals"] = goals
        # Distribute goals to relevant organizations
        for org in self.organizations.values():
            org_goals = [g for g in goals if g.get("org_type") == org.org_type]
            await org.set_goals(org_goals)
        return True
        
    async def allocate_global_resources(self, resources: Dict[str, Any]) -> bool:
        """Allocate resources across organizations."""
        self.state["global_resources"].update(resources)
        # Distribute resources based on organization needs and priorities
        for org in self.organizations.values():
            org_resources = self._calculate_org_resources(org, resources)
            await org.allocate_resources(org_resources)
        return True
        
    def _calculate_org_resources(self, org: Organization, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource allocation for an organization."""
        # Implement resource allocation strategy
        # This is a simple example - should be enhanced based on specific needs
        total_orgs = len(self.organizations)
        return {k: v / total_orgs for k, v in resources.items()}
        
    async def monitor_health(self) -> None:
        """Continuously monitor system health."""
        while self.active:
            try:
                metrics = await self._collect_system_metrics()
                self.state["global_metrics"] = metrics
                health_status = self._evaluate_health(metrics)
                self.state["health_status"] = health_status
                
                if health_status == "degraded":
                    await self._initiate_self_healing()
                    
                await asyncio.sleep(60)  # Adjust monitoring interval as needed
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {str(e)}")
                
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all organizations."""
        metrics = {
            "org_metrics": {},
            "system_metrics": {
                "active_orgs": len([o for o in self.organizations.values() if o.active]),
                "total_teams": sum(len(o.teams) for o in self.organizations.values()),
                "goals_achieved": len(self.state["achievements"]),
                "pending_goals": len(self.state["system_goals"])
            }
        }
        
        for org in self.organizations.values():
            metrics["org_metrics"][str(org.id)] = await org.evaluate_performance()
            
        return metrics
        
    def _evaluate_health(self, metrics: Dict[str, Any]) -> str:
        """Evaluate system health based on metrics."""
        active_orgs = metrics["system_metrics"]["active_orgs"]
        total_orgs = len(self.organizations)
        
        if active_orgs == total_orgs:
            return "healthy"
        elif active_orgs >= total_orgs * 0.8:
            return "degraded"
        else:
            return "critical"
            
    async def _initiate_self_healing(self) -> None:
        """Attempt to recover from degraded state."""
        self.logger.warning("Initiating self-healing procedures")
        for org in self.organizations.values():
            if not org.active:
                try:
                    await org.start()
                except Exception as e:
                    self.logger.error(f"Failed to restart organization {org.name}: {str(e)}")
                    
    async def broadcast_system_message(self, message: Dict[str, Any]) -> bool:
        """Broadcast a message to all organizations."""
        try:
            for org in self.organizations.values():
                await org.broadcast(message)
            return True
        except Exception as e:
            self.logger.error(f"Error broadcasting system message: {str(e)}")
            return False 