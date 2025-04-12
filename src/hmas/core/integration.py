"""Integration module for H-MAS AGI system."""

from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
import asyncio
import logging
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
import torch
import numpy as np

from .consciousness import ConsciousnessCore
from .reasoning import ReasoningEngine
from .perception import PerceptionCore
from .memory import MemorySystem, MemoryType

class ProcessingState(Enum):
    """States of cognitive processing."""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    REASONING = "reasoning"
    LEARNING = "learning"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"

class ProcessPriority(Enum):
    """Priority levels for cognitive processes."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class IntegrationConfig:
    """Configuration for integration system."""
    max_concurrent_processes: int = 5
    attention_threshold: float = 0.6
    context_window_size: int = 10
    process_timeout: float = 5.0
    min_confidence_threshold: float = 0.7
    max_planning_steps: int = 20
    feedback_interval: float = 1.0
    learning_rate: float = 0.001

@dataclass
class CognitiveProcess:
    """Representation of a cognitive process."""
    id: UUID
    type: ProcessingState
    priority: ProcessPriority
    inputs: Dict[str, Any]
    state: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class ExecutiveControl:
    """Core executive control system."""
    
    def __init__(
        self,
        config: IntegrationConfig,
        consciousness: ConsciousnessCore,
        reasoning: ReasoningEngine,
        perception: PerceptionCore,
        memory: MemorySystem
    ):
        """Initialize executive control."""
        self.config = config
        self.consciousness = consciousness
        self.reasoning = reasoning
        self.perception = perception
        self.memory = memory
        self.logger = logging.getLogger("executive_control")
        
        # Initialize process management
        self.active_processes: Dict[UUID, CognitiveProcess] = {}
        self.process_queue: List[CognitiveProcess] = []
        self.completed_processes: Dict[UUID, CognitiveProcess] = {}
        
        # Initialize state tracking
        self.current_state = ProcessingState.IDLE
        self.global_context: Dict[str, Any] = {}
        self.attention_focus: Optional[UUID] = None
        
        # Initialize performance metrics
        self.performance_metrics: Dict[str, List[float]] = {
            "response_times": [],
            "success_rates": [],
            "confidence_scores": [],
            "resource_usage": []
        }
        
    async def process_input(
        self,
        input_data: Dict[str, Any],
        priority: ProcessPriority = ProcessPriority.MEDIUM
    ) -> UUID:
        """Process new input through the cognitive pipeline."""
        process_id = uuid4()
        
        # Create cognitive process
        process = CognitiveProcess(
            id=process_id,
            type=ProcessingState.PERCEIVING,
            priority=priority,
            inputs=input_data,
            state={"stage": "initial"}
        )
        
        # Add to queue
        await self._queue_process(process)
        
        # Start processing if capacity available
        await self._process_queue()
        
        return process_id
        
    async def get_process_status(self, process_id: UUID) -> Optional[CognitiveProcess]:
        """Get status of a cognitive process."""
        if process_id in self.active_processes:
            return self.active_processes[process_id]
        return self.completed_processes.get(process_id)
        
    async def update_global_context(self, context_update: Dict[str, Any]) -> None:
        """Update global context with new information."""
        self.global_context.update(context_update)
        
        # Trim context window if needed
        if len(self.global_context) > self.config.context_window_size:
            oldest_keys = sorted(
                self.global_context.keys(),
                key=lambda k: self.global_context[k].get("timestamp", 0)
            )[:len(self.global_context) - self.config.context_window_size]
            
            for key in oldest_keys:
                del self.global_context[key]
                
        # Update consciousness with context change
        await self._notify_consciousness(context_update)
        
    async def shift_attention(self, target_id: UUID) -> bool:
        """Shift attention focus to specific target."""
        # Check if target exists in any memory store
        target_exists = (
            target_id in self.memory.episodic_store or
            target_id in self.memory.working_memory
        )
        
        if not target_exists:
            return False
            
        # Update attention focus
        self.attention_focus = target_id
        
        # Notify consciousness of attention shift
        await self._notify_consciousness({
            "event": "attention_shift",
            "target": target_id
        })
        
        return True
        
    async def execute_cognitive_cycle(self) -> None:
        """Execute one complete cognitive cycle."""
        try:
            # Process perception
            perception_results = await self._process_perception()
            
            # Update working memory
            await self._update_working_memory(perception_results)
            
            # Perform reasoning
            reasoning_results = await self._perform_reasoning()
            
            # Generate response/action
            action_results = await self._generate_action(reasoning_results)
            
            # Learn from cycle
            await self._learn_from_cycle({
                "perception": perception_results,
                "reasoning": reasoning_results,
                "action": action_results
            })
            
            # Update metrics
            self._update_performance_metrics({
                "cycle_success": True,
                "response_time": 0.0,  # Add actual timing
                "confidence": reasoning_results.get("confidence", 0.0)
            })
            
        except Exception as e:
            self.logger.error(f"Error in cognitive cycle: {str(e)}")
            self._update_performance_metrics({"cycle_success": False})
            
    async def _process_queue(self) -> None:
        """Process queued cognitive processes."""
        while (
            len(self.active_processes) < self.config.max_concurrent_processes and
            self.process_queue
        ):
            # Get highest priority process
            next_process = max(
                self.process_queue,
                key=lambda p: p.priority.value
            )
            self.process_queue.remove(next_process)
            
            # Start process
            next_process.started_at = datetime.now()
            self.active_processes[next_process.id] = next_process
            
            # Schedule process execution
            asyncio.create_task(self._execute_process(next_process))
            
    async def _execute_process(self, process: CognitiveProcess) -> None:
        """Execute a cognitive process."""
        try:
            if process.type == ProcessingState.PERCEIVING:
                results = await self._handle_perception(process)
            elif process.type == ProcessingState.REASONING:
                results = await self._handle_reasoning(process)
            elif process.type == ProcessingState.LEARNING:
                results = await self._handle_learning(process)
            elif process.type == ProcessingState.PLANNING:
                results = await self._handle_planning(process)
            elif process.type == ProcessingState.EXECUTING:
                results = await self._handle_execution(process)
            elif process.type == ProcessingState.REFLECTING:
                results = await self._handle_reflection(process)
            else:
                raise ValueError(f"Unknown process type: {process.type}")
                
            # Update process with results
            process.results = results
            process.completed_at = datetime.now()
            
            # Move to completed processes
            del self.active_processes[process.id]
            self.completed_processes[process.id] = process
            
            # Process next in queue
            await self._process_queue()
            
        except Exception as e:
            process.error = str(e)
            self.logger.error(f"Error executing process {process.id}: {str(e)}")
            
    async def _handle_perception(self, process: CognitiveProcess) -> Dict[str, Any]:
        """Handle perception process."""
        # Process input through perception module
        perception_results = await self.perception.process_input(process.inputs)
        
        # Store in episodic memory
        memory_id = await self.memory.store_episodic(
            episode_data=perception_results,
            context=self.global_context
        )
        
        # Update attention if significant
        if perception_results.get("significance", 0.0) > self.config.attention_threshold:
            await self.shift_attention(memory_id)
            
        return {
            "perception_results": perception_results,
            "memory_id": memory_id
        }
        
    async def _handle_reasoning(self, process: CognitiveProcess) -> Dict[str, Any]:
        """Handle reasoning process."""
        # Get relevant memories
        memories = await self._gather_relevant_memories(process.inputs)
        
        # Perform reasoning
        reasoning_results = await self.reasoning.reason(
            inputs=process.inputs,
            context=memories,
            global_context=self.global_context
        )
        
        # Store reasoning results
        await self.memory.store_semantic(
            concept=reasoning_results["concept"],
            properties=reasoning_results["properties"],
            relationships=reasoning_results["relationships"],
            confidence=reasoning_results["confidence"]
        )
        
        return reasoning_results
        
    async def _handle_learning(self, process: CognitiveProcess) -> Dict[str, Any]:
        """Handle learning process."""
        # Extract learning targets
        learning_data = process.inputs["learning_data"]
        
        # Update relevant systems
        updates = {
            "consciousness": await self.consciousness.learn(learning_data),
            "reasoning": await self.reasoning.update_knowledge(learning_data),
            "perception": await self.perception.adapt_processing(learning_data)
        }
        
        # Store learning results
        await self.memory.store_procedural(
            procedure_name=f"learning_{process.id}",
            steps=process.inputs["steps"],
            prerequisites=process.inputs.get("prerequisites", [])
        )
        
        return {
            "learning_updates": updates,
            "success": all(update["success"] for update in updates.values())
        }
        
    async def _handle_planning(self, process: CognitiveProcess) -> Dict[str, Any]:
        """Handle planning process."""
        # Generate plan using reasoning engine
        plan = await self.reasoning.generate_plan(
            goal=process.inputs["goal"],
            constraints=process.inputs.get("constraints", {}),
            context=self.global_context
        )
        
        # Validate plan
        validation_results = await self._validate_plan(plan)
        
        if validation_results["valid"]:
            # Store plan in procedural memory
            await self.memory.store_procedural(
                procedure_name=f"plan_{process.id}",
                steps=plan["steps"],
                prerequisites=plan["prerequisites"]
            )
            
        return {
            "plan": plan,
            "validation": validation_results,
            "success": validation_results["valid"]
        }
        
    async def _handle_execution(self, process: CognitiveProcess) -> Dict[str, Any]:
        """Handle execution process."""
        # Get execution procedure
        procedure = await self.memory.retrieve_procedural(
            process.inputs["procedure_name"]
        )
        
        if not procedure:
            raise ValueError("Procedure not found")
            
        # Execute steps
        results = []
        for step in procedure.steps:
            step_result = await self._execute_step(step)
            results.append(step_result)
            
            if not step_result["success"]:
                break
                
        # Update procedure success rate
        new_success_rate = sum(r["success"] for r in results) / len(results)
        procedure.success_rate = (
            0.9 * procedure.success_rate +
            0.1 * new_success_rate
        )
        
        return {
            "step_results": results,
            "success": all(r["success"] for r in results),
            "procedure": procedure.procedure_name
        }
        
    async def _handle_reflection(self, process: CognitiveProcess) -> Dict[str, Any]:
        """Handle reflection process."""
        # Analyze recent experiences
        recent_memories = await self.memory.retrieve_episodic(
            query={"timeframe": "recent"},
            limit=10
        )
        
        # Extract patterns and insights
        patterns = await self.reasoning.analyze_patterns(recent_memories)
        
        # Generate improvements
        improvements = await self._generate_improvements(patterns)
        
        # Store insights
        await self.memory.store_semantic(
            concept=f"reflection_{process.id}",
            properties={
                "patterns": patterns,
                "improvements": improvements
            },
            relationships={},
            confidence=0.8
        )
        
        return {
            "patterns": patterns,
            "improvements": improvements,
            "success": bool(improvements)
        }
        
    async def _validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a generated plan."""
        # Check prerequisites
        prerequisites_met = await self._check_prerequisites(plan["prerequisites"])
        
        # Estimate success probability
        success_prob = await self._estimate_success_probability(plan)
        
        # Check resource requirements
        resources_available = await self._check_resources(plan["resources"])
        
        # Validate individual steps
        step_validation = await self._validate_steps(plan["steps"])
        
        return {
            "valid": (
                prerequisites_met and
                success_prob > self.config.min_confidence_threshold and
                resources_available and
                step_validation["valid"]
            ),
            "prerequisites_met": prerequisites_met,
            "success_probability": success_prob,
            "resources_available": resources_available,
            "step_validation": step_validation
        }
        
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in a procedure."""
        try:
            # Update consciousness with current step
            await self._notify_consciousness({
                "event": "step_execution",
                "step": step
            })
            
            # Execute step based on type
            if step["type"] == "perception":
                result = await self.perception.process_input(step["inputs"])
            elif step["type"] == "reasoning":
                result = await self.reasoning.reason(
                    inputs=step["inputs"],
                    context=self.global_context
                )
            elif step["type"] == "memory":
                result = await self._handle_memory_step(step)
            else:
                raise ValueError(f"Unknown step type: {step['type']}")
                
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _notify_consciousness(self, event: Dict[str, Any]) -> None:
        """Notify consciousness of system events."""
        await self.consciousness.update_state({
            "timestamp": datetime.now().isoformat(),
            "event": event
        })
        
    def _update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update system performance metrics."""
        if metrics.get("cycle_success") is not None:
            self.performance_metrics["success_rates"].append(
                float(metrics["cycle_success"])
            )
            
        if "response_time" in metrics:
            self.performance_metrics["response_times"].append(
                metrics["response_time"]
            )
            
        if "confidence" in metrics:
            self.performance_metrics["confidence_scores"].append(
                metrics["confidence"]
            )
            
        # Maintain fixed window of metrics
        window_size = 1000
        for metric_list in self.performance_metrics.values():
            if len(metric_list) > window_size:
                metric_list[:] = metric_list[-window_size:]
                
    async def _queue_process(self, process: CognitiveProcess) -> None:
        """Add process to queue and maintain queue order."""
        self.process_queue.append(process)
        self.process_queue.sort(key=lambda p: p.priority.value, reverse=True)
        
    async def _gather_relevant_memories(
        self,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gather relevant memories for reasoning."""
        results = {
            "episodic": await self.memory.retrieve_episodic(
                query=query,
                limit=5
            ),
            "semantic": await self.memory.retrieve_semantic(
                concept=query.get("concept", ""),
                include_related=True
            ),
            "working": self.memory.working_memory.copy()
        }
        
        return results 