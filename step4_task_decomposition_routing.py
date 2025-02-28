# by Alexis Soto-Yanez
import logging

logging.basicConfig(level=logging.INFO)

class GoalInterpreter:
    def interpret_goal(self, goal):
        """
        Interprets a high-level goal (provided as a dictionary).
        For example, if the goal has a 'type' field, it processes that to determine required subtask(s).
        """
        logging.info("Interpreting goal...")
        # In a production system, you might add natural language processing or rule-based logic here.
        return goal

class TaskDecomposer:
    def decompose(self, goal):
        """
        Decomposes the goal into a list of subtasks.
        If the goal type is 'graph_optimization', we generate a subtask for it.
        Otherwise, we generate default subtasks.
        """
        logging.info("Decomposing goal into subtasks...")
        subtasks = []
        if goal.get("type") == "graph_optimization":
            subtasks.append({
                "name": "graph_optimization",
                "raw_data": goal.get("raw_data", {}),
                "constraints": goal.get("constraints", {})
            })
        else:
            subtasks.append({
                "name": "default_task",
                "data": goal.get("data", {})
            })
        return subtasks

class TaskRouter:
    def route(self, subtasks):
        """
        Routes each subtask to the appropriate specialized processing agent.
        Returns a dictionary mapping subtask names to agent types.
        """
        logging.info("Routing subtasks to specialized agents...")
        routing_info = {}
        for subtask in subtasks:
            if subtask["name"] == "graph_optimization":
                routing_info[subtask["name"]] = "GraphOptimizationAgent"
            else:
                routing_info[subtask["name"]] = "DefaultProcessingAgent"
        return routing_info

class TaskDecompositionRouting:
    """
    Wrapper class that integrates goal interpretation, task decomposition,
    and task routing into one cohesive module.
    """
    def __init__(self):
        self.goal_interpreter = GoalInterpreter()
        self.task_decomposer = TaskDecomposer()
        self.task_router = TaskRouter()
    
    def run(self, working_memory):
        """
        Runs the complete process using the provided working_memory.
        working_memory should be a dictionary containing at least a 'goal' key.
        
        Returns:
            routing_info (dict): Mapping of subtask names to agent types.
        """
        logging.info("Running Task Decomposition and Routing...")
        # Retrieve the high-level goal from working_memory.
        goal = working_memory.get("goal", {})
        interpreted_goal = self.goal_interpreter.interpret_goal(goal)
        subtasks = self.task_decomposer.decompose(interpreted_goal)
        routing_info = self.task_router.route(subtasks)
        logging.info(f"Generated Routing Info: {routing_info}")
        return routing_info

if __name__ == "__main__":
    # For standalone testing, simulate a working memory with a graph optimization goal.
    test_working_memory = {
        "goal": {
            "type": "graph_optimization",
            "raw_data": {
                "nodes": [
                    {"id": 0, "features": [0.8]*10},
                    {"id": 1, "features": [0.3]*10},
                    {"id": 2, "features": [0.5]*10}
                ],
                "edges": [(0, 1), (1, 2), (2, 0)]
            },
            "constraints": {}
        }
    }
    tdr = TaskDecompositionRouting()
    routing_info = tdr.run(test_working_memory)
    print("Routing Info:", routing_info)
