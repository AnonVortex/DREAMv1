# main_pipeline.py
# by Alexis Soto-Yanez
"""
Main Pipeline Module for HMAS Prototype

This module integrates all steps of the HMAS pipeline:
  1. Data Ingestion & Preprocessing
  2. Multi-Sensory Perception
  3. Integration & Working Memory
  4. Task Decomposition & Routing
  5. Specialized Processing Agents
  6. Intermediate Evaluation & Meta-Cognition
  7. Long-Term Memory & Knowledge Base
  8. Decision Aggregation & Output Generation
  9. Feedback Loop & Continuous Learning
 10. Monitoring, Maintenance & Scalability

At the end, it prints "Pipeline execution complete" to indicate success.
"""

import logging

# Import each pipeline step module.
import step1_data_ingestion
import step2_multi_sensory_perception
import step3_integration_working_memory
import step4_task_decomposition_routing as task_decomp
import step5_specialized_processing
import step6_meta_cognition
import step7_long_term_memory
import step8_decision_aggregation
import step9_feedback_loop
import step10_monitoring_maintenance_scalability

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("\n--- HMAS Pipeline: End-to-End Integration Start ---\n")
    
    # Step 1: Data Ingestion & Preprocessing
    logging.info(">> Step 1: Data Ingestion and Preprocessing")
    working_memory = step1_data_ingestion.main()
    
    # Step 2: Multi-Sensory Perception
    logging.info(">> Step 2: Multi-Sensory Perception")
    perception_output = step2_multi_sensory_perception.main()
    
    # Step 3: Integration & Working Memory
    logging.info(">> Step 3: Integration and Working Memory")
    integration_output = step3_integration_working_memory.main()
    working_memory.update(integration_output)
    
    # Step 4: Task Decomposition & Routing
    logging.info(">> Step 4: Task Decomposition and Routing")
    # Ensure working_memory has a goal (for testing, we add a placeholder if missing).
    if "goal" not in working_memory:
        working_memory["goal"] = {
            "type": "graph_optimization",
            "raw_data": {
                "nodes": [
                    {"id": 0, "features": [0.8] * 10},
                    {"id": 1, "features": [0.3] * 10},
                    {"id": 2, "features": [0.5] * 10}
                ],
                "edges": [(0, 1), (1, 2), (2, 0)]
            },
            "constraints": {}
        }
    routing_info = task_decomp.TaskDecompositionRouting().run(working_memory)
    
    # Step 5: Specialized Processing Agents
    logging.info(">> Step 5: Specialized Processing Agents")
    step5_specialized_processing.main()
    
    # Step 6: Intermediate Evaluation & Meta-Cognition
    logging.info(">> Step 6: Intermediate Evaluation and Meta-Cognition")
    step6_meta_cognition.main()
    
    # Step 7: Long-Term Memory & Knowledge Base
    logging.info(">> Step 7: Long-Term Memory and Knowledge Base")
    step7_long_term_memory.main()
    
    # Step 8: Decision Aggregation & Output Generation
    logging.info(">> Step 8: Decision Aggregation and Output Generation")
    step8_decision_aggregation.main()
    
    # Step 9: Feedback Loop & Continuous Learning
    logging.info(">> Step 9: Feedback Loop and Continuous Learning")
    step9_feedback_loop.main()
    
    # Step 10: Monitoring, Maintenance & Scalability
    logging.info(">> Step 10: Monitoring, Maintenance, and Scalability")
    step10_monitoring_maintenance_scalability.main()
    
    print("\nPipeline execution complete")
    return "Pipeline execution complete"

if __name__ == "__main__":
    main()
