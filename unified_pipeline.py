# by Alexis Soto-Yanez
"""
Unified HMAS Pipeline Integration for AGI Prototype

This script integrates:
  - Lower-Level Pipeline: Data Ingestion, Perception, Integration, Task Decomposition, 
    Specialized Processing, Meta-Cognition, Long-Term Memory, Decision Aggregation,
    Feedback Loop, and Monitoring.
  - Higher-Level Collaboration: Grouping agents into teams, organizations, and a federation.
  
The lower-level pipeline processes multimodal input to produce a final decision output.
That output is then fed as input to the higher-level collaboration module (federation)
to produce a unified output across multiple organizations.

Run this script to observe end-to-end behavior from raw data to a federated final output.
"""

# -------------------------------
# Lower-Level Pipeline Modules
# -------------------------------
import step1_data_ingestion as ingestion
import step2_multi_sensory_perception as perception
import step3_integration_working_memory as integration
import step4_task_decomposition_routing as task_decomp
import step5_specialized_processing as specialized
import step6_meta_cognition as meta
import step7_long_term_memory as memory
import step8_decision_aggregation as decision
import step9_feedback_loop as feedback
import step10_monitoring_maintenance_scalability as monitoring

# -------------------------------
# Higher-Level Collaboration Modules
# -------------------------------
# These modules were defined in our higher-level collaboration examples.
from higher_level_collaboration import BaseAgent, Team, Organization
from federation_collaboration import Federation

def main():
    print("\n--- Unified HMAS Pipeline Integration Start ---\n")
    
    # === LOWER-LEVEL PIPELINE EXECUTION ===
    print(">> Step 1: Data Ingestion and Preprocessing")
    preprocessed_data = ingestion.DataIngestionPreprocessing().run()
    
    print("\n>> Step 2: Multi-Sensory Perception")
    perception_results = perception.PerceptionSystem().process(preprocessed_data)
    
    print("\n>> Step 3: Integration and Working Memory")
    working_memory = integration.IntegrationAndMemory().run(perception_results)
    
    print("\n>> Step 4: Task Decomposition and Routing")
    routing_info = task_decomp.TaskDecompositionRouting().run(working_memory)
    
    print("\n>> Step 5: Specialized Processing Agents")
    specialized_outputs = specialized.SpecializedProcessing().run(routing_info, working_memory)
    
    print("\n>> Step 6: Intermediate Evaluation and Meta-Cognition")
    evaluation_report = meta.MetaCognition().run(specialized_outputs)
    
    print("\n>> Step 7: Long-Term Memory and Knowledge Base")
    long_term_memory = memory.LongTermMemory()
    long_term_memory.run(specialized_outputs)
    
    print("\n>> Step 8: Decision Aggregation and Output Generation")
    final_output = decision.DecisionAggregation().run(specialized_outputs, evaluation_report)
    print("\nLower-Level Final Decision Output:")
    print(final_output)
    
    print("\n>> Step 9: Feedback Loop and Continuous Learning")
    feedback.FeedbackLoop().run(final_output)
    
    print("\n>> Step 10: Monitoring, Maintenance, and Scalability")
    monitor = monitoring.Monitoring()
    monitor.run("UnifiedPipeline", "Lower-level pipeline complete.")
    
    # === HIGHER-LEVEL COLLABORATION (FLEEDERATION) ===
    print("\n>> Higher-Level Collaboration: Federation Integration")
    # Here, we use the final output from the lower-level pipeline as the input for higher-level agents.
    # For demonstration, we simulate two organizations each with a team of agents.
    
    # Organization 1
    agent1 = BaseAgent("Org1_Agent1")
    agent2 = BaseAgent("Org1_Agent2")
    team1 = Team("Org1_Team", [agent1, agent2])
    organization1 = Organization("Organization1", [team1])
    
    # Organization 2
    agent3 = BaseAgent("Org2_Agent1")
    agent4 = BaseAgent("Org2_Agent2")
    team2 = Team("Org2_Team", [agent3, agent4])
    organization2 = Organization("Organization2", [team2])
    
    # Create a federation that aggregates outputs from both organizations.
    federation = Federation([organization1, organization2])
    
    # Use the lower-level final output as input to the federation.
    federation_final_output = federation.collaborate(final_output)
    print("\nFederation Final Output:")
    print(federation_final_output)
    
    print("\n--- Unified HMAS Pipeline Integration Complete ---\n")

if __name__ == "__main__":
    main()
