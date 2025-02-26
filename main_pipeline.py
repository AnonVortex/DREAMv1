# by Alexis Soto-Yanez
"""
Main Pipeline Integration for HMAS (AGI Prototype)

This script integrates all steps of the HMAS pipeline into one end-to-end process:
  1. Data Ingestion and Preprocessing
  2. Multi-Sensory Perception
  3. Integration and Working Memory
  4. Task Decomposition and Routing
  5. Specialized Processing Agents
  6. Intermediate Evaluation and Meta-Cognition
  7. Long-Term Memory and Knowledge Base
  8. Decision Aggregation and Output Generation
  9. Feedback Loop and Continuous Learning
 10. Monitoring, Maintenance, and Scalability

Ensure that each module (step1 through step10) is in your Python path or same directory.
"""

# Import modules from each step.
# These imports assume that you have saved each step's script with the given file names.
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

def main():
    print("\n--- HMAS Pipeline: End-to-End Integration Start ---\n")
    
    # Step 1: Data Ingestion and Preprocessing
    print(">> Step 1: Data Ingestion and Preprocessing")
    preprocessed_data = ingestion.DataIngestionPreprocessing().run()
    
    # Step 2: Multi-Sensory Perception
    print("\n>> Step 2: Multi-Sensory Perception")
    perception_results = perception.PerceptionSystem().process(preprocessed_data)
    
    # Step 3: Integration and Working Memory
    print("\n>> Step 3: Integration and Working Memory")
    working_memory = integration.IntegrationAndMemory().run(perception_results)
    
    # Step 4: Task Decomposition and Routing
    print("\n>> Step 4: Task Decomposition and Routing")
    routing_info = task_decomp.TaskDecompositionRouting().run(working_memory)
    
    # Step 5: Specialized Processing Agents
    print("\n>> Step 5: Specialized Processing Agents")
    specialized_outputs = specialized.SpecializedProcessing().run(routing_info, working_memory)
    
    # Step 6: Intermediate Evaluation and Meta-Cognition
    print("\n>> Step 6: Intermediate Evaluation and Meta-Cognition")
    evaluation_report = meta.MetaCognition().run(specialized_outputs)
    
    # Step 7: Long-Term Memory and Knowledge Base
    print("\n>> Step 7: Long-Term Memory and Knowledge Base")
    long_term_memory = memory.LongTermMemory()
    long_term_memory.run(specialized_outputs)
    
    # Step 8: Decision Aggregation and Output Generation
    print("\n>> Step 8: Decision Aggregation and Output Generation")
    final_output = decision.DecisionAggregation().run(specialized_outputs, evaluation_report)
    print("\nFinal Decision Output:")
    print(final_output)
    
    # Step 9: Feedback Loop and Continuous Learning
    print("\n>> Step 9: Feedback Loop and Continuous Learning")
    feedback.FeedbackLoop().run(final_output)
    
    # Step 10: Monitoring, Maintenance, and Scalability
    print("\n>> Step 10: Monitoring, Maintenance, and Scalability")
    monitor = monitoring.Monitoring()
    monitor.run("FinalPipeline", "Running complete end-to-end pipeline.")
    
    print("\n--- HMAS Pipeline: End-to-End Integration Complete ---\n")

if __name__ == "__main__":
    main()
