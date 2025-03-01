# step9_feedback_loop.py
# by Alexis Soto-Yanez
"""
Feedback Loop & Continuous Learning Module for HMAS Prototype

This module collects performance and communication metrics from the pipeline,
updates system parameters accordingly, and applies reinforcement learning (RL)
updates to improve overall system performance. This enhanced version includes:
  - Expanded metric collection (accuracy, latency, error_rate, plus communication metrics).
  - Adaptive updating that factors in communication performance.
  - A simulated RL update process.
"""

import logging
import random
import time

# Configure logging.
logging.basicConfig(level=logging.INFO)

class FeedbackCollector:
    def collect(self):
        """
        Simulate collection of feedback metrics from various pipeline stages.
        
        Returns:
            dict: A dictionary with performance metrics and communication-specific metrics.
        """
        # Simulated performance metrics.
        performance_metrics = {
            "accuracy": round(random.uniform(0.85, 0.95), 3),
            "latency": round(random.uniform(0.2, 0.5), 3),
            "error_rate": round(random.uniform(0.02, 0.05), 3)
        }
        # Simulated communication metrics.
        communication_metrics = {
            "message_latency": round(random.uniform(0.1, 0.5), 3),
            "message_success_rate": round(random.uniform(0.8, 1.0), 3),
            "negotiation_time": round(random.uniform(0.05, 0.2), 3)
        }
        # Combine metrics.
        feedback = {**performance_metrics, **{"comm_" + k: v for k, v in communication_metrics.items()}}
        logging.info("Feedback collected: %s", feedback)
        return feedback

class AdaptiveUpdater:
    def update_parameters(self, feedback):
        """
        Updates system parameters based on feedback, including both performance
        and communication metrics.
        
        Parameters:
            feedback (dict): Feedback metrics.
        
        Returns:
            dict: Updated parameters.
        """
        # Update learning rate based on error_rate.
        lr = 0.001 * (1 - feedback["error_rate"])
        # Adjust batch size based on accuracy.
        batch_size = max(16, int(32 * feedback["accuracy"]))
        # Adjust update frequency based on latency and communication negotiation time.
        update_frequency = max(1, int(10 * (feedback["latency"] + feedback["comm_negotiation_time"]) / 2))
        
        updated_params = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "update_frequency": update_frequency
        }
        logging.info("System parameters updated based on feedback: %s", updated_params)
        return updated_params

class ReinforcementLearningModule:
    def apply_rl(self, updated_params):
        """
        Simulate applying reinforcement learning updates based on updated parameters.
        
        Parameters:
            updated_params (dict): Updated parameters from feedback.
        
        Returns:
            str: A status message indicating the RL update.
        """
        # Simulate processing delay.
        time.sleep(0.5)
        status = f"Reinforcement learning applied with parameters: {updated_params}"
        logging.info(status)
        return status

class PerformanceLogger:
    def log(self, metrics):
        """
        Log performance and communication metrics.
        
        Parameters:
            metrics (dict): The metrics to log.
        """
        logging.info("Performance metrics logged: %s", metrics)

def main():
    """
    Main function for the Feedback Loop & Continuous Learning module.
    
    Performs the following:
      1. Collects feedback metrics (performance and communication).
      2. Updates system parameters based on feedback.
      3. Applies reinforcement learning updates.
      4. Logs the performance metrics.
      
    Returns:
        dict: A summary of the feedback loop execution.
    """
    logging.info(">> Step 9: Feedback Loop and Continuous Learning")
    
    collector = FeedbackCollector()
    updater = AdaptiveUpdater()
    rl_module = ReinforcementLearningModule()
    perf_logger = PerformanceLogger()
    
    # Step 1: Collect metrics.
    feedback = collector.collect()
    
    # Step 2: Update parameters.
    updated_params = updater.update_parameters(feedback)
    
    # Step 3: Apply RL updates.
    rl_status = rl_module.apply_rl(updated_params)
    
    # Step 4: Log the performance metrics.
    perf_logger.log(feedback)
    
    summary = {
        "feedback": feedback,
        "updated_params": updated_params,
        "rl_status": rl_status
    }
    print("Feedback Loop Summary:", summary)
    return summary

if __name__ == "__main__":
    main()
