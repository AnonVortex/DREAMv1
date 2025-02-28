# step3_integration_working_memory.py
# by Alexis Soto-Yanez
"""
Integration & Working Memory Module for HMAS Prototype

This module fuses outputs from the multi-sensory perception layer into a unified
representation and stores it in a working memory buffer. In this example, we simulate
the fusion using placeholder numeric arrays.
"""

import logging
import numpy as np

# Configure logging.
logging.basicConfig(level=logging.INFO)

def fuse_features(features_list):
    """
    Fuses a list of feature vectors by computing their element-wise average.
    
    Parameters:
        features_list (list of np.array): List of numeric feature arrays.
    
    Returns:
        np.array: The averaged (fused) feature vector.
    """
    try:
        fused = np.mean(np.array(features_list), axis=0)
    except Exception as e:
        logging.error("Error fusing features: %s", e)
        fused = None
    return fused

class ContextBuffer:
    """
    A simple working memory buffer that stores the fused feature representation.
    """
    def __init__(self):
        self.buffer = None

    def store(self, fused_features):
        """
        Stores the fused features in the working memory.
        
        Parameters:
            fused_features: The fused feature vector.
        
        Returns:
            The stored feature vector.
        """
        self.buffer = fused_features
        return self.buffer

    def retrieve(self):
        """
        Retrieves the stored feature vector.
        
        Returns:
            The stored feature vector.
        """
        return self.buffer

def main():
    """
    Main function for the Integration & Working Memory module.
    
    Simulates the integration of multi-modal features produced by the perception layer.
    Returns a dictionary containing the unified 'fused' representation.
    """
    logging.info(">> Step 3: Integration and Working Memory")
    
    # Simulate placeholder outputs from the perception module.
    # In a real system, these would be derived from the outputs of step2.
    vision = np.array([0.1, 0.2, 0.3])
    audition = np.array([0.2, 0.3, 0.4])
    smell = np.array([0.3, 0.4, 0.5])
    touch = np.array([0.4, 0.5, 0.6])
    taste = np.array([0.5, 0.6, 0.7])
    
    features_list = [vision, audition, smell, touch, taste]
    fused_features = fuse_features(features_list)
    
    if fused_features is None:
        logging.error("Fusing features failed.")
        return {}
    
    logging.info("Fused features computed.")
    
    # Store the fused features in a working memory buffer.
    context_buffer = ContextBuffer()
    buffer_output = context_buffer.store(fused_features)
    logging.info("Fused features stored in working memory.")
    
    # Return a dictionary representing the working memory.
    return {"fused": buffer_output}

if __name__ == "__main__":
    output = main()
    print("Integration Output:", output)
