# by Alexis Soto-Yanez
"""
Data Ingestion and Preprocessing Module for HMAS Prototype

This module ingests multi-modal data (images, audio, text, etc.), applies
preprocessing (cleaning, normalization, etc.), and returns a working memory
dictionary that will be used by subsequent modules in the HMAS pipeline.
"""

import cv2
import librosa
import numpy as np
import logging

# Set up logging for debugging purposes.
logging.basicConfig(level=logging.INFO)

def ingest_data():
    """
    Ingests raw data from various sources. In this example, we attempt to load
    an image and an audio file. If they are not found, placeholder data is used.
    
    Returns:
        dict: A dictionary containing raw data for each modality.
    """
    # Attempt to read an image using OpenCV.
    image = cv2.imread('sample_image.jpg')
    if image is None:
        logging.warning("sample_image.jpg not found. Using placeholder image.")
        # Create a placeholder 100x100 black image.
        image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Attempt to load an audio file using librosa.
    try:
        audio, sr = librosa.load("sample_audio.wav", sr=None)
    except Exception as e:
        logging.warning(f"sample_audio.wav not found or error loading audio: {e}. Using placeholder audio.")
        # Create 1 second of silence at 22050 Hz.
        sr = 22050
        audio = np.zeros(sr, dtype=np.float32)
    
    # For text data, use a placeholder string.
    text_data = "Placeholder text data for HMAS."
    
    return {
        "vision": image,
        "audition": {"audio": audio, "sr": sr},
        "text": text_data
    }

def preprocess_data(data):
    """
    Preprocess the raw data. For the image, normalize pixel values to [0, 1].
    
    Parameters:
        data (dict): Raw data dictionary.
    
    Returns:
        dict: Processed data dictionary.
    """
    if data.get("vision") is not None:
        # Normalize image pixel values to [0, 1].
        image = data["vision"].astype(np.float32) / 255.0
        data["vision"] = image
    # Additional preprocessing steps for other modalities can be added here.
    return data

def main():
    """
    Main function to ingest and preprocess data.
    
    Returns:
        dict: A working memory dictionary with processed data.
    """
    logging.info("Starting data ingestion...")
    raw_data = ingest_data()
    processed_data = preprocess_data(raw_data)
    logging.info("Data ingestion and preprocessing complete.")
    
    # Construct the working memory dictionary. In a real scenario, you might add more keys.
    working_memory = {
        "vision": processed_data.get("vision"),
        "audition": processed_data.get("audition"),
        "text": processed_data.get("text"),
        # For testing purposes, include a placeholder goal.
        "goal": {
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
    }
    return working_memory

if __name__ == "__main__":
    wm = main()
    print("Working Memory:", wm)

