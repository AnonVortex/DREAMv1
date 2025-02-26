#by Alexis Soto-Yanez
"""
Step 1: Data Ingestion and Preprocessing for HMAS (AGI Prototype)

This script demonstrates real data ingestion and preprocessing.
It uses OpenCV and librosa to load image and audio files,
and simulates data for smell, touch, and taste.
The processing steps include:
1. Ingestion: Loading raw data from disk/sensors.
2. Cleaning: Removing noise (e.g., applying a Gaussian blur to images).
3. Normalization: Scaling data to standard ranges.
4. Encoding: Transforming data into machine-readable feature vectors.
5. Synchronization: Aligning multi-modal data (placeholder implementation).
"""

import cv2
import librosa
import numpy as np

class DataIngestor:
    def ingest(self):
        # Vision: Attempt to load an image file.
        image = cv2.imread("sample_image.jpg")
        if image is None:
            print("[DataIngestor] Warning: 'sample_image.jpg' not found. Using placeholder image.")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Audition: Attempt to load an audio file.
        try:
            audio, sr = librosa.load("sample_audio.wav", sr=None)
        except Exception as e:
            print("[DataIngestor] Warning: 'sample_audio.wav' not found. Using placeholder audio.")
            audio = np.zeros(22050)  # 1 second of silence at 22050 Hz
            sr = 22050
        
        # For smell, touch, taste: Simulate sensor data as random arrays.
        smell = np.random.rand(10)
        touch = np.random.rand(5)
        taste = np.random.rand(3)
        
        raw_data = {
            "vision": image,
            "audition": (audio, sr),
            "smell": smell,
            "touch": touch,
            "taste": taste
        }
        print("[DataIngestor] Real data ingested.")
        return raw_data

class DataCleaner:
    def clean(self, raw_data):
        # For vision: Apply Gaussian blur to reduce image noise.
        image = raw_data.get("vision")
        cleaned_image = cv2.GaussianBlur(image, (5, 5), 0) if image is not None else image
        
        # For audition: (Placeholder) Assume audio is clean.
        audio, sr = raw_data.get("audition", (None, None))
        cleaned_audio = audio  # Extend with real noise reduction as needed.
        
        # For smell, touch, taste: Assume data is already in a clean state.
        cleaned_data = {
            "vision": cleaned_image,
            "audition": (cleaned_audio, sr),
            "smell": raw_data.get("smell"),
            "touch": raw_data.get("touch"),
            "taste": raw_data.get("taste")
        }
        print("[DataCleaner] Data cleaned.")
        return cleaned_data

class DataNormalizer:
    def normalize(self, cleaned_data):
        # For vision: Normalize pixel values to the range [0,1].
        image = cleaned_data.get("vision")
        normalized_image = image.astype(np.float32) / 255.0 if image is not None else image
        
        # For audition: Normalize audio amplitude.
        audio, sr = cleaned_data.get("audition", (None, None))
        if audio is not None and np.max(np.abs(audio)) > 0:
            normalized_audio = audio / np.max(np.abs(audio))
        else:
            normalized_audio = audio
        
        # For smell, touch, taste: Normalize each to [0,1] using min-max scaling.
        smell = cleaned_data.get("smell")
        normalized_smell = (smell - np.min(smell)) / np.ptp(smell) if smell is not None and np.ptp(smell) > 0 else smell
        
        touch = cleaned_data.get("touch")
        normalized_touch = (touch - np.min(touch)) / np.ptp(touch) if touch is not None and np.ptp(touch) > 0 else touch
        
        taste = cleaned_data.get("taste")
        normalized_taste = (taste - np.min(taste)) / np.ptp(taste) if taste is not None and np.ptp(taste) > 0 else taste
        
        normalized_data = {
            "vision": normalized_image,
            "audition": (normalized_audio, sr),
            "smell": normalized_smell,
            "touch": normalized_touch,
            "taste": normalized_taste
        }
        print("[DataNormalizer] Data normalized.")
        return normalized_data

class DataEncoder:
    def encode(self, normalized_data):
        # For vision: Flatten the normalized image as a simple feature vector.
        image = normalized_data.get("vision")
        encoded_image = image.flatten() if image is not None else image
        
        # For audition: Compute MFCCs from audio and flatten the result.
        audio, sr = normalized_data.get("audition", (None, None))
        if audio is not None:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            encoded_audio = mfcc.flatten()
        else:
            encoded_audio = audio
        
        # For smell, touch, taste: Use the normalized values directly.
        encoded_data = {
            "vision": encoded_image,
            "audition": encoded_audio,
            "smell": normalized_data.get("smell"),
            "touch": normalized_data.get("touch"),
            "taste": normalized_data.get("taste")
        }
        print("[DataEncoder] Data encoded.")
        return encoded_data

class DataSynchronizer:
    def synchronize(self, encoded_data):
        # Placeholder: In a full system, this would align data temporally or contextually.
        # Here, we simply pass the encoded data along.
        synchronized_data = encoded_data
        print("[DataSynchronizer] Data synchronized.")
        return synchronized_data

class DataIngestionPreprocessing:
    def __init__(self):
        self.ingestor = DataIngestor()
        self.cleaner = DataCleaner()
        self.normalizer = DataNormalizer()
        self.encoder = DataEncoder()
        self.synchronizer = DataSynchronizer()

    def run(self):
        raw_data = self.ingestor.ingest()
        cleaned_data = self.cleaner.clean(raw_data)
        normalized_data = self.normalizer.normalize(cleaned_data)
        encoded_data = self.encoder.encode(normalized_data)
        preprocessed_data = self.synchronizer.synchronize(encoded_data)
        print("[DataIngestionPreprocessing] Data ingestion and preprocessing complete.")
        return preprocessed_data

if __name__ == "__main__":
    pipeline = DataIngestionPreprocessing()
    preprocessed_data = pipeline.run()
    print("Preprocessed Data Keys:", preprocessed_data.keys())
