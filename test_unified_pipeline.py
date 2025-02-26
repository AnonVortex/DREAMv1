#by Alexis Soto-Yanez
import sys
import subprocess
import unittest

class TestUnifiedPipeline(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        # Run the unified pipeline as a subprocess and capture its output.
        result = subprocess.run(
            [sys.executable, "unified_pipeline.py"],
            capture_output=True,
            text=True
        )
        # Ensure the process completed successfully.
        self.assertEqual(result.returncode, 0, "Pipeline did not exit successfully.")
        
        output = result.stdout
        
        # Check that key phrases from each stage are present in the output.
        self.assertIn("Data ingestion and preprocessing complete", output)
        self.assertIn("All sensory modalities processed", output)
        self.assertIn("Working memory updated", output)
        self.assertIn("Task decomposition and routing complete", output)
        self.assertIn("Specialized processing complete", output)
        self.assertIn("Intermediate evaluation complete", output)
        self.assertIn("Long-term memory updated", output)
        self.assertIn("Decision aggregation complete", output)
        self.assertIn("Feedback loop complete", output)
        self.assertIn("Unified HMAS Pipeline Integration Complete", output)

if __name__ == "__main__":
    unittest.main()
