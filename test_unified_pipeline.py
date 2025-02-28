# by Alexis Soto-Yanez

import unittest
import subprocess

class TestUnifiedPipeline(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        proc = subprocess.run(
            ["python", "C:/Users/sotoy/OneDrive/Desktop/H-MAS(AGI DREAM)/main_pipeline.py"],
            capture_output=True,
            text=True
        )

        # Print output to help debug
        print("=== STDOUT ===")
        print(proc.stdout)
        print("=== STDERR ===")
        print(proc.stderr)

        # Check exit code
        self.assertEqual(proc.returncode, 0, f"Pipeline script failed with error code {proc.returncode}.\nStdErr:\n{proc.stderr}")

        # Check for success message (optional)
        self.assertIn("Pipeline execution complete", proc.stdout)

if __name__ == "__main__":
    unittest.main()
