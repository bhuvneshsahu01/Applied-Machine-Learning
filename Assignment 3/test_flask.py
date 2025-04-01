import unittest
import requests
import subprocess
import time

class TestFlaskAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize the Flask server before executing tests."""
        cls.server_process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Allow time for the server to start
        time.sleep(5)  # Adjust if needed

    @classmethod
    def tearDownClass(cls):
        """Shut down the Flask server after testing."""
        cls.server_process.terminate()
        cls.server_process.wait()  # Ensure complete shutdown

    def test_flask_api(self):
        """Verify API response correctness."""
        url = "http://127.0.0.1:5000/score"
        payload = {"text": "Call FREEPHONE 8005420578 now!, to win bike", "threshold": 0.7}

        response = requests.post(url, json=payload)

        # Check if the server responds correctly
        self.assertEqual(response.status_code, 200, "Expected 200 OK response")

        data = response.json()

        # Validate response keys
        self.assertIn("prediction", data)
        self.assertIn("propensity", data)

        # Ensure correct data types
        self.assertIsInstance(data["prediction"], int)
        self.assertIsInstance(data["propensity"], float)

        # Ensure probability is within a valid range
        self.assertGreaterEqual(data["propensity"], 0)
        self.assertLessEqual(data["propensity"], 1)

if __name__ == '__main__':
    unittest.main()
