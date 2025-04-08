import subprocess
import time
import requests

def test_docker():
    print("🔧 Running Docker container test...")

    # Step 1: Build the Docker image
    subprocess.run(["docker", "build", "-t", "flask-score-app", "."], check=True)

    # Step 2: Start the Docker container
    subprocess.run([
        "docker", "run", "--rm", "-d",
        "-p", "5000:5000",
        "--name", "test-container",
        "flask-score-app"
    ], check=True)

    # Allow time for the app to start
    time.sleep(5)

    try:
        test_payload = {
            "text":
                "Todays Vodafone numbers ending with 4882 are selected to a receive a £350 award. If your number matches call 09064019014 to receive your Â£350 award."
            }

        url = "http://localhost:5000/score"
        res = requests.post(url, json=test_payload)

        assert res.status_code == 200
        result = res.json()

        assert "prediction" in result
        assert "propensity" in result

        print("✅ Docker test successful!")
        print(f"📊 Prediction: {result['prediction']}")
        print(f"📈 Propensity: {result['propensity']}")

    finally:
        # Clean up: stop the container
        subprocess.run(["docker", "stop", "test-container"])
