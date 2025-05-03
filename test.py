import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    url = f"{BASE_URL}/health"
    try:
        resp = requests.get(url, timeout=5)
        print(f"[HEALTH] Status code: {resp.status_code}")
        print(f"[HEALTH] Response JSON: {resp.json()}")
    except Exception as e:
        print(f"[HEALTH] Request failed: {e}")

def test_predict(sample_text):
    url = f"{BASE_URL}/predict"
    headers = {"Content-Type": "application/json"}
    payload = {"text": sample_text}
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        print(f"[PREDICT] Status code: {resp.status_code}")
        # If JSON parse fails, print raw text
        try:
            print(f"[PREDICT] Response JSON: {resp.json()}")
        except json.JSONDecodeError:
            print(f"[PREDICT] Response Text: {resp.text}")
    except Exception as e:
        print(f"[PREDICT] Request failed: {e}")

if __name__ == "__main__":
    print("=== Testing /health ===")
    test_health()
    print("\n=== Testing /predict ===")
    test_predict("I absolutely loved this product—highly recommend!")
    test_predict("This was the worst experience I've ever had.")
