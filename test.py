import requests
import json

BASE_URL = "http://localhost:5000"

def test_version():
    url = f"{BASE_URL}/version"
    try:
        resp = requests.get(url, timeout=5)
        print(f"[VERSION] Status code: {resp.status_code}")
        print(f"[VERSION] Response JSON: {resp.json()}")
    except Exception as e:
        print(f"[VERSION] Request failed: {e}")

def test_analyze(sample_review):
    url = f"{BASE_URL}/analyze"
    headers = {"Content-Type": "application/json"}
    payload = {"review": sample_review}
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        print(f"[ANALYZE] Status code: {resp.status_code}")
        try:
            print(f"[ANALYZE] Response JSON: {resp.json()}")
        except json.JSONDecodeError:
            print(f"[ANALYZE] Response Text: {resp.text}")
    except Exception as e:
        print(f"[ANALYZE] Request failed: {e}")

if __name__ == "__main__":
    print("=== Testing /version ===")
    test_version()
    print("\n=== Testing /analyze ===")
    test_analyze("The food was decent but the service could be better.")
    test_analyze("This was the worst experience I've ever had.")
    print("\n=== Testing error case (missing field) ===")
    # send empty payload to trigger 400
    try:
        resp = requests.post(f"{BASE_URL}/analyze", headers={"Content-Type": "application/json"}, data=json.dumps({}), timeout=5)
        print(f"[ANALYZE-ERROR] Status code: {resp.status_code}")
        print(f"[ANALYZE-ERROR] Response JSON: {resp.json()}")
    except Exception as e:
        print(f"[ANALYZE-ERROR] Request failed: {e}")
