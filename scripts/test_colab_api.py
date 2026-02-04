import requests
import time
import json
import argparse
import sys

def test_api(base_url):
    print(f"Testing API at {base_url}...")
    
    # Clean up URL
    base_url = base_url.rstrip('/')
    
    # 1. Health check
    try:
        resp = requests.get(f"{base_url}/health")
        if resp.status_code == 200:
            print("Health check passed:", resp.json())
        else:
            print(f"Health check failed: {resp.status_code} - {resp.text}")
            return
    except Exception as e:
        print(f"Failed to connect to {base_url}: {e}")
        return

    # 2. Release Task (Generate Music)
    print("\nSending generation task...")
    payload = {
        "prompt": "An upbeat 8-bit chip tune",
        "duration": 10, # Short duration for testing
        "inference_steps": 4, # Fast generation
        "thinking": False, # Disable LM thinking for speed
    }
    
    try:
        resp = requests.post(f"{base_url}/release_task", json=payload)
        if resp.status_code != 200:
            print(f"Task release failed: {resp.status_code} - {resp.text}")
            return
        
        data = resp.json()
        if data.get('code') == 200:
            task_id = data['data']['task_id']
            print(f"Task started successfully. Task ID: {task_id}")
        else:
            print(f"Task release error: {data}")
            return
            
    except Exception as e:
        print(f"Error sending task: {e}")
        return

    # 3. Query Result
    # Note: In api_routes.py implementation, release_task waits for generation,
    # so query_result should have the result immediately.
    print(f"\nQuerying result for task {task_id}...")
    
    try:
        # api_routes.py query_result expects task_id_list
        query_payload = {
            "task_id_list": [task_id]
        }
        resp = requests.post(f"{base_url}/query_result", json=query_payload)
        
        if resp.status_code == 200:
            data = resp.json()
            # print("Query raw response:", json.dumps(data, indent=2))
            
            if data.get('code') == 200:
                results = data['data']
                for res in results:
                    if res.get('status') == 1:
                        print("Generation succeeded!")
                        result_details = json.loads(res['result'])
                        # print(json.dumps(result_details, indent=2))
                        
                        # Find audio URL
                        for item in result_details:
                           if 'url' in item:
                               full_url = f"{base_url}{item['url']}"
                               print(f"Audio available at: {full_url}")
                    else:
                        print("Task status is not 'succeeded' (1).")
        else:
             print(f"Query Result failed: {resp.status_code} - {resp.text}")

    except Exception as e:
        print(f"Error querying result: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ACE-Step Colab API")
    parser.add_argument("url", nargs="?", help="Colab Gradio URL (e.g., https://xxxx.gradio.live)")
    
    args = parser.parse_args()
    
    url = args.url
    if not url:
        url = input("Enter the Colab Gradio URL (e.g., https://xxxx.gradio.live): ")
    
    if url:
        test_api(url)
    else:
        print("No URL provided.")
