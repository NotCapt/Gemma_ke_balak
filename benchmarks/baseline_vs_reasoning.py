import time
import requests
import json

GEMMA_API_URL = "http://localhost:8000/classify"

test_cases = [
    "I lost my child near the main gate.",
    "People are pushing aggressively at the food stalls.",
    "My grandmother fainted in the heat."
]

def benchmark_reasoning():
    print("🚀 Starting Benchmark: Baseline vs Reasoning Output")
    print("=" * 60)
    
    total_time = 0
    
    for case in test_cases:
        print(f"\n📝 Input: '{case}'")
        
        payload = {
            "text": case,
            "max_tokens": 150
        }
        
        start_time = time.time()
        try:
            response = requests.post(GEMMA_API_URL, json=payload, timeout=30)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if response.status_code == 200:
                result = response.json()
                print(f"⏱️  Latency: {elapsed:.2f}s")
                print(f"📊 Result:")
                print(json.dumps(result, indent=2))
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Connection error (is GemmaServer running?): {e}")
            
    if len(test_cases) > 0:
        print("\n" + "=" * 60)
        print(f"📈 Average Latency: {total_time/len(test_cases):.2f}s per request")

if __name__ == "__main__":
    benchmark_reasoning()
