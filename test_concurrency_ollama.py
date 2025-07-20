import requests
import concurrent.futures
import time

OLLAMA_URL = "http://localhost:11434/v1/completions"
HEADERS = {"Content-Type": "application/json"}

prompts = [
       "why sky is blue?",
    "Give me a quick summary of quantum computing.",
    "How does dynamic batching work?",
    "What is the capital of Iran?"
]

def ask(prompt: str):
    payload = {
        "model": "phi4-mini:3.8b-q8_0",
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9
    }
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, headers=HEADERS)
    dt = time.time() - t0
    data = resp.json()
    # assuming Ollama returns choices like OpenAI
    text = data["choices"][0]["text"].strip()
    print(f"[{dt:.3f}s] Prompt: {prompt!r}\n         Response: {text!r}\n")

def main():
    # run all 4 asks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [ pool.submit(ask, p) for p in prompts ]
        # wait for all to finish
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
