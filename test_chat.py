# test_chat.py
import httpx

def main():
    client = httpx.Client(base_url="http://localhost:8000")

    print("🦙 vLLM chat (blank line to exit)\n")
    while True:
        prompt = input("You ► ")
        if not prompt.strip():
            break

        resp = client.post(
            "/v1/chat/completions",     # <-- note “chat/completions”
            json={
                "model":       "microsoft/phi-4-mini-instruct",
                "messages":   [
                    {"role":"system","content":"You are a helpful assistant."},
                    {"role":"user"  ,"content":prompt}
                ],
                "max_tokens": 128,
                "temperature":0.7
            },
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"].lstrip("\n")
        print(f"vLLM ► {reply}\n")

if __name__ == "__main__":
    main()
