import asyncio, aiohttp

prompts = [
    "why sky is blue?",
    "Give me a quick summary of quantum computing.",
    "How does dynamic batching work?",
    "What is the capital of Iran?"
]
#   "prompt": [
#       "What is the capital of Iran?",
#       "what is quantum computing?",
#       "why sky is blue?",
async def ask(prompt):
    async with aiohttp.ClientSession() as sess:
        payload = {
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9
        }
        async with sess.post("http://localhost:8000/v1/completions",
                             json=payload) as resp:
            data = await resp.json()
            # print(f"Prompt:   {prompt}\nResponse: {data}\n")
            # extract the first choice’s text
            text = data["choices"][0]["text"]
            print(f"Prompt:   {prompt}\nResponse: {text}\n")

async def main():
    # fire all four at once
    await asyncio.gather(*(ask(p) for p in prompts))

if __name__ == "__main__":
    asyncio.run(main())
