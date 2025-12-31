import aiohttp
import asyncio

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

requests_data = [
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 100,
        "temperature": 0.7
    },
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "What is the capital of Germany?"}],
        "max_tokens": 100,
        "temperature": 0.7
    },
    # Add more requests as needed
]

async def send_request(session, data):
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, data) for data in requests_data]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"Response {i+1}: {result}")

asyncio.run(main())
