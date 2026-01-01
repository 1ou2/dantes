import requests
import aiohttp
import asyncio
from contextlib import nullcontext

def format_messages(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def parse_response(result)->str:
    if "choices" not in result or not result["choices"]:
        import json
        error_msg = f"La réponse ne contient pas de choix valide. Réponse reçue: {json.dumps(result, indent=2)}"
        raise ValueError(error_msg)
    content = result["choices"][0]["message"]["content"]
    #reasoning = result["choices"][0]["message"].get("reasoning_content", "")

    return content

class LLM:
    def __init__(self, url, model, system_prompt):
        self.url = url
        self.model = model
        self.system_prompt = system_prompt

    def get_response(self, prompt):
        messages = format_messages(self.system_prompt, prompt)

        data = {
            "messages":messages,
            "model": self.model,
            "temperature": 0.7,  
        }
        response = requests.post(self.url, json=data)
        result = response.json()
        return parse_response(result)

class AsyncLLM:
    def __init__(self, url="http://localhost:8000/v1/chat/completions", 
                 model="mistralai/Mistral-7B-Instruct-v0.3", 
                 system_prompt=None):
        self.url = url
        self.model = model
        self.system_prompt = system_prompt

    async def get_response(self, prompt,semaphore=None)->str:
        messages = format_messages(self.system_prompt, prompt)
        data = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
        }
        # Use a null context manager if no semaphore is provided
        sem_ctx = semaphore if semaphore else nullcontext()
        
        async with sem_ctx:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=data) as response:
                    result = await response.json()
                    return parse_response(result)
       

async def main():
    async_llm = AsyncLLM(
        url="http://localhost:8000/v1/chat/completions",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        system_prompt="You are a helpful assistant."
    )
    response = await async_llm.get_response("Hello, how are you?")
    print(response)

if __name__ == "__main__":
    # URL de l'API locale
    HOST="http://0.0.0.0"
    PORT=12001

    url = f"{HOST}:{PORT}/v1/chat/completions"
    model = "gpt-oss-20b-default"
    #model = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
    #model = "mistralai/Mistral-7B-Instruct-v0.3"
    sp = "you are an helpful assistant"

    sp = "tu es un Edmond Dantès, tu réponds avec ses idées et tu t’exprimes comme lui."
    up = "Explique ce qu’est le finetuning des LLM."

    system_prompt = {"role":"system","content":f"{sp}"}
    user_prompt = {"role":"user","content":f"{up}"}


    messages = [system_prompt,user_prompt]

    # Paramètres de la requête
    data = {
        "messages":messages,
        "model": model,
        #"n_predict": 128,  # Nombre de tokens à générer
        "temperature": 0.7,  # Contrôle la créativité (0.0 = déterministe, 1.0 = aléatoire)
    }

    # Envoi de la requête POST
    response = requests.post(url, json=data)

    # Affichage de la réponse
    if response.status_code == 200:
        print(response.json())
    else:
        print("Erreur:", response.status_code, response.text)

    result = response.json()
    if "choices" not in result or not result["choices"]:
                raise ValueError("La réponse ne contient pas de choix valide.")

    # Extraction du contenu
    content = result["choices"][0]["message"]["content"]
    reasoning = result["choices"][0]["message"].get("reasoning_content", "")

    print("--- REASONING ---")
    print(reasoning)
    print("\n--- RESPONSE ----")
    print(content)
    
    # ASYNC example
    #asyncio.run(main())