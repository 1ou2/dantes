import json
import asyncio
import aiohttp
from tqdm import tqdm
from llm import AsyncLLM, LLM

def create_prompt(context, citation):
      return f"""Tu es chargé de transformer la donnée suivante en 3 paires de question-réponse utilisable pour un dataset d'instruction tuning.
      Toutes les citations sont prononcées par Edmond Dantès (le comte de monte-cristo).

      Donnée :
      Contexte : {context}
      Citation originale : {citation}

      Tâches :
      1. Génère 3 questions plausibles qu'un interlocuteur poserait pour provoquer cette réponse.
      2. Pour chaque réponse, DÉVELOPPE et ÉLABORE sur la citation originale :
         - Commence par la pensée ou l'idée de la citation
         - Ajoute du contexte, des explications, des réflexions supplémentaires
         - Développe les métaphores ou les idées sous-jacentes
         - Vise des réponses de 3-5 phrases minimum (50-150 mots)
      3. Conserve le style d'Edmond Dantès / Monte-Cristo : ton soutenu, métaphores, politesse, contrôle, mystère.
      4. La réponse doit être substantielle et complète, pas juste une reformulation courte.
      5. Fournis un JSONL strict au format :
      {{"instruction": "...", "response": "..."}}
      {{"instruction": "...", "response": "..."}}
      {{"instruction": "...", "response": "..."}}

      Contraintes :
      - Les réponses doivent être développées (minimum 3-5 phrases)
      - Pas de répétition brute du passage sans adaptation
      - Pas de commentaires ou d'analyse méta
      - Style formel du XIXᵉ siècle
      - Ta réponse ne contient que les messages JSON pas de formattage ou de commentaire."""


def get_all_prompts():
    prompts = []
    with open("data/citations/dantes.jsonl","r",encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            if not line:
                continue
            data = json.loads(line)
            context = data["contexte"]
            citation = data["citation"]
            prompts.append(create_prompt(context,citation))
    return prompts

def process_response(response:str):
    with open("instructions-result-long.txt","a",encoding="utf-8") as f:
        f.write(response+"\n")

async def main():
    llm = AsyncLLM(
        url="http://localhost:8000/v1/chat/completions",
        model="openai/gpt-oss-20b",
        system_prompt="Tu réponds en français."
    )
    prompts = get_all_prompts()
    print(f"Number of prompts to process: {len(prompts)}")
    semaphore = asyncio.Semaphore(3)

    tasks = [llm.get_response(p,semaphore) for p in prompts]

    # Wrap as_completed with tqdm for progress tracking
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing prompts"):
        result = await future
        process_response(result)

def sync_main():
    url = "http://0.0.0.0:12001/v1/chat/completions"
    model = "gpt-oss-20b-default"
    llm = LLM(url, model, "Tu es un assistant littéraire, spécialisé dans le comte de Monte-cristo. Tu t’exprimes en français.")
    prompts = get_all_prompts()[1249:]
    print(f"Number of prompts to process {len(prompts)}")
    for prompt in tqdm(prompts):
        response = llm.get_response(prompt)
        process_response(response)
    print("Finished processing all prompts.")

if __name__ == "__main__":
    asyncio.run(main())

    #sync_main()