import requests
import os
from datetime import datetime

class LLM:
    def __init__(self, url, model, system_prompt):
        self.url = url
        self.model = model
        self.system_prompt = system_prompt

    def get_response(self, prompt):
        system_prompt = {"role":"system","content":f"{self.system_prompt}"}
        user_prompt = {"role":"user","content":f"{prompt}"}

        messages = [system_prompt,user_prompt]

        # Paramètres de la requête
        data = {
            "messages":messages,
            "model": self.model,
            #"n_predict": 128,  # Nombre de tokens à générer
            "temperature": 0.7,  # Contrôle la créativité (0.0 = déterministe, 1.0 = aléatoire)
        }
        response = requests.post(self.url, json=data)
        result = response.json()
        if "choices" not in result or not result["choices"]:
                    raise ValueError("La réponse ne contient pas de choix valide.")

        # Extraction du contenu
        content = result["choices"][0]["message"]["content"]
        reasoning = result["choices"][0]["message"].get("reasoning_content", "")

        #print("--- REASONING ---")
        #print(reasoning)
        #print("\n--- RESPONSE ----")
        #print(content)
        return reasoning, content

class Chunker:
    def __init__(self, filename, nb_lines=100, overlap=0):
        self.filename = filename
        self.chunk_lines = nb_lines
        self.overlap = overlap
        self.current_chunk = 0
        
        # Lire toutes les lignes du fichier
        with open(self.filename, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        self.total_lines = len(self.lines)
        print(f"Chunker size : {self.total_lines}")

    def __iter__(self):
        self.current_chunk = 0  # Réinitialiser l'itérateur
        return self

    def __next__(self):
        if self.current_chunk >= self.total_lines:
            raise StopIteration
        
        start = self.current_chunk
        end = min(self.current_chunk + self.chunk_lines, self.total_lines)
        chunk = self.lines[start:end]
        
        # Passer à la prochaine position en tenant compte de l'overlap
        self.current_chunk += self.chunk_lines - self.overlap
        return ''.join(chunk)

import json

def save_valid_json_lines(text, output_file):
    """
    Vérifie ligne par ligne si c'est du JSON valide et l'ajoute à output_file.
    
    text : str : le texte renvoyé par le LLM (plusieurs lignes JSON)
    output_file : str : chemin du fichier où sauvegarder
    """
    lines = text.splitlines()
    with open(output_file, "a", encoding="utf-8") as f:  # 'a' pour append
        for line in lines:
            line = line.strip()
            if not line:
                continue  # ignorer les lignes vides
            try:
                # Vérifie que la ligne est un JSON valide
                data = json.loads(line)
                # Réécrire la ligne dans le fichier
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                print("Ligne invalide JSON ignorée :", line)


if __name__ == "__main__":
    print(datetime.now())
    chunker = Chunker("data/gutenberg/pg17992.txt",nb_lines=100,overlap=10)

    output_dir = "data/citations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = "dantes"

    url = "http://0.0.0.0:12001/v1/chat/completions"
    model = "gpt-oss-20b-default"
    gptoss = LLM(url, model, "Tu es un assistant littéraire, spécialisé dans le comte de Monte-cristo. Tu t’exprimes en français.")

    prompt = """Tu dois analyser des extraits du livre le comte de Monte Cristo.
    Ton but est de trouver les parties du texte qui correspondent aux pensées, paroles et écrits d’Edomnd Dantès (le comte de monte-cristo).
    À chaque fois que tu trouves un élément qui correspond soit à une de ses pensées, soit à une phrase qu’il prononce génère une entrée au format json.
    La syntaxe de json est :
    - "contexte" : un résumé du paragraphe d’où est extrait la citation
    - "citation" : la phrase, la ligne ou le paragraphe tel qu’énoncé par Edmond Dantès.
    Tu dois donc générer une réponse au format jsonl (liste d’entrée json).

    Exemple de réponse
    {"contexte": "arrivée du bateau à Marseille, dialogue entre Morel et Dantès", "citation":"--Oh! monsieur Morrel, s'écria le jeune marin, saisissant, les larmes aux yeux, les mains de l'armateur; monsieur Morrel, je vous remercie, au nom de mon père et de Mercédès."}
    {"contexte": "arrivée du bateau à Marseille, dialogue entre Morel et Dantès", "citation":"--Mais vous ne voulez pas que je vous ramène à terre?"} 
    
    Ne donne pas de contexte, répond juste avec les documents jsonl, car ces éléments seront ensuite analysés par un programme.
    Extrait:
    """
    max_chunks = 70
    processed_chunks = 0
    start_chunk = 0
    for chunk in chunker:
        processed_chunks +=1
        if processed_chunks <= start_chunk:
            continue
        if processed_chunks >max_chunks:
            break
        

        print(f"--- CHUNK {processed_chunks}---")
        #print(chunk)
        reasoning, content = gptoss.get_response(prompt + chunk)
        save_valid_json_lines(content,f"{output_dir}/{output_file}")
        print(datetime.now())
    print("END of processing")
