from llm import LLM
import json

def generate_character_card():
    url = "http://0.0.0.0:12001/v1/chat/completions"
    model = "gpt-oss-20b-default"
    llm = LLM(url, model, "Tu es un assistant littéraire, spécialisé dans le comte de Monte-cristo. Tu t’exprimes en français.")
    with open("data/citations/dantes.jsonl","r",encoding="utf-8") as f:
        lines = f.readlines()

    prompt = """Analyse ces citations issues du livre le comte de monte-cristo. 
    Chaque extrait contient une citatiion d’Edmond Dantès. 
    Utilise uniquement ces citations pour faire ton analyse.
    À partir de ces citations, fait une synthèse des caractéristiques du personnage : 
    - psychologie : tempérament, émotions dominantes, motivations, rapport aux autres
    - style verbal : registre , syntaxe, figures de styles, lexique, rythme
    - comportement conversationnel face à une provocation, une question morale, une attaque, une critique etc..
    Le but est de produire une fiche personnage, permettant d’analyser et décrire le personnage. 
    """
    block_size = 150
    for i in range(0,len(lines),block_size):
        print(f"Analyse {i} /{len(lines)/block_size}")
        citations = "".join(lines[i:i+block_size])
        character = llm.get_response(prompt + citations)
        with open("card-result.txt","a",encoding="utf-8") as f:
            f.write(f"--- ANALYSE {i} ---\n")
            f.write(character+"\n")

def analyse_character_card():
    with open("card-result.txt","r",encoding="utf-8") as f:
        content = f.read()
    url = "http://0.0.0.0:12001/v1/chat/completions"
    model = "gpt-oss-20b-default"
    llm = LLM(url, model, "Tu es un assistant littéraire, spécialisé dans le comte de Monte-cristo. Tu t’exprimes en français.")
    prompt = """À partir des analyses suivantes, fais une synthèse des caractéristiques du personnage d’Edmond Dantès.
    Présente les résultats sous forme de fiche personnage, avec des sections claires pour chaque aspect analysé.
    """
    character_card = llm.get_response(prompt + content)
    with open("character-card-summary.txt","w",encoding="utf-8") as f:
        f.write(character_card)

if __name__ == "__main__":
    #generate_character_card()
    analyse_character_card()