import requests

# URL de l'API locale
url = "http://0.0.0.0:12001/v1/chat/completions"
model = "gpt-oss-20b-default"
model = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
sp = "you are an helpful assistant"

sp = "tu es un assistant qui parle français"
up = "décris la personnalité d’edmond dantès"

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