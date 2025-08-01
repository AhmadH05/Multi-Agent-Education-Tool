import requests

def ollama_chat(model: str, system_message: str, user_prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
    )
    return response.json()["message"]["content"]
