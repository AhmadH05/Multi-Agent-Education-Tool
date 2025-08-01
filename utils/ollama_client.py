import ollama

def query_ollama(model, system_prompt, user_prompt=None):
    if user_prompt is None:
        # single prompt mode
        messages = [{"role": "user", "content": system_prompt}]
    else:
        # chat-style system+user prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]
