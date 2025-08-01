from utils.ollama_client import query_ollama

def generate_prompt(raw_prompt):
    system_prompt = (
        "You are a Prompt Generator Agent (PGA). Enrich raw prompts to make them clearer, more engaging, "
        "and more educational. DO NOT write a story or fictional content. Avoid exposition. "
        "Make the prompt suitable for teaching or exploring real-world concepts, preferably in STEM or engineering."
    )

    user_prompt = f'Raw Prompt:\n"""{raw_prompt}"""'

    from utils.ollama_utils import ollama_chat
    return ollama_chat("llama3", system_prompt, user_prompt).strip()
