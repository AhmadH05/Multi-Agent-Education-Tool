import re
from utils.ollama_client import query_ollama

def extract_score(label: str, response: str) -> float:
    pattern = rf"{label}:\s*(\d+(\.\d+)?)/10"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract '{label}' score. Pattern used: {pattern}")

def score_prompt_with_feedback(prompt: str) -> dict:
    system_message = """
You are a Critic Agent evaluating educational prompts. Use this format:

Coherence: x/10  
Creativity: y/10  
Educational: z/10  

Then write a section titled "Suggestions:" with actionable feedback.

Be strict. Use the full 1–10 scale. Format must match exactly.
"""

    user_prompt = f"Prompt:\n\"\"\"{prompt}\"\"\""
    response = query_ollama("llama3", system_message + "\n" + user_prompt)

    print("\n[CRITIC RAW RESPONSE]:\n", response)

    try:
        coherence = extract_score("Coherence", response)
        creativity = extract_score("Creativity", response)
        educational = extract_score("Educational", response)
    except Exception as e:
        raise ValueError(f"Failed to extract scores from response:\n{response}\n\nError: {e}")

    match = re.search(r"Suggestions(?: for improvement)?:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
    suggestions = match.group(1).strip() if match else "(No suggestions found.)"

    return {
        "coherence": coherence,
        "creativity": creativity,
        "educational": educational,
        "suggestions": suggestions
    }
