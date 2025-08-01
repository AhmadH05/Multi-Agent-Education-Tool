from utils.ollama_utils import ollama_chat
import re

def evaluate_output(generated_text):
    system_prompt = (
        "You are an Output Evaluator Agent (OEA) scoring student-facing educational responses. Use the following:\n\n"
        "- Coherence – Is it logically structured and easy to follow?\n"
        "- Creativity – Does it present ideas in an engaging, novel way?\n"
        "- Educational – Does it teach or explain something effectively?\n\n"
        "SCORING RULES:\n"
        "- Use the full 1–10 scale.\n"
        "- 5 is average, 7 is strong, 8+ is rare.\n"
        "- No fluff. No generosity.\n\n"
        "FORMAT YOUR REPLY STRICTLY:\n"
        "Coherence: X/10\n"
        "Creativity: X/10\n"
        "Educational: X/10\n"
        "Comment: <1–2 sentence critique>\n"
        "Suggestions: <Optional tips to improve>"
    )

    user_prompt = f'Generated Output:\n"""{generated_text}"""'
    response = ollama_chat("llama3", system_prompt, user_prompt).strip()

    try:
        coherence = float(re.search(r"(?i)Coherence:\s*(\d+(\.\d+)?)/10", response).group(1))
        creativity = float(re.search(r"(?i)Creativity:\s*(\d+(\.\d+)?)/10", response).group(1))
        educational = float(re.search(r"(?i)Educational:\s*(\d+(\.\d+)?)/10", response).group(1))
        comment = re.search(r"(?i)Comment:\s*(.+?)(?:\n|Suggestions:|$)", response, re.DOTALL).group(1).strip()
        suggestions_match = re.search(r"(?i)Suggestions:\s*(.+)", response, re.DOTALL)
        suggestions = suggestions_match.group(1).strip() if suggestions_match else ""
    except Exception as e:
        print(f"[Evaluation Parse Error] Raw response:\n{response}\n\nError: {e}")
        return {
            "coherence": 0.0,
            "creativity": 0.0,
            "educational": 0.0,
            "comment": "Could not extract scores. Make sure the model follows strict format.",
            "suggestions": ""
        }

    return {
        "coherence": coherence,
        "creativity": creativity,
        "educational": educational,
        "comment": comment,
        "suggestions": suggestions
    }
