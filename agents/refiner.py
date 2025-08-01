from utils.ollama_client import query_ollama

def refine_prompt(prompt, critic_eval, output_eval, enriched_suggestions=None, refined_suggestions=None):
    system_prompt = (
        "You are a Prompt Refiner tasked with improving prompts for maximum educational value. "
        "Avoid creative stories. Focus on clarity, teaching value, and brief but informative guidance. "
        "Incorporate critic and output feedback to improve the quality of resulting responses. "
        "Do not generate any narrative or fictional content. No storytelling. "
        "Refined prompts should inspire concise, technical, and helpful outputs for students. "
        "Provide 2 bullet points explaining how your refinement improves the prompt."
    )

    user_prompt = f"""Current Prompt:
{prompt}

Critic Evaluation:
- Coherence: {critic_eval.get('coherence', 'N/A')}
- Creativity: {critic_eval.get('creativity', 'N/A')}
- Educational: {critic_eval.get('educational', 'N/A')}
- Suggestions: {critic_eval.get('suggestions', '')}

Output Evaluation:
- Coherence: {output_eval.get('coherence', 'N/A')}
- Creativity: {output_eval.get('creativity', 'N/A')}
- Educational: {output_eval.get('educational', 'N/A')}
- Suggestions: {output_eval.get('suggestions', '')}
"""

    if enriched_suggestions:
        user_prompt += f"\nPrevious Enriched Output Suggestions:\n- {enriched_suggestions}"
    if refined_suggestions:
        user_prompt += f"\nPrevious Refined Output Suggestions:\n- {refined_suggestions}"

    user_prompt += "\n\nNow return a revised prompt, not a story."

    return query_ollama("llama3", system_prompt, user_prompt).strip()
