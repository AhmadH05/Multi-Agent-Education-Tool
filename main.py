import json
import os
import numpy as np
from agents.generator import generate_prompt
from agents.critic import score_prompt_with_feedback
from agents.refiner import refine_prompt
from agents.output_evaluator import evaluate_output
from utils.ollama_client import query_ollama

# Basic settings
rounds = int(input("How many rounds? "))
trial_enabled = input("Enable trial generation? (y/n): ").lower().strip() == 'y'
human_enabled = input("Enable human scoring? (y/n): ").lower().strip() == 'y'
backprop_enabled = input("Enable feedback to PGA (backpropagation)? (y/n): ").lower().strip() == 'y'

# Reset results and file
all_results = []
if os.path.exists("results/scoring_log.json"):
    os.remove("results/scoring_log.json")

def trial_prompt_execution(prompt):
    instruction = f"""
You are a creative AI. Use the following prompt to generate a short creative and educational response.

Prompt: "{prompt}"

Respond only with the generated content.
""".strip()

    result = query_ollama("llama3", instruction)
    return result.strip() if isinstance(result, str) else "No result generated."

def extract_suggestions(eval_obj):
    if isinstance(eval_obj, dict):
        return eval_obj.get("suggestions", "")
    return ""

def clean_title(text):
    return text.replace("Enriched Prompt:", "").replace("Revised Prompt:", "").strip()

def run_loop(prompts):
    current_prompts = prompts.copy()
    base_outputs = {}
    base_evals = {}

    for round_num in range(rounds):
        print(f"\n========= ROUND {round_num + 1} =========")
        new_prompts = []

        for i, raw in enumerate(current_prompts):
            print("\n==============================")
            print(f"Raw Prompt:\n{raw}")

            if round_num == 0:
                base_output = base_output_eval = ""
                human_score = human_feedback = None
                if trial_enabled:
                    base_output = trial_prompt_execution(raw)
                    base_output_eval = evaluate_output(base_output)
                    print(f"\n--- Raw Output ---\n{base_output}")
                    print(f"\n--- Raw Output Eval ---\n{base_output_eval}")

                base_outputs[raw] = base_output
                base_evals[raw] = base_output_eval

                result_entry = {
                    "raw": raw,
                    "enriched": "",
                    "refined": "",
                    "critic_score_enriched": {},
                    "critic_score_refined": {},
                    "human_score": None,
                    "human_feedback": None,
                    "generated_output_enriched": "",
                    "output_eval_enriched": {},
                    "generated_output_refined": "",
                    "output_eval_refined": {},
                    "base_output": base_output,
                    "base_output_eval": base_output_eval
                }
                all_results.append(result_entry)
                new_prompts.append(raw)
                continue

            enriched = clean_title(generate_prompt(raw).strip())
            print(f"\n**Enriched Prompt:**\n{enriched}")

            try:
                score1 = score_prompt_with_feedback(enriched)
            except ValueError as e:
                print(f"\nError during critic evaluation (enriched): {e}")
                score1 = {}
            print(f"\n**Critic Feedback (Enriched):**\n{score1}")

            enriched_output = enriched_output_eval = ""
            refined = refined_output = refined_output_eval = ""
            score2 = ""
            human_score = human_feedback = None

            if trial_enabled:
                enriched_output = trial_prompt_execution(enriched)
                enriched_output_eval = evaluate_output(enriched_output)
                print(f"\n--- Enriched Output ---\n{enriched_output}")
                print(f"\n--- Enriched Output Eval ---\n{enriched_output_eval}")

                if human_enabled:
                    human_score = input("Your rating of this output (1–10): ").strip()
                    human_feedback = input("Your reasoning (optional): ").strip()

                enriched_suggestions = extract_suggestions(enriched_output_eval)

                temp_refined = clean_title(refine_prompt(
                    enriched,
                    score1,
                    enriched_output_eval,
                    enriched_suggestions,
                    ""
                ).strip())
                print(f"\n**Refined Prompt:**\n{temp_refined}")

                temp_refined_output = trial_prompt_execution(temp_refined)
                temp_refined_output_eval = evaluate_output(temp_refined_output)
                print(f"\n--- Refined Output ---\n{temp_refined_output}")
                print(f"\n--- Refined Output Eval ---\n{temp_refined_output_eval}")

                enriched_avg = np.mean([
                    enriched_output_eval.get("coherence", 0),
                    enriched_output_eval.get("creativity", 0),
                    enriched_output_eval.get("educational", 0)
                ])
                refined_avg = np.mean([
                    temp_refined_output_eval.get("coherence", 0),
                    temp_refined_output_eval.get("creativity", 0),
                    temp_refined_output_eval.get("educational", 0)
                ])

                if refined_avg >= enriched_avg:
                    refined, refined_output, refined_output_eval = temp_refined, temp_refined_output, temp_refined_output_eval
                    try:
                        score2 = score_prompt_with_feedback(refined)
                    except ValueError as e:
                        print(f"\nError during critic evaluation (refined): {e}")
                        score2 = {}
                    print(f"\n**Critic Feedback (Refined):**\n{score2}")
                else:
                    print("\nRefined output not better. Keeping enriched version.")
                    refined, refined_output, refined_output_eval = enriched, enriched_output, enriched_output_eval
                    score2 = score1

            result_entry = {
                "raw": raw,
                "enriched": enriched,
                "refined": refined,
                "critic_score_enriched": score1,
                "critic_score_refined": score2,
                "human_score": human_score,
                "human_feedback": human_feedback,
                "generated_output_enriched": enriched_output,
                "output_eval_enriched": enriched_output_eval,
                "generated_output_refined": refined_output,
                "output_eval_refined": refined_output_eval,
                "base_output": base_outputs.get(raw, ""),
                "base_output_eval": base_evals.get(raw, {})
            }

            all_results.append(result_entry)
            new_prompts.append(refined)

        current_prompts = new_prompts

if __name__ == "__main__":
    with open('prompts/test_prompts.json') as f:
        test_prompts = json.load(f)

    run_loop(test_prompts)

    with open("results/scoring_log.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n Saved all results to results/scoring_log.json")
