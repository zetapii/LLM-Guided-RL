import os
from typing import List, Dict, Any

# Assuming gemini_api.py is in the same directory
from .gemini_api import call_gemini_api 

# Path to the few-shot examples for reflection
FEW_SHOT_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "prompts", "reflexion_few_shot_examples.txt")

try:
    with open(FEW_SHOT_EXAMPLES_PATH, 'r') as f:
        FEW_SHOT_EXAMPLES = f.read()
except FileNotFoundError:
    print(f"Warning: Reflection few-shot examples not found at {FEW_SHOT_EXAMPLES_PATH}")
    FEW_SHOT_EXAMPLES = "You need to provide few-shot examples for reflection here." # Placeholder

def _get_failed_episode_scenario(env_history_string: str) -> str:
    """
    Parses the scenario (initial observation and task) from the full environment history string.
    Assumes the history string starts with the few-shot examples followed by the initial obs.
    """
    # Find the start of the actual episode history after the few-shot examples
    # This might need adjustment based on how env_history.__str__ is formatted
    parts = env_history_string.split("Here are two examples.\n") # Split based on common text in prompts
    if len(parts) > 1:
        # Find the start of the task description within the initial observation part
        task_part = parts[-1] # Get the last part which should contain the current task run
        task_marker = "Your task is to:"
        task_start_index = task_part.find(task_marker)
        if task_start_index != -1:
            # Return the initial observation and task description
            # We might want to include the initial "You are in the middle..." part too
            initial_obs_marker = "You are in the middle of a room."
            obs_start_index = task_part.find(initial_obs_marker)
            if obs_start_index != -1 and obs_start_index < task_start_index:
                 # Include initial observation up to the end of the history provided
                 # The history string itself contains the failed trajectory actions/obs
                 return task_part[obs_start_index:] 
            else:
                 # Return from "Your task is to:" onwards
                 return task_part[task_start_index:]
        else:
            # Fallback if "Your task is to:" isn't found as expected
            print("Warning: Could not parse task description accurately for reflection.")
            return task_part # Return the relevant part of the history string
    else:
        # Fallback if the prompt structure is unexpected
        print("Warning: Could not parse scenario accurately for reflection.")
        return env_history_string # Return the whole history as fallback

def generate_reflection_prompt(failed_episode_history_str: str, past_reflections: List[str]) -> str:
    """
    Generates the prompt to ask the LLM for reflection on a failed episode.
    """
    scenario_and_history = _get_failed_episode_scenario(failed_episode_history_str)
    
    # Modified prompt structure slightly for clarity and Gemini
    # Asking for reflection first, then plan.
    prompt = f"""You previously failed the following task. Analyze the history, identify the mistake, and devise a concise, improved plan.

{FEW_SHOT_EXAMPLES}

Failed Task History:
{scenario_and_history}
"""

    if past_reflections:
        prompt += '\n\nReflections from previous failed attempts on this task:\n'
        for i, reflection in enumerate(past_reflections):
            # Limit length of past reflections if needed
            prompt += f'Attempt {i+1} Reflection: {reflection[:500]}...\n' 

    prompt += """
Analyze your mistake in the 'Failed Task History' above.
Then, provide a new, concise plan to complete the task successfully, avoiding the previous mistake.

Respond in the following format:
Reflection: <Your brief analysis of the mistake>
Plan: <Your new concise plan>"""
    
    return prompt

def get_reflection(failed_episode_history_str: str, past_reflections: List[str]) -> str:
    """
    Generates a reflection on a failed episode using the LLM.
    """
    prompt = generate_reflection_prompt(failed_episode_history_str, past_reflections)
    
    # Use gemini_api to get the completion
    # Temperature might be lower for reflection generation (e.g., 0.2)
    best_response, _ = call_gemini_api(
        prompt_text=prompt,
        candidate_count=1,
        temperature=0.2, 
        max_output_tokens=300 # Adjust as needed
    )
    
    reflection_text = best_response[0]
    
    if reflection_text == 'skip':
        print("Warning: Reflection generation failed or skipped by API.")
        return "Reflection generation failed."
        
    # Basic parsing (can be improved)
    parsed_reflection = reflection_text.strip()
    # Optional: Extract just the plan if needed, or return the full text
    # Example: Find "Plan:" and return text after it.
    
    return parsed_reflection # Return the full generated text for now

if __name__ == '__main__':
    # Example Usage (conceptual - requires a sample history string)
    print("Reflection Utils structure defined.")
    # sample_history = """
    # Interact with a household to solve a task. Here are two examples.
    # <Example 1 text...>
    # <Example 2 text...>
    # You are in the middle of a room...
    # Your task is to: put some spraybottle on toilet.
    # > go to cabinet 1
    # On the cabinet 1, you see a cloth 1...
    # > go to sinkbasin 1
    # On the sinkbasin 1, you see nothing.
    # (sequence of failed actions)
    # """
    # sample_memory = ["Plan: First check cabinets, then countertops for the spraybottle."]
    
    # reflection_prompt = generate_reflection_prompt(sample_history, sample_memory)
    # print("\n--- Generated Reflection Prompt ---")
    # print(reflection_prompt)
    
    # print("\n--- Attempting Reflection Generation (requires API key) ---")
    # try:
    #     from gemini_api import genai as gemini_genai_module
    #     if gemini_genai_module.API_KEY:
    #          reflection = get_reflection(sample_history, sample_memory)
    #          print(f"Generated Reflection:\n{reflection}")
    #     else:
    #          print("Skipping reflection generation, API key not configured.")
    # except (ImportError, AttributeError):
    #      print("Skipping reflection generation, gemini_api not available or key not configured.")
