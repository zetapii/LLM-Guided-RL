import ollama
from helper import convert_minigrid_to_text_S3R1
import re

# Initialize the Ollama client
client = ollama.Client()

# Define the model
model = 'llama3.2'

def suggest_llm_action(env, env_name, action_dim, state):
    # Convert the state to an LLM-readable format
    llm_readable_format = convert_minigrid_to_text_S3R1(env, state)

    # Prepare the prompt
    prompt = f"""
    You are given an RL environment '{env_name}'.
    The current state is: 
    '{llm_readable_format}'
    
    You are an RL expert. Using your knowledge of the environment, suggest a possible action.
    Respond with a single integer in the range 0 to {action_dim - 1}, do not tell any reasoning.
    
    Description of actions:
    - 0: turn left
    - 1: turn right
    - 2: move forward
    - 3: pickup
    - 4: drop
    - 5: toggle (to open/close doors)
    - 6: done/noop
    """

    # Call the model and get the response 
    response = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    response_text = response['message']['content']

    
    # Attempt to extract the first integer from the response
    match = re.search(r'\b([0-6])\b', response_text)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"LLM response did not contain a valid action: {response_text}")
