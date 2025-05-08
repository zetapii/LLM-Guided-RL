import torch
import alfworld
import alfworld.agents.environment
import yaml
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from dqn_agent import DQNAgent # Assuming dqn_agent.py is in the same directory
from env_history import EnvironmentHistory # Assuming env_history.py is in the same directory
from reflection_utils import get_reflection # Import reflection utility

# --- Configuration ---
CONFIG_FILE = '../Language-Integrated-VI/alfworld/base_config.yaml' # Path to AlfWorld base config
PROMPT_FILE_PATH = '../Language-Integrated-VI/alfworld/prompts/alfworld_3prompts.json' # For LLM examples

# DQN Hyperparameters (can be moved to a config file or argparse)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # For SentenceTransformer
LLM_CANDIDATE_COUNT = 5 # How many actions the LLM should suggest / Q-network action_size
BUFFER_SIZE = int(1e4) # Replay buffer size (smaller for faster iteration initially)
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4 # How often to update the Q-network
SEED = 42

# Training parameters
NUM_EPISODES = 100 # Number of training episodes
MAX_STEPS_PER_EPISODE = 100 # Max steps per episode
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# --- Helper Functions ---
def process_ob(ob: str) -> str:
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

def load_llm_prompt_examples(prompt_file_path: str, task_prefix: str) -> str:
    """Loads few-shot examples for the LLM based on task type."""
    import json
    with open(prompt_file_path, 'r') as f:
        prompt_data = json.load(f)
    # Using 'react_put_0' and 'react_put_1' as generic examples for now.
    # This should be adapted based on the actual task from AlfWorld env.
    # The original code dynamically selected prompts based on gamefile name.
    # Example: base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
    # For now, we'll use a fixed general prompt.
    # PREFIXES from original code:
    # 'pick_and_place': 'put', 'pick_clean_then_place': 'clean', etc.
    # We'll need to map gamefile to these prefixes.
    # For simplicity, using 'put' examples.
    example_key_1 = f'react_{task_prefix}_1'
    example_key_0 = f'react_{task_prefix}_0'
    
    if example_key_1 in prompt_data and example_key_0 in prompt_data:
         return 'Interact with a household to solve a task. Here are two examples.\n' + \
               prompt_data[example_key_1] + \
               prompt_data[example_key_0]
    else: # Fallback to generic 'put' if specific task prefix not found
        print(f"Warning: Prompt examples for '{task_prefix}' not found. Using generic 'put' examples.")
        return 'Interact with a household to solve a task. Here are two examples.\n' + \
               prompt_data['react_put_1'] + \
               prompt_data['react_put_0']


def get_task_prefix_from_gamefile(gamefile_path: str) -> str:
    """Extracts task prefix (e.g., 'put', 'clean') from gamefile path."""
    # Based on PREFIXES in original alfworld_trial_wv.py
    PREFIXES_MAP = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }
    task_type_key = gamefile_path.split('/')[-3] # e.g. pick_and_place_simple
    for k, v in PREFIXES_MAP.items():
        if task_type_key.startswith(k):
            return v
    print(f"Warning: Could not determine task prefix for gamefile {gamefile_path}. Defaulting to 'put'.")
    return 'put' # Default if no match

# --- Main Training Function ---
def train_dqn():
    # Setup seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load AlfWorld base config
    with open(CONFIG_FILE, 'r') as reader:
        alfworld_config = yaml.safe_load(reader)
    
    # Initialize AlfWorld environment
    # Using 'eval_out_of_distribution' split as in the original paper's setup for alfworld
    split = "eval_out_of_distribution" 
    env = getattr(alfworld.agents.environment, alfworld_config["env"]["type"])(alfworld_config, train_eval=split)
    env = env.init_env(batch_size=1) # Batch size 1 for single agent interaction

    # Initialize DQN Agent
    agent = DQNAgent(embedding_model_name=EMBEDDING_MODEL_NAME,
                     llm_candidate_count=LLM_CANDIDATE_COUNT,
                     buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                     gamma=GAMMA, tau=TAU, lr=LR, update_every=UPDATE_EVERY, seed=SEED)

    print(f"Training DQN Agent on device: {agent.device}")
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME} (State size: {agent.state_size})")
    print(f"LLM candidates / Q-network action_size: {agent.action_size}")

    epsilon = EPSILON_START
    total_steps = 0
    episode_rewards = []
    task_memory: Dict[str, List[str]] = defaultdict(list) # Store reflections per task (game_file)
    MAX_MEMORY_PER_TASK = 3 # Limit number of reflections stored/used

    for i_episode in range(1, NUM_EPISODES + 1):
        # Reset environment and get initial observation
        obs_list, infos = env.reset()
        obs = process_ob(obs_list[0])
        game_file = infos['extra.gamefile'][0] # Use game_file as task identifier
        task_prefix = get_task_prefix_from_gamefile(game_file)
        llm_few_shot_examples = load_llm_prompt_examples(PROMPT_FILE_PATH, task_prefix)
        
        # Retrieve past reflections for this task
        past_reflections = task_memory[game_file][-MAX_MEMORY_PER_TASK:]
        
        # Initialize environment history for LLM prompts
        # Incorporate past reflections into the base prompt or context
        # Task description is part of the initial observation in AlfWorld (e.g., "Your task is to...")
        # We need to extract it or ensure it's part of `obs`.
        # The original `alfworld_run` prepends "Interact with a household..." and examples.
        # `env_history` will store the sequence of (action, observation)
        
        # Constructing the initial part of the prompt for the LLM
        # This includes the few-shot examples and the current game's initial observation + task
        initial_llm_prompt_context = llm_few_shot_examples + "\n" + obs
        
        # We need an object to manage the history of interactions for the LLM prompt
        # Modify base_prompt to include reflections if available
        # Note: EnvironmentHistory's 'memory' param was for a different purpose in original code.
        # We'll prepend reflections to the prompt context instead.
        
        initial_prompt_context = llm_few_shot_examples
        if past_reflections:
             initial_prompt_context += "\n\n--- Past Reflections for this Task ---"
             for i, ref in enumerate(past_reflections):
                 initial_prompt_context += f"\nAttempt {i+1} Reflection:\n{ref}"
             initial_prompt_context += "\n-------------------------------------\n"
        
        env_history_for_llm = EnvironmentHistory(
            base_prompt=initial_prompt_context, # Base prompt now includes reflections
            initial_obs=obs, 
            memory=[], # Keep original memory param empty for now
            env_history=[] 
        )
        env_history_for_llm.reset() 

        current_episode_reward = 0
        
        # Initial state embedding
        current_obs_text_for_embedding = obs # The raw observation text for the current state
        current_state_embedding = agent._preprocess_state_text(current_obs_text_for_embedding)

        for t in range(MAX_STEPS_PER_EPISODE):
            total_steps += 1

            # Construct prompt for LLM using current history
            llm_prompt = str(env_history_for_llm) + "\n>"

            # Agent selects action (text and index)
            action_text, action_index = agent.select_action(llm_prompt, current_obs_text_for_embedding, epsilon)

            # Environment steps
            next_obs_list, reward_list, done_list, infos = env.step([action_text])
            
            next_raw_obs = process_ob(next_obs_list[0]) # Raw text of next observation
            reward = reward_list[0]
            done = done_list[0]
            
            # Update environment history for LLM prompt generation
            if action_text.startswith("think:"):
                env_history_for_llm.add("action", action_text)
                env_history_for_llm.add("observation", "OK.")
            else:
                env_history_for_llm.add("action", action_text)
                env_history_for_llm.add("observation", next_raw_obs)

            # Get embedding for the next state
            next_obs_text_for_embedding = next_raw_obs
            next_state_embedding = agent._preprocess_state_text(next_obs_text_for_embedding)

            # Store experience using embeddings and action_index
            agent.add_experience(current_state_embedding, action_text, action_index, reward, next_state_embedding, done)
            
            current_state_embedding = next_state_embedding
            current_obs_text_for_embedding = next_raw_obs # Update for the next iteration's embedding
            current_episode_reward += reward
            
            # --- Episode End Handling ---
            episode_finished = done or (t + 1) == MAX_STEPS_PER_EPISODE
            if episode_finished:
                success = infos.get('won', [False])[0] and done # Ensure 'won' is True and episode is 'done'
                print(f"Episode {i_episode} finished after {t+1} steps. Reward: {current_episode_reward}. Success: {success}")
                
                if not success:
                    # Generate and store reflection on failure
                    print(f"Generating reflection for failed task: {game_file}")
                    # Pass the full history string (which includes the base prompt with past reflections)
                    failed_history_str = str(env_history_for_llm) 
                    # Pass only the most recent reflections used in the prompt to avoid redundancy in reflection prompt
                    reflection = get_reflection(failed_history_str, past_reflections) 
                    print(f"Generated Reflection:\n{reflection}")
                    task_memory[game_file].append(reflection) # Add new reflection
                    # Optional: Trim memory if it exceeds max size
                    if len(task_memory[game_file]) > MAX_MEMORY_PER_TASK * 2: # Keep a bit more than used
                         task_memory[game_file] = task_memory[game_file][-MAX_MEMORY_PER_TASK*2:]

                break # End step loop for this episode

        # --- After Episode ---
        episode_rewards.append(current_episode_reward)
        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon) # Decay epsilon

        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {i_episode}\tAverage Reward (last 10): {avg_reward:.2f}\tEpsilon: {epsilon:.3f}")
            # Potentially save model checkpoint here

    env.close()
    print("Training complete.")
    # TODO: Save final model, plot rewards, etc.


if __name__ == '__main__':
    # Note: This script requires AlfWorld and its dependencies to be installed.
    # It also requires the GEMINI_API_KEY environment variable to be set.
    # The paths to 'base_config.yaml' and 'alfworld_3prompts.json' need to be correct.
    
    # Check if Gemini API is configured in dqn_agent's imported gemini_api module
    try:
        from gemini_api import genai as gemini_genai_module
        if not gemini_genai_module.API_KEY:
            print("GEMINI_API_KEY is not configured in gemini_api.py. Please set the environment variable.")
            exit(1)
    except ImportError:
        print("Could not import gemini_api. Ensure it's in the same directory or PYTHONPATH.")
        exit(1)
    except AttributeError:
         print("Gemini API key not found in the genai object from gemini_api.py. Ensure GEMINI_API_KEY is set.")
         exit(1)


    print("Starting DQN training with LLM guidance...")
    train_dqn()
