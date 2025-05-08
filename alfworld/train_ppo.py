import torch
import alfworld
import alfworld.agents.environment
import yaml
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from ppo_agent import PPOAgent # Assuming ppo_agent.py is in the same directory
from env_history import EnvironmentHistory # Assuming env_history.py is in the same directory
from reflection_utils import get_reflection # Import reflection utility

# --- Configuration ---
CONFIG_FILE = '../Language-Integrated-VI/alfworld/base_config.yaml'
PROMPT_FILE_PATH = '../Language-Integrated-VI/alfworld/prompts/alfworld_3prompts.json'

# PPO Hyperparameters
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_CANDIDATE_COUNT = 5
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
EPOCHS_PER_UPDATE = 10
MINI_BATCH_SIZE = 64 # Should be <= STEPS_PER_COLLECTION_CYCLE
LR = 3e-4
VF_COEF = 0.5
ENTROPY_COEF = 0.01
SEED = 42

# Training parameters
NUM_EPISODES = 500 # Total number of episodes for training
STEPS_PER_COLLECTION_CYCLE = 2048 # Number of steps to collect before an update
MAX_STEPS_PER_EPISODE = 150 # Max steps within a single episode
MAX_MEMORY_PER_TASK = 3 # Limit number of reflections stored/used

# --- Helper Functions (copied from train_dqn.py, could be moved to a utils.py) ---
def process_ob(ob: str) -> str:
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

def load_llm_prompt_examples(prompt_file_path: str, task_prefix: str) -> str:
    import json
    with open(prompt_file_path, 'r') as f:
        prompt_data = json.load(f)
    example_key_1 = f'react_{task_prefix}_1'
    example_key_0 = f'react_{task_prefix}_0'
    if example_key_1 in prompt_data and example_key_0 in prompt_data:
         return 'Interact with a household to solve a task. Here are two examples.\n' + \
               prompt_data[example_key_1] + \
               prompt_data[example_key_0]
    else:
        print(f"Warning: Prompt examples for '{task_prefix}' not found. Using generic 'put' examples.")
        return 'Interact with a household to solve a task. Here are two examples.\n' + \
               prompt_data['react_put_1'] + \
               prompt_data['react_put_0']

def get_task_prefix_from_gamefile(gamefile_path: str) -> str:
    PREFIXES_MAP = {
        'pick_and_place': 'put', 'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat', 'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine', 'pick_two_obj': 'puttwo'
    }
    task_type_key = gamefile_path.split('/')[-3]
    for k, v in PREFIXES_MAP.items():
        if task_type_key.startswith(k):
            return v
    print(f"Warning: Could not determine task prefix for gamefile {gamefile_path}. Defaulting to 'put'.")
    return 'put'

# --- Main Training Function ---
def train_ppo():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with open(CONFIG_FILE, 'r') as reader:
        alfworld_config = yaml.safe_load(reader)
    
    split = "eval_out_of_distribution" 
    env = getattr(alfworld.agents.environment, alfworld_config["env"]["type"])(alfworld_config, train_eval=split)
    env = env.init_env(batch_size=1)

    agent = PPOAgent(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_candidate_count=LLM_CANDIDATE_COUNT,
        gamma=GAMMA, gae_lambda=GAE_LAMBDA, ppo_clip_epsilon=PPO_CLIP_EPSILON,
        epochs_per_update=EPOCHS_PER_UPDATE, mini_batch_size=MINI_BATCH_SIZE,
        lr=LR, vf_coef=VF_COEF, entropy_coef=ENTROPY_COEF, seed=SEED
    )

    print(f"Training PPO Agent on device: {agent.device}")
    print(f"State size: {agent.state_size}, Action_size (LLM candidates): {agent.action_size}")

    total_timesteps_collected = 0
    task_memory: Dict[str, List[str]] = defaultdict(list) # Store reflections per task
    
    # Initialize current observation and history for the very first step
    obs_list, infos = env.reset()
    current_raw_obs = process_ob(obs_list[0])
    current_game_file = infos['extra.gamefile'][0] # Track current game file
    task_prefix = get_task_prefix_from_gamefile(current_game_file)
    llm_few_shot_examples = load_llm_prompt_examples(PROMPT_FILE_PATH, task_prefix)
    
    # Retrieve past reflections for this task
    past_reflections = task_memory[current_game_file][-MAX_MEMORY_PER_TASK:]
    
    # Construct initial prompt context with reflections
    initial_prompt_context = llm_few_shot_examples
    if past_reflections:
         initial_prompt_context += "\n\n--- Past Reflections for this Task ---"
         for i, ref in enumerate(past_reflections):
             initial_prompt_context += f"\nAttempt {i+1} Reflection:\n{ref}"
         initial_prompt_context += "\n-------------------------------------\n"

    env_history_for_llm = EnvironmentHistory(
        base_prompt=initial_prompt_context, initial_obs=current_raw_obs, memory=[], env_history=[]
    )
    env_history_for_llm.reset()
    
    current_episode_reward = 0 # Track reward for the current episode segment
    current_episode_steps = 0 # Track steps for the current episode segment

    # Main loop runs indefinitely until NUM_EPISODES worth of experience is processed (approx)
    # Or based on total timesteps collected. Let's use total timesteps.
    # We need a way to track episode count properly across collection cycles.
    
    episode_count = 0 # Track completed episodes

    while total_timesteps_collected < NUM_EPISODES * MAX_STEPS_PER_EPISODE: # Approximate total steps

        # PPO collects data for STEPS_PER_COLLECTION_CYCLE before updating
        # The loop for collecting one batch of trajectories
        collection_finished = False
        while not collection_finished:
            
            llm_prompt = str(env_history_for_llm) + "\n>"
            
            action_text, action_index, log_prob, state_value, current_state_embedding = \
                agent.select_action(llm_prompt, current_raw_obs)

            next_obs_list, reward_list, done_list, infos = env.step([action_text])
            
            next_raw_obs = process_ob(next_obs_list[0])
            reward = reward_list[0]
            done = done_list[0]

            next_state_embedding = agent._preprocess_state_text(next_raw_obs) # For storage

            agent.store_transition(
                current_state_embedding, action_text, action_index, log_prob,
                reward, state_value, done, next_state_embedding 
            )
            
            current_episode_reward += reward
            total_timesteps_collected += 1
            current_episode_steps += 1

            if action_text.startswith("think:"):
                env_history_for_llm.add("action", action_text)
                env_history_for_llm.add("observation", "OK.")
            else:
                env_history_for_llm.add("action", action_text)
                env_history_for_llm.add("observation", next_raw_obs)
            
            current_raw_obs = next_raw_obs # Update current_raw_obs for the next iteration

            # Check if collection cycle is full OR episode ended
            collection_cycle_full = len(agent.trajectory_data["states"]) >= STEPS_PER_COLLECTION_CYCLE
            episode_ended = done or current_episode_steps >= MAX_STEPS_PER_EPISODE

            if episode_ended:
                episode_count += 1
                success = infos.get('won', [False])[0] and done
                print(f"Episode {episode_count} finished. Steps: {current_episode_steps}, Reward: {current_episode_reward}, Success: {success}")

                if not success:
                    # Generate and store reflection
                    print(f"Generating reflection for failed task: {current_game_file}")
                    failed_history_str = str(env_history_for_llm)
                    # Use the reflections that were active during this failed attempt
                    active_reflections = task_memory[current_game_file][-MAX_MEMORY_PER_TASK:]
                    reflection = get_reflection(failed_history_str, active_reflections)
                    print(f"Generated Reflection:\n{reflection}")
                    task_memory[current_game_file].append(reflection)
                    if len(task_memory[current_game_file]) > MAX_MEMORY_PER_TASK * 2:
                         task_memory[current_game_file] = task_memory[current_game_file][-MAX_MEMORY_PER_TASK*2:]

                # Reset environment and history for the next episode
                obs_list, infos = env.reset()
                current_raw_obs = process_ob(obs_list[0])
                current_game_file = infos['extra.gamefile'][0] # Update game file
                task_prefix = get_task_prefix_from_gamefile(current_game_file)
                llm_few_shot_examples = load_llm_prompt_examples(PROMPT_FILE_PATH, task_prefix)
                past_reflections = task_memory[current_game_file][-MAX_MEMORY_PER_TASK:] # Get reflections for new task
                initial_prompt_context = llm_few_shot_examples
                if past_reflections:
                     initial_prompt_context += "\n\n--- Past Reflections for this Task ---"
                     for i, ref in enumerate(past_reflections):
                         initial_prompt_context += f"\nAttempt {i+1} Reflection:\n{ref}"
                     initial_prompt_context += "\n-------------------------------------\n"
                env_history_for_llm = EnvironmentHistory(
                    base_prompt=initial_prompt_context, initial_obs=current_raw_obs, memory=[], env_history=[]
                )
                env_history_for_llm.reset()
                current_episode_reward = 0
                current_episode_steps = 0

            # End the collection loop if the cycle is full
            if collection_cycle_full:
                collection_finished = True

        # --- After Collection Cycle ---
        print(f"Collected {len(agent.trajectory_data['states'])} transitions. Updating PPO agent...")
        # Need the observation *after* the last stored transition for GAE calculation
        agent.update(current_raw_obs) 

        if episode_count > 0 and episode_count % 10 == 0: # Log overall progress periodically
             # Note: This logging might not align perfectly with episode boundaries due to collection cycles
             print(f"--- Overall Progress: Approx Episode {episode_count}, Total Timesteps: {total_timesteps_collected} ---")


    env.close()
    print("PPO Training complete.")

if __name__ == '__main__':
    try:
        from gemini_api import genai as gemini_genai_module
        if not gemini_genai_module.API_KEY:
            print("GEMINI_API_KEY is not configured. Please set the environment variable.")
            exit(1)
    except ImportError:
        print("Could not import gemini_api.")
        exit(1)
    except AttributeError:
         print("Gemini API key not found in the genai object from gemini_api.py.")
         exit(1)

    print("Starting PPO training with LLM guidance...")
    train_ppo()
