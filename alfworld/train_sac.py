import torch
import alfworld
import alfworld.agents.environment
import yaml
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from sac_agent import SACAgent # Assuming sac_agent.py is in the same directory
from env_history import EnvironmentHistory # Assuming env_history.py is in the same directory
from reflection_utils import get_reflection # Import reflection utility

# --- Configuration ---
CONFIG_FILE = '../Language-Integrated-VI/alfworld/base_config.yaml'
PROMPT_FILE_PATH = '../Language-Integrated-VI/alfworld/prompts/alfworld_3prompts.json'

# SAC Hyperparameters
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_CANDIDATE_COUNT = 5
GAMMA = 0.99
TAU = 0.005 # Soft update factor
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
ALPHA_INIT = 0.2 # Initial temperature
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
UPDATE_EVERY = 1 # Update frequency
TARGET_ENTROPY_SCALE = 0.98
SEED = 42

# Training parameters
NUM_EPISODES = 500 # Total number of training episodes
MAX_STEPS_PER_EPISODE = 150 # Max steps within a single episode
START_STEPS = 1000 # Number of initial random steps before training starts
MAX_MEMORY_PER_TASK = 3 # Limit number of reflections stored/used

# --- Helper Functions (copied, consider utils.py) ---
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
def train_sac():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with open(CONFIG_FILE, 'r') as reader:
        alfworld_config = yaml.safe_load(reader)
    
    split = "eval_out_of_distribution" 
    env = getattr(alfworld.agents.environment, alfworld_config["env"]["type"])(alfworld_config, train_eval=split)
    env = env.init_env(batch_size=1)

    agent = SACAgent(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_candidate_count=LLM_CANDIDATE_COUNT,
        gamma=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, lr_alpha=LR_ALPHA,
        alpha_init=ALPHA_INIT, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        update_every=UPDATE_EVERY, target_entropy_scale=TARGET_ENTROPY_SCALE, seed=SEED
    )

    print(f"Training SAC Agent on device: {agent.device}")
    print(f"State size: {agent.state_size}, Action_size (LLM candidates): {agent.action_size}")

    total_timesteps = 0
    episode_rewards = []
    task_memory: Dict[str, List[str]] = defaultdict(list) # Store reflections per task

    for i_episode in range(1, NUM_EPISODES + 1):
        obs_list, infos = env.reset()
        current_raw_obs = process_ob(obs_list[0])
        game_file = infos['extra.gamefile'][0] # Use game_file as task identifier
        task_prefix = get_task_prefix_from_gamefile(game_file)
        llm_few_shot_examples = load_llm_prompt_examples(PROMPT_FILE_PATH, task_prefix)
        
        # Retrieve past reflections for this task
        past_reflections = task_memory[game_file][-MAX_MEMORY_PER_TASK:]
        
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

        current_episode_reward = 0
        
        for t in range(MAX_STEPS_PER_EPISODE):
            total_timesteps += 1
            llm_prompt = str(env_history_for_llm) + "\n>"

            if total_timesteps < START_STEPS:
                # Initial random exploration phase (using LLM suggestions randomly)
                 # Need access to gemini_api call here, maybe move call_gemini_api to agent?
                 # For now, assuming agent has access or we call it directly
                 try:
                     from gemini_api import call_gemini_api as direct_call_gemini
                     _, all_llm_actions_with_scores = direct_call_gemini(
                         prompt_text=llm_prompt,
                         candidate_count=agent.llm_candidate_count,
                         temperature=1.0 # High temperature for random start
                     )
                 except ImportError:
                     print("Error: Cannot import call_gemini_api directly in train_sac.py for random start.")
                     all_llm_actions_with_scores = [] # Fallback

                 suggested_actions_texts = [text for text, score in all_llm_actions_with_scores if text.strip() != '' and text != 'skip']
                 if not suggested_actions_texts:
                     action_text = "look"
                     action_index = 0 
                 else:
                     action_text = random.choice(suggested_actions_texts)
                     try:
                         # Index relative to the *suggested* list during random phase
                         action_index = suggested_actions_texts.index(action_text) 
                     except ValueError:
                         action_index = 0 # Fallback index
                 state_embedding = agent._preprocess_state_text(current_raw_obs) # Still need state embedding
            else:
                # Select action using SAC policy
                action_text, action_index, state_embedding = agent.select_action(llm_prompt, current_raw_obs, evaluate=False)

            # Environment step
            next_obs_list, reward_list, done_list, infos = env.step([action_text])
            next_raw_obs = process_ob(next_obs_list[0])
            reward = reward_list[0]
            done = done_list[0]

            next_state_embedding = agent._preprocess_state_text(next_raw_obs)

            # Store experience
            # Note: action_index here is the index within the suggested actions (0 to k-1)
            agent.add_experience(state_embedding, action_text, action_index, reward, next_state_embedding, done)

            # Update history for LLM
            if action_text.startswith("think:"):
                env_history_for_llm.add("action", action_text)
                env_history_for_llm.add("observation", "OK.")
            else:
                env_history_for_llm.add("action", action_text)
                env_history_for_llm.add("observation", next_raw_obs)

            current_raw_obs = next_raw_obs
            current_episode_reward += reward

            # --- Episode End Handling ---
            episode_finished = done or (t + 1) == MAX_STEPS_PER_EPISODE
            if episode_finished:
                success = infos.get('won', [False])[0] and done
                print(f"Episode {i_episode} finished after {t+1} steps. Reward: {current_episode_reward}. Success: {success}")
                
                if not success:
                    # Generate and store reflection on failure
                    print(f"Generating reflection for failed task: {game_file}")
                    failed_history_str = str(env_history_for_llm) 
                    # Pass only the most recent reflections used in the prompt
                    reflection = get_reflection(failed_history_str, past_reflections) 
                    print(f"Generated Reflection:\n{reflection}")
                    task_memory[game_file].append(reflection) # Add new reflection
                    if len(task_memory[game_file]) > MAX_MEMORY_PER_TASK * 2: 
                         task_memory[game_file] = task_memory[game_file][-MAX_MEMORY_PER_TASK*2:]
                
                break # End step loop for this episode

        # --- After Episode ---
        episode_rewards.append(current_episode_reward)

        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {i_episode}\tAverage Reward (last 10): {avg_reward:.2f}\tTotal Timesteps: {total_timesteps}")
            # Potentially save model checkpoint here

    env.close()
    print("SAC Training complete.")


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

    print("Starting SAC training with LLM guidance...")
    train_sac()
