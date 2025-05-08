import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

from .gemini_api import call_gemini_api # Relative import
from .actor_critic import ActorCriticNetwork # Relative import

class PPOAgent:
    def __init__(self,
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 llm_candidate_count: int = 5,
                 gamma: float = 0.99,       # Discount factor
                 gae_lambda: float = 0.95,  # Lambda for GAE
                 ppo_clip_epsilon: float = 0.2, # PPO clipping parameter
                 epochs_per_update: int = 10, # Number of epochs for updating policy per batch
                 mini_batch_size: int = 64,
                 lr: float = 3e-4,
                 vf_coef: float = 0.5,      # Value function loss coefficient
                 entropy_coef: float = 0.01, # Entropy bonus coefficient
                 device: str = None,
                 seed: int = 42):

        self.llm_candidate_count = llm_candidate_count
        self.action_size = llm_candidate_count # Actor outputs probs for these candidates
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.mini_batch_size = mini_batch_size
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{embedding_model_name}': {e}")
            raise
        self.state_size = self.embedding_model.get_sentence_embedding_dimension()
        print(f"PPOAgent: Initialized SentenceTransformer, state_size: {self.state_size}")

        self.network = ActorCriticNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Temporary storage for a batch of trajectories
        self.trajectory_data = {
            "states": [], "action_texts": [], "action_indices": [], "log_probs": [],
            "rewards": [], "values": [], "dones": [], "next_states": []
        }
        print(f"PPOAgent initialized on device: {self.device}")

    def _preprocess_state_text(self, observation_text: str) -> np.ndarray:
        """Converts observation text to a state embedding."""
        return self.embedding_model.encode(observation_text, convert_to_numpy=True)

    def select_action(self, state_text_for_llm_prompt: str, current_observation_text: str) -> Tuple[str, int, float, float, np.ndarray]:
        """
        Selects an action using LLM guidance and Actor-Critic network.
        Returns:
            Tuple[str, int, float, float, np.ndarray]: 
                - chosen_action_text (str)
                - chosen_action_index (int, 0 to llm_candidate_count-1)
                - log_prob_of_action (float)
                - state_value (float)
                - state_embedding (np.ndarray)
        """
        state_embedding = self._preprocess_state_text(current_observation_text)
        state_tensor = torch.from_numpy(state_embedding).float().unsqueeze(0).to(self.device)

        _, all_llm_actions_with_scores = call_gemini_api(
            prompt_text=state_text_for_llm_prompt,
            candidate_count=self.llm_candidate_count,
            temperature=0.7 # Exploration via LLM temperature
        )
        
        suggested_actions_texts = [text for text, score in all_llm_actions_with_scores if text.strip() != '' and text != 'skip']
        
        if not suggested_actions_texts: # Fallback
            # This case needs careful handling. If LLM fails, PPO actor has no valid actions.
            # For now, return a "look" action with placeholder values.
            # The actor network's output for this "look" action (if slot 0) will be used.
            print("Warning: LLM provided no valid actions. Using 'look' as fallback.")
            suggested_actions_texts = ["look"] 
            # We might need a default embedding or handling if "look" is always slot 0.

        self.network.eval() # Set network to evaluation mode for action selection
        with torch.no_grad():
            action_distribution = self.network.get_action_distribution(state_tensor)
            state_value = self.network.get_value(state_tensor).item()
        self.network.train() # Set back to train mode

        # Sample an action based on the policy output over the *slots* for LLM suggestions.
        # We only consider the slots for which LLM provided an action.
        num_valid_suggestions = len(suggested_actions_texts)
        
        # Create a distribution over the validly suggested actions
        # Mask logits for non-suggested actions if network output > num_valid_suggestions
        # For simplicity, assume network's action_size matches llm_candidate_count
        # and we only sample from the first num_valid_suggestions.
        
        # Get logits for the valid suggestions
        logits_for_valid_suggestions = action_distribution.logits[0, :num_valid_suggestions]
        
        if num_valid_suggestions == 0: # Should have been caught, but safeguard
            chosen_action_index = 0 
            chosen_action_text = "look"
            # Create a dummy distribution for "look" at index 0
            dummy_logits = torch.zeros(self.action_size).to(self.device)
            dummy_logits[0] = 1.0 # Give some probability to "look"
            action_distribution_for_log_prob = Categorical(logits=dummy_logits.unsqueeze(0))
            log_prob_of_action = action_distribution_for_log_prob.log_prob(torch.tensor([0]).to(self.device)).item()

        elif num_valid_suggestions == 1:
            chosen_action_index = 0
            chosen_action_text = suggested_actions_texts[0]
            # Log prob of selecting the 0-th slot from the original distribution
            log_prob_of_action = action_distribution.log_prob(torch.tensor([0]).to(self.device)).item()
        else:
            # Create a new distribution only over the validly suggested actions
            dist_over_suggested = Categorical(logits=logits_for_valid_suggestions)
            sampled_idx_within_suggested = dist_over_suggested.sample()
            
            chosen_action_index = sampled_idx_within_suggested.item()
            chosen_action_text = suggested_actions_texts[chosen_action_index]
            # Log prob comes from the original distribution for the chosen index
            log_prob_of_action = action_distribution.log_prob(torch.tensor([chosen_action_index]).to(self.device)).item()
            
        return chosen_action_text, chosen_action_index, log_prob_of_action, state_value, state_embedding

    def store_transition(self, state_embedding, action_text, action_index, log_prob, reward, value, done, next_state_embedding):
        self.trajectory_data["states"].append(state_embedding)
        self.trajectory_data["action_texts"].append(action_text) # For debugging/logging
        self.trajectory_data["action_indices"].append(action_index)
        self.trajectory_data["log_probs"].append(log_prob)
        self.trajectory_data["rewards"].append(reward)
        self.trajectory_data["values"].append(value)
        self.trajectory_data["dones"].append(done)
        self.trajectory_data["next_states"].append(next_state_embedding) # For GAE calculation if needed for last step

    def _calculate_advantages_gae(self, rewards, values, dones, last_next_value_tensor):
        advantages = []
        gae = 0.0
        # Ensure values include the value of the state *after* the last action in the trajectory
        # If the trajectory ends in 'done', V(s_next_terminal) = 0.
        # Otherwise, V(s_next_non_terminal) is estimated by the critic.
        
        # `values` are V(s_t), `rewards` are R_t+1, `dones` are D_t+1
        # We need V(s_t+1) for delta.
        # The `values` list has T entries. `rewards` and `dones` have T entries.
        # We need one more value for the state *after* the last action.
        
        extended_values = values + [last_next_value_tensor.item()]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * extended_values[t+1] * (1 - dones[t]) - extended_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = advantages_tensor + torch.FloatTensor(values).to(self.device) # Rt = At + Vt
        return advantages_tensor, returns_tensor

    def update(self, last_next_observation_text_for_embedding: str):
        if not self.trajectory_data["states"]:
            return # No data to update

        # Convert trajectory data to tensors
        states_np = np.array(self.trajectory_data["states"])
        action_indices_np = np.array(self.trajectory_data["action_indices"])
        log_probs_np = np.array(self.trajectory_data["log_probs"])
        rewards_np = np.array(self.trajectory_data["rewards"])
        values_np = np.array(self.trajectory_data["values"])
        dones_np = np.array(self.trajectory_data["dones"])

        states_tensor = torch.from_numpy(states_np).float().to(self.device)
        action_indices_tensor = torch.LongTensor(action_indices_np).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs_np).to(self.device)
        
        # Calculate value for the state after the last action in the trajectory
        with torch.no_grad():
            last_next_state_embedding = self._preprocess_state_text(last_next_observation_text_for_embedding)
            last_next_state_tensor = torch.from_numpy(last_next_state_embedding).float().unsqueeze(0).to(self.device)
            last_next_value_tensor = self.network.get_value(last_next_state_tensor)
            if self.trajectory_data["dones"][-1]: # If last step was terminal
                 last_next_value_tensor.fill_(0.0)


        advantages_tensor, returns_tensor = self._calculate_advantages_gae(
            self.trajectory_data["rewards"], self.trajectory_data["values"], self.trajectory_data["dones"], last_next_value_tensor
        )
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8) # Normalize

        num_samples = len(states_np)

        for _ in range(self.epochs_per_update):
            # Create minibatches by shuffling indices
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states_tensor[batch_indices]
                batch_actions = action_indices_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Get new log_probs, values, entropy from current policy
                new_action_dist = self.network.get_action_distribution(batch_states)
                new_log_probs = new_action_dist.log_prob(batch_actions)
                entropy = new_action_dist.entropy().mean()
                new_values = self.network.get_value(batch_states).squeeze(-1) # Remove last dim

                # Actor (Policy) Loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic (Value) Loss
                critic_loss = F.mse_loss(new_values, batch_returns)

                # Total Loss
                loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Grad clipping
                self.optimizer.step()

        # Clear trajectory data after update
        self.trajectory_data = {k: [] for k in self.trajectory_data}


if __name__ == '__main__':
    print("PPOAgent structure defined.")
    # try:
    #     agent = PPOAgent(embedding_model_name='all-MiniLM-L6-v2', llm_candidate_count=3)
    #     print(f"PPO Agent initialized on device: {agent.device} with state_size: {agent.state_size}")
    #     # Dummy data for testing select_action
    #     dummy_llm_prompt = "Initial observation.\nYour task is to X.\n>"
    #     dummy_obs_text = "Initial observation."
    #     action, idx, log_prob, value, state_emb = agent.select_action(dummy_llm_prompt, dummy_obs_text)
    #     print(f"Selected action: '{action}' (idx: {idx}), log_prob: {log_prob:.3f}, value: {value:.3f}")
    # except Exception as e:
    #     print(f"Could not instantiate or use PPOAgent: {e}")
