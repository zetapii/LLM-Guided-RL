import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

from .gemini_api import call_gemini_api

# Replay Buffer (can be shared or copied from dqn_agent.py)
Experience = namedtuple("Experience", field_names=["state_embedding", "action_text", "action_index", "reward", "next_state_embedding", "done"])

class ReplayBuffer: # Identical to DQN's, assuming state_embedding is stored
    def __init__(self, buffer_size: int, batch_size: int, seed: int = 42):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        np.random.seed(seed)

    def add(self, state_embedding, action_text, action_index, reward, next_state_embedding, done):
        e = Experience(state_embedding, action_text, action_index, reward, next_state_embedding, done)
        self.memory.append(e)

    def sample(self) -> List[Experience]:
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self) -> int:
        return len(self.memory)

# Define Networks for SAC
class SACPolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 42):
        super(SACPolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_logits = nn.Linear(128, action_size) # Outputs logits for discrete actions

    def forward(self, state_embedding: torch.Tensor) -> Categorical:
        x = F.relu(self.fc1(state_embedding))
        x = F.relu(self.fc2(x))
        action_logits = self.fc_logits(x)
        return Categorical(logits=action_logits)

class SACQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 42):
        super(SACQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Q-network takes state and action (index) to predict Q-value.
        # For discrete actions, it's common for Q-network to output Q-values for all actions.
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size) # Outputs Q-value for each discrete action slot

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state_embedding))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACAgent:
    def __init__(self,
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 llm_candidate_count: int = 5,
                 gamma: float = 0.99,       # Discount factor
                 tau: float = 0.005,        # Soft update factor for target networks
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4,    # Learning rate for temperature alpha
                 alpha_init: float = 0.2,   # Initial temperature, or target entropy if auto
                 buffer_size: int = int(1e5),
                 batch_size: int = 128,
                 update_every: int = 1,     # How often to update networks
                 target_entropy_scale: float = 0.98, # For auto alpha: target_entropy = -target_entropy_scale * log(|A|)
                 device: str = None,
                 seed: int = 42):

        self.llm_candidate_count = llm_candidate_count
        self.action_size = llm_candidate_count # Policy/Q-networks operate on these candidate slots
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
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
        print(f"SACAgent: Initialized SentenceTransformer, state_size: {self.state_size}")

        # Actor Network
        self.actor = SACPolicyNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic Networks (Q1, Q2 and their targets)
        self.critic1 = SACQNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.critic1_target = SACQNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)

        self.critic2 = SACQNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.critic2_target = SACQNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Temperature (alpha) - for discrete actions, can be fixed or learned
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        # Target entropy: For discrete actions, it's often -log(1/|A|) * scale = log(|A|) * scale
        # Here |A| is self.action_size (number of LLM candidate slots)
        self.target_entropy = -target_entropy_scale * np.log(1.0 / self.action_size) if self.action_size > 0 else 0.0


        self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        self.t_step = 0
        print(f"SACAgent initialized on device: {self.device}")

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _preprocess_state_text(self, observation_text: str) -> np.ndarray:
        return self.embedding_model.encode(observation_text, convert_to_numpy=True)

    def select_action(self, state_text_for_llm_prompt: str, current_observation_text: str, evaluate: bool = False) -> Tuple[str, int, np.ndarray]:
        """
        Selects an action using LLM guidance and SAC policy.
        Returns: Tuple[str, int, np.ndarray] - chosen_action_text, chosen_action_index, state_embedding
        """
        state_embedding = self._preprocess_state_text(current_observation_text)
        state_tensor = torch.from_numpy(state_embedding).float().unsqueeze(0).to(self.device)

        _, all_llm_actions_with_scores = call_gemini_api(
            prompt_text=state_text_for_llm_prompt,
            candidate_count=self.llm_candidate_count,
            temperature=0.7 # LLM exploration
        )
        suggested_actions_texts = [text for text, score in all_llm_actions_with_scores if text.strip() != '' and text != 'skip']
        
        if not suggested_actions_texts:
            return "look", 0, state_embedding # Fallback

        num_valid_suggestions = len(suggested_actions_texts)
        
        self.actor.eval()
        with torch.no_grad():
            action_dist = self.actor(state_tensor) # Categorical distribution over action_size slots
        self.actor.train()

        # Consider only the part of distribution corresponding to valid suggestions
        logits_for_valid_suggestions = action_dist.logits[0, :num_valid_suggestions]
        
        if num_valid_suggestions == 0: # Safeguard
            chosen_action_index = 0
            chosen_action_text = "look"
        elif evaluate: # Deterministic action for evaluation
            if num_valid_suggestions == 1:
                 chosen_action_index = 0
            else:
                 chosen_action_index = torch.argmax(logits_for_valid_suggestions).item()
            chosen_action_text = suggested_actions_texts[chosen_action_index]
        else: # Stochastic action for training
            if num_valid_suggestions == 1:
                chosen_action_index = 0
            else:
                dist_over_suggested = Categorical(logits=logits_for_valid_suggestions)
                chosen_action_index = dist_over_suggested.sample().item()
            chosen_action_text = suggested_actions_texts[chosen_action_index]
            
        return chosen_action_text, chosen_action_index, state_embedding


    def add_experience(self, state_embedding, action_text, action_index, reward, next_state_embedding, done):
        self.memory.add(state_embedding, action_text, action_index, reward, next_state_embedding, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences: List[Experience]):
        state_embeddings, _, action_indices, rewards, next_state_embeddings, dones = zip(*experiences)

        states_tensor = torch.from_numpy(np.array(state_embeddings)).float().to(self.device)
        actions_tensor = torch.LongTensor(action_indices).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.from_numpy(np.array(next_state_embeddings)).float().to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Update Critic Networks ---
        with torch.no_grad():
            next_action_dist = self.actor(next_states_tensor)
            next_action_probs = next_action_dist.probs # (batch_size, action_size)
            next_log_probs = next_action_dist.logits # Using logits as approx for log_probs before softmax for stability
                                                    # Or use next_action_dist.log_prob(next_action_dist.sample()) if needed
                                                    # For discrete SAC, target is E_{a'~pi}[Q_target(s',a') - alpha * log pi(a'|s')]
                                                    # This means summing over next actions weighted by their probs.
            
            q1_target_next = self.critic1_target(next_states_tensor) # (batch_size, action_size)
            q2_target_next = self.critic2_target(next_states_tensor) # (batch_size, action_size)
            
            # Expected value under current policy for next state's Q-values
            min_q_target_next_expected = torch.sum(next_action_probs * (torch.min(q1_target_next, q2_target_next) - self.alpha * next_action_dist.logits), dim=1, keepdim=True)
            # Using logits for log_probs here. For Categorical, log_probs = logits - logsumexp(logits)
            # A more accurate way for discrete SAC:
            # next_actions_sampled, next_log_probs_sampled = self.actor.sample(next_states_tensor) # if actor had sample method
            # min_q_target_next = torch.min(
            #    self.critic1_target(next_states_tensor).gather(1, next_actions_sampled),
            #    self.critic2_target(next_states_tensor).gather(1, next_actions_sampled)
            # ) - self.alpha * next_log_probs_sampled
            # For now, using the expectation over all action slots.

            q_target = rewards_tensor + (self.gamma * (1 - dones_tensor) * min_q_target_next_expected)

        q1_current = self.critic1(states_tensor).gather(1, actions_tensor)
        q2_current = self.critic2(states_tensor).gather(1, actions_tensor)

        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Update Actor Network (and Alpha if auto) ---
        current_action_dist = self.actor(states_tensor)
        # To get log_probs for policy loss, we need log_prob of *some* action.
        # Typically, one would sample actions or use the ones from buffer if on-policy.
        # For off-policy SAC, new actions are sampled from current policy.
        # Here, we evaluate Q based on current policy's expectation.
        
        # For discrete actions, actor loss is E_{s~D} [ E_{a~pi} [alpha * log pi(a|s) - Q(s,a)] ]
        # Q(s,a) is min(Q1(s,a), Q2(s,a))
        
        current_action_probs = current_action_dist.probs
        current_log_probs = current_action_dist.logits # Approx log_probs
        
        q1_for_actor_loss = self.critic1(states_tensor).detach() # Detach critic from actor's update
        q2_for_actor_loss = self.critic2(states_tensor).detach()
        min_q_for_actor_loss = torch.min(q1_for_actor_loss, q2_for_actor_loss)
        
        # Actor loss: sum over action probs * (alpha * log_probs - Q_values)
        actor_loss = (current_action_probs * (self.alpha.detach() * current_log_probs - min_q_for_actor_loss)).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Alpha (Temperature) ---
        alpha_loss = -(self.log_alpha * (current_log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft Update Target Networks ---
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

if __name__ == '__main__':
    print("SACAgent structure defined.")
    # try:
    #     agent = SACAgent(embedding_model_name='all-MiniLM-L6-v2', llm_candidate_count=3)
    #     print(f"SAC Agent initialized on device: {agent.device} with state_size: {agent.state_size}")
    #     dummy_llm_prompt = "Initial observation.\nYour task is to X.\n>"
    #     dummy_obs_text = "Initial observation."
    #     action_text, action_idx, s_emb = agent.select_action(dummy_llm_prompt, dummy_obs_text)
    #     print(f"Selected action: '{action_text}' (idx: {action_idx})")
    # except Exception as e:
    #     print(f"Could not instantiate or use SACAgent: {e}")
