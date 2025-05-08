import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Any
from sentence_transformers import SentenceTransformer # Added import

# Assuming gemini_api.py is in the same directory or accessible via PYTHONPATH
from .gemini_api import call_gemini_api # Relative import

# Define the Q-network structure
class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 42):
        """
        Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action (or number of discrete actions)
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Simple feed-forward network - layers might need adjustment based on state/action representation
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the Replay Buffer
Experience = namedtuple("Experience", field_names=["state", "action_text", "action_index", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, seed: int = 42):
        """
        Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        np.random.seed(seed)

    def add(self, state: Any, action_text: str, action_index: int, reward: float, next_state: Any, done: bool):
        """Add a new experience to memory."""
        e = Experience(state, action_text, action_index, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> List[Experience]:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        return experiences

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)

# Define the DQN Agent
class DQNAgent:
    def __init__(self,
                 embedding_model_name: str = 'all-MiniLM-L6-v2', # Name of sentence transformer
                 llm_candidate_count: int = 5, # Number of actions LLM should suggest
                 buffer_size: int = int(1e5),
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 tau: float = 1e-3,
                 lr: float = 5e-4,
                 update_every: int = 4, # How often to update the network
                 device: str = None,
                 seed: int = 42):
        """
        Initialize an Agent object.
        """
        self.llm_candidate_count = llm_candidate_count
        # Action size for Q-network is the number of candidates LLM provides
        self.action_size = llm_candidate_count 
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sentence Transformer for state embeddings
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{embedding_model_name}': {e}")
            print("Please ensure 'sentence-transformers' is installed and the model name is correct.")
            # Depending on policy, you might want to re-raise or handle differently
            raise
        self.state_size = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Initialized SentenceTransformer, state_size: {self.state_size}")

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def add_experience(self, state_embedding: np.ndarray, action_text: str, action_index: int, reward: float, next_state_embedding: np.ndarray, done: bool):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # state_embedding and next_state_embedding are already processed numpy arrays
        self.memory.add(state_embedding, action_text, action_index, reward, next_state_embedding, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self._learn_from_experiences(experiences)

    def _preprocess_state_text(self, observation_text: str) -> np.ndarray:
        """Converts observation text to a state embedding."""
        embedding = self.embedding_model.encode(observation_text, convert_to_numpy=True)
        return embedding

    def select_action(self, state_text_for_llm_prompt: str, current_observation_text: str, epsilon: float = 0.1) -> Tuple[str, int]:
        """
        Selects an action using LLM guidance and Q-network.
        Args:
            state_text_for_llm_prompt (str): The full prompt text for the LLM (history + current obs + prompt char).
            current_observation_text (str): The latest observation text to be embedded for Q-network state.
            epsilon (float): for epsilon-greedy action selection.

        Returns:
            Tuple[str, int]: The chosen action string and its index among suggestions (0 to llm_candidate_count-1).
                             Returns ("look", 0) as fallback.
        """
        # 1. Get action suggestions from LLM
        _, all_llm_actions_with_scores = call_gemini_api(
            prompt_text=state_text_for_llm_prompt,
            candidate_count=self.llm_candidate_count,
            temperature=0.7 # Or other configured temperature for exploration
        )
        
        suggested_actions_texts = [action_text for action_text, score in all_llm_actions_with_scores if action_text.strip() != '' and action_text != 'skip']
        
        if not suggested_actions_texts:
            return "look", 0 # Fallback action text and index 0

        chosen_action_text: str
        chosen_action_index: int # This will be the index for the Q-network (0 to self.action_size-1)

        if random.random() < epsilon:
            # Explore: pick a random action from the LLM's suggestions
            random_idx_among_suggested = random.randrange(len(suggested_actions_texts))
            chosen_action_text = suggested_actions_texts[random_idx_among_suggested]
            # The chosen_action_index must correspond to the slot in Q-network's output.
            # If LLM suggests < llm_candidate_count actions, this index is relative to suggested_actions_texts.
            chosen_action_index = random_idx_among_suggested 
        else:
            # Exploit: use Q-network to pick the best action among LLM's suggestions
            state_embedding = self._preprocess_state_text(current_observation_text)
            state_tensor = torch.from_numpy(state_embedding).float().unsqueeze(0).to(self.device)

            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state_tensor) # Shape: (1, self.action_size)
            self.qnetwork_local.train()
            
            # We need to pick the best Q-value *for an action that was actually suggested*.
            # Q-values in action_values correspond to "slots" 0 to self.action_size - 1.
            # We only consider the first N slots, where N = len(suggested_actions_texts).
            num_valid_suggestions = len(suggested_actions_texts)
            if num_valid_suggestions == 0: # Should be caught earlier, but as a safeguard
                 return "look", 0

            # Consider Q-values only for the valid suggestions
            q_values_for_suggested_slots = action_values[0, :num_valid_suggestions]
            
            best_suggested_idx = torch.argmax(q_values_for_suggested_slots).item()
            chosen_action_text = suggested_actions_texts[best_suggested_idx]
            chosen_action_index = best_suggested_idx # This index is directly usable for Q-network

        return chosen_action_text, chosen_action_index


    def _learn_from_experiences(self, experiences: List[Experience]):
        """
        Update Q-values using given batch of experience tuples.
        Args:
            experiences (List[Experience]): list of (state_embedding, action_text, action_index, reward, next_state_embedding, done) tuples
        """
        # state_embeddings are already numpy arrays from _preprocess_state_text
        state_embeddings, _, action_indices, rewards, next_state_embeddings, dones = zip(*experiences)

        # Ensure all elements in state_embeddings and next_state_embeddings are numpy arrays
        # This is important if any 'None' states were stored (e.g. terminal next_state)
        # However, our current add_experience stores the embedding directly.
        
        states_tensor = torch.from_numpy(np.array(state_embeddings)).float().to(self.device)
        actions_tensor = torch.LongTensor(action_indices).unsqueeze(1).to(self.device) # action_indices are 0 to k-1
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.from_numpy(np.array(next_state_embeddings)).float().to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Get max predicted Q values for next states from target model
        # The Q-values from target network are for the self.action_size "slots"
        q_targets_next = self.qnetwork_target(next_states_tensor).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        q_targets = rewards_tensor + (self.gamma * q_targets_next * (1 - dones_tensor))

        # Get expected Q values from local model
        # actions_tensor contains the indices of the actions taken (0 to self.action_size-1)
        q_expected = self.qnetwork_local(states_tensor).gather(1, actions_tensor)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update_target_network(self.qnetwork_local, self.qnetwork_target)

    def _soft_update_target_network(self, local_model: nn.Module, target_model: nn.Module):
        """Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    # Removed old placeholder preprocessing methods

if __name__ == '__main__':
    # Example Usage (conceptual)
    print("DQNAgent structure defined.")
    # Example instantiation (ensure sentence-transformers is installed and model is downloadable)
    # try:
    #     agent = DQNAgent(embedding_model_name='all-MiniLM-L6-v2', llm_candidate_count=5)
    #     print(f"Agent initialized on device: {agent.device} with state_size: {agent.state_size}")
    # except Exception as e:
    #     print(f"Could not instantiate DQNAgent: {e}")
    
    # This would require actual state/action data and implemented preprocessing to run fully.
