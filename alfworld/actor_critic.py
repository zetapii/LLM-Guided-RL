import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 42):
        """
        Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state (embedding dimension)
            action_size (int): Number of discrete actions to choose from (llm_candidate_count)
            seed (int): Random seed
        """
        super(ActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Shared layers (optional, can have separate networks too)
        self.shared_fc1 = nn.Linear(state_size, 128)
        self.shared_fc2 = nn.Linear(128, 64)

        # Actor head: outputs logits for action probabilities
        self.actor_head = nn.Linear(64, action_size)

        # Critic head: outputs state value
        self.critic_head = nn.Linear(64, 1)

    def forward(self, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a network that maps state -> action probabilities and state value.
        Args:
            state_embedding (torch.Tensor): Tensor representation of the state.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Action logits (for policy).
                - State value (for critic).
        """
        x = F.relu(self.shared_fc1(state_embedding))
        x = F.relu(self.shared_fc2(x))
        
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        
        return action_logits, state_value

    def get_action_distribution(self, state_embedding: torch.Tensor) -> Categorical:
        """Returns a categorical distribution over actions."""
        action_logits, _ = self.forward(state_embedding)
        return Categorical(logits=action_logits)

    def get_value(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Returns the state value."""
        _, state_value = self.forward(state_embedding)
        return state_value

if __name__ == '__main__':
    # Example Usage
    dummy_state_size = 384 # Example embedding size
    dummy_action_size = 5  # Example: 5 LLM candidate actions
    
    network = ActorCriticNetwork(dummy_state_size, dummy_action_size)
    print("ActorCriticNetwork initialized:")
    print(network)

    # Create a dummy state tensor (batch size 1)
    dummy_state = torch.randn(1, dummy_state_size)
    
    action_logits, state_value = network(dummy_state)
    print(f"\nOutput for dummy state:")
    print(f"Action Logits: {action_logits}")
    print(f"State Value: {state_value}")

    action_dist = network.get_action_distribution(dummy_state)
    sampled_action = action_dist.sample()
    log_prob = action_dist.log_prob(sampled_action)
    print(f"Sampled Action: {sampled_action.item()}, Log Probability: {log_prob.item()}")
    
    value_output = network.get_value(dummy_state)
    print(f"Value from get_value: {value_output.item()}")
