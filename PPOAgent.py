import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical

from constants import name_of_environment, env, lr, gamma 
from llama import suggest_llm_action

class ActorCriticCNN(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(ActorCriticCNN, self).__init__()
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate feature size
        feature_size = self._get_feature_size(obs_shape)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_feature_size(self, obs_shape):
        # Forward pass with dummy input to get feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            return self.features(dummy_input).shape[1]
    
    def forward(self, x):
        features = self.features(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, x, action=None):
        action_logits, value = self(x)
        dist = Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        
        return action, action_log_probs, entropy, value

class PPOAgent:
    def __init__(self, obs_shape, action_dim, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01, lambda_gae=0.95, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.network = ActorCriticCNN(obs_shape, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.action_dim = action_dim
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lambda_gae = lambda_gae
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        
        # PPO buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def get_action(self, obs):
        # Epsilon decay
        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon = 1  # As specified in original code
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # return random.randint(0, self.action_dim - 1)
            action = suggest_llm_action(env, name_of_environment, env.action_space.n, obs) 
            return action 
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[None], dtype=torch.float32) / 10.0
                action_logits, _ = self.network(obs_tensor)
                return action_logits.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self):
        # Convert to numpy arrays for easier processing
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = values[-1]
        
        # Compute GAE (Generalized Advantage Estimation)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + gamma * self.lambda_gae * next_non_terminal * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, epochs=4, batch_size=64):
        if len(self.states) < batch_size:
            return
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages()
        
        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32) / 10.0
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch training
        for _ in range(epochs):
            # Generate random indices
            indices = np.random.permutation(len(self.states))
            
            # Train in mini-batches
            for start_idx in range(0, len(self.states), batch_size):
                idx = indices[start_idx:start_idx + batch_size]
                
                # Get mini-batch
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # Forward pass
                _, new_log_probs, entropy, values = self.network.get_action_and_value(mb_states, mb_actions)
                
                # Calculate ratio and clipped ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                
                # Clipped surrogate objective
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                # Optional: gradient clipping
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def get_value_and_logprob(self, state, action):
        with torch.no_grad():
            state_tensor = torch.tensor(state[None], dtype=torch.float32) / 10.0
            _, log_prob, _, value = self.network.get_action_and_value(state_tensor, 
                                                torch.tensor([action], dtype=torch.long))
        return value.item(), log_prob.item()