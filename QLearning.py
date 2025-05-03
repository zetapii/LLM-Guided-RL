import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from minigrid.wrappers import ImgObsWrapper
from llama import suggest_llm_action


# Hyperparameters
lr = 3e-4
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
max_episodes = 10000
max_steps = 500

name_of_environment = 'MiniGrid-DoorKey-6x6-v0'

env = gym.make(name_of_environment, render_mode='rgb_array', agent_view_size=3)

class QNetworkCNN(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Dynamically calculate CNN output size
        with torch.no_grad():
            dummy = torch.rand(1, 3, *obs_shape[:2])
            cnn_out = self.cnn(dummy).shape[1]
        
        self.q_network = nn.Sequential(
            nn.Linear(cnn_out, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # Permute to (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)
        return self.q_network(features)

class QLearningAgent:
    def __init__(self, obs_shape, action_dim):
        self.q_network = QNetworkCNN(obs_shape, action_dim)
        self.target_network = QNetworkCNN(obs_shape, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.steps = 0
        
        # Replay buffer
        self.memory = []
        self.batch_size = 64
        self.memory_size = 10000
    def get_action(self, obs):
        # Epsilon decay
        self.steps += 1
        self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)
        self.epsilon = 1
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # return random.randint(0, self.action_dim - 1)
            action = suggest_llm_action(env, name_of_environment, env.action_space.n, obs) 
            return action 
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[None], dtype=torch.float32) / 10.0
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        # Store in memory, remove oldest if full
        transition = (state, action, reward, next_state, done)
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(transition)
    def update(self):
        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32) / 10.0
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32) / 10.0
        dones = torch.tensor(dones, dtype=torch.float32)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update of target network
        self._soft_update()
    def _soft_update(self, tau=0.001):
        for target_param, param in zip(self.target_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Training setup
env = gym.make('MiniGrid-KeyCorridorS3R1-v0', render_mode='rgb_array')
env = ImgObsWrapper(env)

obs_shape = env.observation_space.shape
action_dim = env.action_space.n
agent = QLearningAgent(obs_shape, action_dim)

# Training loop
for episode in range(max_episodes):
    obs, _ = env.reset()
    obs = obs.astype(np.float32)
    total_reward = 0
    
    for step in range(max_steps):
        # Get action
        action = agent.get_action(obs)
        
        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_obs = next_obs.astype(np.float32)
        
        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)
        
        # Update
        agent.update()
        
        obs = next_obs
        total_reward += reward
        
        if done:
            break
    
    # Print episode statistics
    print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

env.close()