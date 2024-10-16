import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

from maze_generator import generate_maze
from dijkstra import maze_to_graph, dijkstra

class MazeEnv(gym.Env):
    """Custom environment for maze navigation."""
    def __init__(self, maze, entrance, exit):
        super(MazeEnv, self).__init__()
        self.maze = maze
        self.n = len(maze)
        self.entrance = entrance[0]
        self.exit = exit[0]
        self.action_space = spaces.Discrete(4)  # North, South, East, West
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.n * self.n,), dtype=np.uint8)
        self.reset()
        
    def reset(self):
        """Reset the environment to the entrance."""
        self.agent_pos = self.entrance
        self.steps = 0
        return self._get_observation()
    
    def step(self, action):
        """Take an action and return the result."""
        x, y = self.agent_pos
        if action == 0 and not self.maze[y][x]['N']:
            y = max(0, y - 1)
        elif action == 1 and not self.maze[y][x]['S']:
            y = min(self.n - 1, y + 1)
        elif action == 2 and not self.maze[y][x]['W']:
            x = max(0, x - 1)
        elif action == 3 and not self.maze[y][x]['E']:
            x = min(self.n - 1, x + 1)
        
        self.agent_pos = (x, y)
        self.steps += 1
        
        done = self.agent_pos == self.exit
        reward = 0 if done else -1
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """Get the current state observation."""
        obs = np.zeros((self.n, self.n), dtype=np.uint8)
        x, y = self.agent_pos
        obs[y][x] = 1
        return obs.flatten()

class DQN(nn.Module):
    """Deep Q-Network architecture."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        """Forward pass."""
        return self.model(x)

class ReplayMemory:
    """Experience replay buffer."""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        """Store a transition."""
        self.memory.append(transition)
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return memory size."""
        return len(self.memory)

# Generate the maze
n = 8  # Maze size
maze, entrance, exit = generate_maze(n)

# Convert maze to graph and find optimal path
graph = maze_to_graph(maze)
entrance_pos = entrance[0]
exit_pos = exit[0]
optimal_path = max(len(dijkstra(graph, entrance_pos, exit_pos)) - 1, 0)
print(f"Optimal path: {optimal_path} steps")

env = MazeEnv(maze, entrance, exit)
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# Initialize networks
policy_net = DQN(obs_size, n_actions)
target_net = DQN(obs_size, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(10000)

# Hyperparameters
num_episodes = 500
max_steps = n**4
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
target_update = 10

epsilon = epsilon_start

# Initialize a list to store the number of steps per episode
steps_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0)
    total_reward = 0
    for _ in range(max_steps):
        # Select action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()
        
        # Execute action
        next_state, reward, done, _ = env.step(action)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
        action_tensor = torch.tensor([[action]], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        total_reward += reward

        # Store transition
        memory.push((state, action_tensor, next_state_tensor, reward_tensor, done))        
        state = next_state_tensor
        
        # Optimize model
        if len(memory) >= batch_size:
            transitions = memory.sample(batch_size)
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
            
            batch_state = torch.cat(batch_state)
            batch_action = torch.cat(batch_action)
            batch_next_state = torch.cat(batch_next_state)
            batch_reward = torch.cat(batch_reward)
            batch_done = torch.tensor(batch_done, dtype=torch.float32)
            
            q_values = policy_net(batch_state).gather(1, batch_action)
            
            with torch.no_grad():
                next_q_values = target_net(batch_next_state).max(1)[0]
                target_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))
                target_q_values = target_q_values.unsqueeze(1)
            
            loss = nn.functional.mse_loss(q_values, target_q_values)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done:
            break
    
    # Decay epsilon
    if epsilon > epsilon_end:
        epsilon *= epsilon_decay
    
    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    # Append the number of steps taken in this episode
    steps_per_episode.append(env.steps)
    
    print(f"Episode {episode + 1}/{num_episodes}, Steps Taken: {env.steps}, Epsilon: {epsilon:.2f}")

# Save trained model
torch.save(policy_net.state_dict(), 'dqn_maze_solver.pth')

# Plotting the number of steps taken vs episode
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_episodes + 1), steps_per_episode, label='Steps per Episode')
plt.axhline(y=optimal_path, color='r', linestyle='--', label='Optimal Steps')
plt.xlabel('Episode')
plt.ylabel('Number of Steps Taken')
plt.title('DQN Maze Solver Performance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()