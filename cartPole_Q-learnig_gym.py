"""
Generated using ChatGPT 4o
"""

import numpy as np
import gymnasium as gym

# Create the CartPole-v1 environment with render_mode="human" to visualize the simulation
#env = gym.make('CartPole-v1', render_mode='human')
# Create the CartPole-v1 environment without simulation
env = gym.make('CartPole-v1')


# Initialize Q-table with zeros (simplified example)
state_space = (20, 20, 20, 20)  # Discretized state space for cart position, velocity, pole angle, and angular velocity
action_space = env.action_space.n  # Number of possible actions
q_table = np.zeros(state_space + (action_space,))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 1000
total_rewards = []

# Function to discretize the state space
def discretize_state(state, bins):
    # Discretize each component of the state using corresponding bins
    discretized = tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))
    return discretized

# Discretization bins
state_bins = [
    np.linspace(-2.4, 2.4, state_space[0] - 1),    # Cart position
    np.linspace(-3.0, 3.0, state_space[1] - 1),    # Cart velocity
    np.linspace(-0.5, 0.5, state_space[2] - 1),    # Pole angle
    np.linspace(-2.0, 2.0, state_space[3] - 1)     # Pole angular velocity
]

# Q-learning algorithm
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state, state_bins)
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
        
        # Take action and observe outcome
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_state, state_bins)
        done = terminated or truncated
        
        # Update Q-table
        best_future_q = np.max(q_table[next_state])
        #q_table[state + (action,)] += alpha * (reward + gamma * best_future_q - q_table[state + (action,)])
        q_table[state + (action,)] = (1-alpha) * q_table[state + (action,)] + alpha * (reward + gamma * best_future_q)
        # Update state and reward
        state = next_state
        total_reward += reward
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    total_rewards.append(total_reward)

print(f"Average Reward with Q-Learning: {np.mean(total_rewards)}")

env.close()
