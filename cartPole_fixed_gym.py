"""
Generated using ChatGPT 4o
"""

import gymnasium as gym
import numpy as np
# Create the CartPole-v1 environment with render_mode="human" to visualize the simulation
#env = gym.make('CartPole-v1', render_mode='human')
# Create the CartPole-v1 environment without simulation
env = gym.make('CartPole-v1')

# Number of episodes to run
num_episodes = 1000
total_rewards = []

for episode in range(num_episodes):
    # Reset the environment to get the initial state
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    
    while not done:
        # The state consists of: [cart position, cart velocity, pole angle, pole angular velocity]
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
        
        # Fixed strategy: if the pole is leaning right, push right; if leaning left, push left
        action = 0 if pole_angle < 1 else 1
        
        # Take the selected action and observe the next state and reward
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Check if the episode is done
        done = terminated or truncated
        
        # Update the total reward
        total_reward += reward
        
        # Update the current state
        state = next_state
        
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    total_rewards.append(total_reward)

print(f"Average Reward with Fixed Policy: {np.mean(total_rewards)}")

# Close the environment when done
env.close()
