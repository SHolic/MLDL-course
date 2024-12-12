import gymnasium as gym
import numpy as np

# Create the CartPole-v1 environment with render_mode="human" to visualize the simulation
env = gym.make('CartPole-v1', render_mode='human')

# Number of episodes to run
num_episodes = 10
total_rewards = []

for episode in range(num_episodes):
    # Reset the environment to get the initial state and info
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select a random action (0 or 1 for CartPole-v1)
        action = env.action_space.sample()
        
        # Take the selected action and observe the next state, reward, termination status, truncation status, and info
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Check if the episode is done
        done = terminated or truncated
        
        # Update the total reward
        total_reward += reward
        
        # Update the current state
        state = next_state
        
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    total_rewards.append(total_reward)

print(f"Average Reward with Random Policy: {np.mean(total_rewards)}")

# Close the environment when done
env.close()
