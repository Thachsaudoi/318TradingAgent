import numpy as np
import pandas as pd
from agent import Agent
from environment import InvestmentEnv

# Load the trained Q-table
q_table = np.load("q_table.npy")
# Initialize the agent with the loaded Q-table
agent = Agent(n_states=100, n_actions=5)  # Ensure these match your environment
agent.q_table = q_table

# Initialize the environment
env = InvestmentEnv()
total_reward = 0

# Load the evaluation data
evaluation_data = pd.read_csv('evaluation_data.csv')
# Create a direct mapping from action indices to names
index_to_action = {
    0: "Stocks",
    1: "Real_Estates",
    2: "Commodities",
    3: "Cryptocurrencies",
    4: "Forex"
}

# Evaluate the agent on each state in the evaluation data
for index, row in evaluation_data.iterrows():
    state = int(row['State']) - 1  # Convert state to zero-based index
    env.state = state  # Set the environment to the specific state for evaluation

    # Select action based on the policy (max Q-value)
    action = np.argmax(agent.q_table[state])
    next_state, reward, done, info = env.step(action)

    # Optionally render the environment
    env.render()
    total_reward += reward
    action_index = np.argmax(agent.q_table[state])
    action_name = index_to_action[action_index]
    
    print(f"Evaluation on state {state + 1}: Action - {action_name}, Reward - {reward}")
print(f"Total reward after evaluation: {total_reward}")
