import numpy as np
import pandas as pd
from openaiTradingBot.envs.agent import Agent



action_mapping = {
    "Stocks": 0,
    "Real_Estates": 1,
    "Commodities": 2,
    "Cryptocurrencies": 3,
    "Forex": 4
}


# Load the dataset
data = pd.read_csv('training_data.csv')

# Initialize the agent
n_states = 100  # This should match the number of states in your environment
n_actions = 5   # This should match the number of actions in your environment
agent = Agent(n_states=n_states, n_actions=n_actions)

# Training loop
for index, row in data.iterrows():
    # Adjust state for zero-based indexing and ensure it's within valid range
    state = int(row['State']) - 1
    if state < 0 or state >= agent.n_states:
        continue  # Skip this row if the state is out of bounds
    
    action = action_mapping[row['Action']]
    
    # Adjust next_state for zero-based indexing and ensure it's within valid range
    next_state = int(row['NextState']) - 1
    if next_state < 0 or next_state >= agent.n_states:
        continue  # Skip this row if the next_state is out of bounds
    
    reward = row['Reward']
    
    # Update the Q-table with the experience from the dataset
    agent.learn(state, action, reward, next_state)

# Save the Q-table for later use
np.save("q_table.npy", agent.q_table)