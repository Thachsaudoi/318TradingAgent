import numpy as np
import pandas as pd

# Define states and actions
states = [str(i) for i in range(1, 101)]
actions = ["Stocks", "Real_Estates", "Commodities", "Cryptocurrencies", "Forex"]

# Define the profit function
def profit(investment):
    return np.floor(2 * np.random.uniform() * investment)

# Define the market environment function
def market_env(state, action):
    budget = int(state)
    next_state = budget

    # Define the reward calculation
    def calculate_reward(budget, investment, expected_profit):
        if (budget - investment) > 0:
            return np.log(budget - investment + expected_profit)
        else:
            return budget - investment + expected_profit
    
    # Determine the profit and next state based on the action
    investment_mapping = {
        "Stocks": 5,
        "Real_Estates": 17,
        "Commodities": 11,
        "Cryptocurrencies": 9,
        "Forex": 7
    }
    
    investment = investment_mapping.get(action, 0)
    expected_profit = profit(investment)
    next_state = budget - investment + expected_profit
    reward = calculate_reward(budget, investment, expected_profit)
    
    return {"NextState": str(next_state), "Reward": reward}

# Sample experience for the dataset
def sample_experience(N, env, states, actions):
    data = []
    for _ in range(N):
        state = np.random.choice(states)
        action = np.random.choice(actions)
        experience = env(state, action)
        data.append([state, action, experience['NextState'], experience['Reward']])
    return pd.DataFrame(data, columns=["State", "Action", "NextState", "Reward"])

# Generate and save training data
training_data = sample_experience(N=3000, env=market_env, states=states, actions=actions)
training_data.to_csv('training_data.csv', index=False)

# Generate and save evaluation data
evaluation_data = pd.DataFrame({"State": [str(i) for i in range(15, 46)]})
evaluation_data.to_csv('evaluation_data.csv', index=False)

print("Data generation complete. Files 'training_data.csv' and 'evaluation_data.csv' have been saved.")
