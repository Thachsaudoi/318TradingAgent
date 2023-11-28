import gym
from gym import spaces
import numpy as np

class InvestmentEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(InvestmentEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # Stocks, Real Estates, etc.
        self.observation_space = spaces.Discrete(100)  # Budget 1 to 100
        self.state = 50  # Starting with a budget of 50 (example)
        self.investment_mapping = {
            0: 5,   # Stocks
            1: 17,  # Real Estates
            2: 11,  # Commodities
            3: 9,   # Cryptocurrencies
            4: 7    # Forex
        }

    def step(self, action):
        investment = self.investment_mapping.get(action, 0)
        expected_profit = np.floor(2 * np.random.uniform() * investment)
        budget = self.state
        
        reward = 0
        if budget - investment + expected_profit > 0:
            reward = np.log(budget - investment + expected_profit)
        else:
            reward = budget - investment + expected_profit
        
        self.state = budget - investment + expected_profit
        done = bool(self.state <= 0)
        
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.randint(1, 100)
        return self.state

    def render(self, mode='human', close=False):
        print("")

    def close(self):
        pass
