from gym.envs.registration import register

register(
    id='InvestmentEnv-v0',
    entry_point='openaiTradingBot.envs.environment:InvestmentEnv',  # Note the '.environment' part
)
