import gym
from env.attacker_env import AttackerEnv

env = AttackerEnv()
obs = env.reset()
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += rew
print("Reward total (aleatorio):", total_reward)