"""
Example of training with Stable-Baselines3 WITH RENDERING
Make sure to install:
  pip install -e .
  pip install "stable-baselines3[extra]" torch
"""
import __init__  # This registers the environment
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Custom callback to render during training
class RenderCallback(BaseCallback):
    def __init__(self, render_freq=100, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.render_env = None
        
    def _on_training_start(self):
        # Create a separate environment for rendering
        self.render_env = gym.make("MathIsFunSokoban-v0", render_mode="human")
        self.render_obs, _ = self.render_env.reset()
        
    def _on_step(self):
        # Render every N steps
        if self.n_calls % self.render_freq == 0:
            # Take action in render environment
            action, _ = self.model.predict(self.render_obs, deterministic=False)
            self.render_obs, reward, done, truncated, _ = self.render_env.step(int(action))
            self.render_env.render()  # Explicitly call render!
            
            if done or truncated:
                if done:
                    print(f"\n🎉 Level solved at step {self.n_calls}! Reward: {reward}")
                self.render_obs, _ = self.render_env.reset()
        
        return True
    
    def _on_training_end(self):
        if self.render_env:
            self.render_env.close()

# Create training environment (without rendering for speed)
env = gym.make("MathIsFunSokoban-v0")
env = DummyVecEnv([lambda: env])

# IMPORTANT: Use CnnPolicy for image observations, NOT MlpPolicy!
model = PPO(
    "CnnPolicy",  # Use CNN for image inputs
    env, 
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)

# Train with rendering callback
print("Starting training with live rendering...")
print("A separate window will show the agent's progress every 100 steps")
render_callback = RenderCallback(render_freq=100)
model.learn(total_timesteps=100000, callback=render_callback)

# Save the model
model.save("sokoban_ppo")
print("Model saved as 'sokoban_ppo'")

# Test the trained agent
print("\nTesting the trained agent...")
env_test = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env_test.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env_test.step(int(action))
    env_test.render()  # Render each step
    
    if done:
        print(f"Level solved! Reward: {reward}")
        obs, _ = env_test.reset()
    elif truncated:
        print("Level truncated (max steps)")
        obs, _ = env_test.reset()

env_test.close()

