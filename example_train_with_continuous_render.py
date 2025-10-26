"""
Example of training with Stable-Baselines3 with CONTINUOUS rendering
This shows every single step during training (slower but more visual)

Make sure to install:
  pip install -e .
  pip install "stable-baselines3[extra]" torch
"""
import __init__  # This registers the environment
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Custom callback to render every step
class ContinuousRenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.render_env = None
        self.episode_count = 0
        
    def _on_training_start(self):
        # Create a separate environment for rendering
        self.render_env = gym.make("MathIsFunSokoban-v0", render_mode="human")
        self.render_obs, _ = self.render_env.reset()
        print("🎮 Rendering window opened - watch the agent learn!")
        
    def _on_step(self):
        # Render EVERY step (continuous visualization)
        action, _ = self.model.predict(self.render_obs, deterministic=False)
        self.render_obs, reward, done, truncated, _ = self.render_env.step(int(action))
        self.render_env.render()  # Explicitly call render!
        
        if done:
            self.episode_count += 1
            print(f"\n🎉 Episode {self.episode_count}: Level solved! Reward: {reward}")
            self.render_obs, _ = self.render_env.reset()
        elif truncated:
            self.episode_count += 1
            print(f"\n⏱️  Episode {self.episode_count}: Truncated (max steps)")
            self.render_obs, _ = self.render_env.reset()
        
        return True
    
    def _on_training_end(self):
        if self.render_env:
            self.render_env.close()
            print("\n✅ Training complete!")

# Create training environment (without rendering for speed)
env = gym.make("MathIsFunSokoban-v0")
env = DummyVecEnv([lambda: env])

# Create PPO model with CnnPolicy (for image observations)
model = PPO(
    "CnnPolicy",
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

# Train with continuous rendering
print("Starting training with CONTINUOUS rendering...")
print("Watch the pygame window to see every step the agent takes!")
print("Note: This is slower than training without rendering\n")

render_callback = ContinuousRenderCallback()
model.learn(total_timesteps=50000, callback=render_callback)  # Fewer steps for demo

# Save the model
model.save("sokoban_ppo_rendered")
print("\n💾 Model saved as 'sokoban_ppo_rendered'")

# Test the trained agent
print("\n🧪 Testing the trained agent...")
env_test = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env_test.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env_test.step(int(action))
    env_test.render()  # Render each step
    
    if done:
        print(f"✅ Level solved! Reward: {reward}")
        obs, _ = env_test.reset()
    elif truncated:
        print("⏱️  Level truncated (max steps)")
        obs, _ = env_test.reset()

env_test.close()

