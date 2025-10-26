import gymnasium as gym
import time
from gymnasium.envs.registration import register
from mathisfun_sokoban import MathIsFunSokoban

# Register the environment
register(
    id="MathIsFunSokoban-v0",
    entry_point="mathisfun_sokoban:MathIsFunSokoban",
    max_episode_steps=300,
)

# Create environment
env = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env.reset()

print("✓ Environment created successfully!")
print(f"✓ Observation shape: {obs.shape}")
print(f"✓ Action space: {env.action_space}")
print("\nRunning random agent for 500 steps...")
print("Press Ctrl+C to stop early\n")

try:
    for step in range(500):
        a = env.action_space.sample()
        obs, r, done, trunc, _ = env.step(a)
        env.render()  # Draw the current state
        if done:
            print(f"Level solved at step {step}!")
            obs, _ = env.reset()
        elif trunc:
            print(f"Level truncated at step {step}")
            obs, _ = env.reset()
        time.sleep(0.03)
except KeyboardInterrupt:
    print("\nStopped by user")

env.close()
print("✓ Test completed!")

