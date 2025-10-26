# Installation Guide

## Method 1: Install as Package (Recommended)

This allows you to use `gym.make("MathIsFunSokoban-v0")` from anywhere.

```bash
# Navigate to the project directory
cd /Users/skr3178/Downloads/math_is_fun

# Install in editable mode
pip install -e .
```

Now you can use it from anywhere:

```python
import gymnasium as gym

env = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env.reset()
# ... use the environment
```

## Method 2: Direct Import (Without Installation)

If you don't want to install the package, you can register it manually:

```python
import gymnasium as gym
from gymnasium.envs.registration import register

# Register the environment manually
register(
    id="MathIsFunSokoban-v0",
    entry_point="math_is_fun.mathisfun_sokoban:MathIsFunSokoban",
    max_episode_steps=300,
)

# Now you can create it
env = gym.make("MathIsFunSokoban-v0", render_mode="human")
```

Or import directly:

```python
from math_is_fun.mathisfun_sokoban import MathIsFunSokoban

env = MathIsFunSokoban(render_mode="human")
```

## Installing with Stable-Baselines3

```bash
# First install the package
pip install -e .

# Then install stable-baselines3 with extras
pip install "stable-baselines3[extra]" torch
```

## Verifying Installation

Run the basic example:
```bash
python example_basic.py
```

Or check if the environment is registered:
```bash
python -c "import gymnasium as gym; env = gym.make('MathIsFunSokoban-v0'); print('Success!')"
```

## Troubleshooting

### ModuleNotFoundError: No module named 'math_is_fun'
- Make sure you ran `pip install -e .` from the project directory
- Or use Method 2 (manual registration)

### ImportError related to pygame
```bash
pip install pygame
```

### Environment not found when using gym.make()
- The package must be installed with `pip install -e .`
- Or import the `__init__.py` first: `import math_is_fun` before calling `gym.make()`

