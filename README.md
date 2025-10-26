# MathIsFun Sokoban - Gymnasium Environment

A complete Gymnasium-compatible environment featuring all 60 Sokoban puzzle levels from [MathsIsFun.com](https://www.mathsisfun.com/games/sokoban.html).

![Sokoban Game](https://img.shields.io/badge/Levels-60-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-orange)

## 🎮 Overview

Sokoban is a classic Japanese puzzle game where you push boxes onto target spots. This implementation provides:
- ✅ All 60 official levels from MathsIsFun.com
- ✅ Beautiful 3D-style graphics with proper shading
- ✅ Full Gymnasium API compatibility
- ✅ Ready for RL training
- ✅ Visual rendering with pygame

## 📦 Installation

### Prerequisites
```bash
conda create -n sokoban python=3.10
conda activate sokoban
```

### Install Dependencies
```bash
pip install gymnasium numpy pygame Pillow
```

## 🚀 Quick Start

### 1. View a Specific Level
```bash
python view_level.py 1      # View Level 1 (easiest)
python view_level.py 30     # View Level 30 (medium)
python view_level.py 60     # View Level 60 (hardest)
```

### 2. Play Through All Levels
```bash
python play_sokoban.py
```

### 3. Use in Your Code
```python
import gymnasium as gym
from gymnasium.envs.registration import register
from mathisfun_sokoban import MathIsFunSokoban

# Register the environment
register(
    id="MathIsFunSokoban-v0",
    entry_point="mathisfun_sokoban:MathIsFunSokoban",
    max_episode_steps=300,
)

# Create and use the environment
env = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    
    if done or truncated:
        obs, info = env.reset()

env.close()
```

### 4. Load a Specific Level
```python
# Reset to a specific level (0-59)
obs, info = env.reset(options={"level_idx": 10})
```

## 🎯 Environment Details

### Action Space
- **Type**: `Discrete(4)`
- **Actions**:
  - `0` - Move Up
  - `1` - Move Right
  - `2` - Move Down
  - `3` - Move Left

### Observation Space
- **Type**: `Box(0, 255, shape=(height, width, 3), dtype=uint8)`
- **Format**: RGB image of the game state
- **Size**: Approximately 440×638 pixels (varies by level)

### Rewards
- `+1.0` - Level solved (all boxes on targets)
- `-0.01` - Per step penalty (encourages efficiency)

### Episode Termination
- **Done**: All boxes placed on target spots
- **Truncated**: 300 steps reached (configurable)

## 🎨 Visual Elements

| Element | Color | Description |
|---------|-------|-------------|
| 🧱 Walls | Dark Gray | Impassable barriers |
| 🟫 Boxes | Brown (3D) | Objects to push onto targets |
| 🟩 Targets | Green Circles | Goal positions for boxes |
| 🔴 Player | Red Circle | Your character |
| ✨ Box on Target | Golden Brown | Box correctly placed |
| 🟦 Background | Gray | Empty space outside level |

## 📁 Project Structure

```
math_is_fun/
├── README.md                      # This file
├── mathisfun_sokoban.py          # Main Gymnasium environment
├── levels_clean.json             # 60 levels in ASCII format
├── levels_mathisfun.json         # Original numeric format
├── view_level.py                 # View specific level
├── play_sokoban.py              # Play through all levels
├── test_env.py                   # Test environment
└── __init__.py                   # Environment registration
```

## 🎓 Training RL Agents

### With Stable-Baselines3
```python
from stable_baselines3 import PPO
from gymnasium.envs.registration import register

register(
    id="MathIsFunSokoban-v0",
    entry_point="mathisfun_sokoban:MathIsFunSokoban",
    max_episode_steps=300,
)

# Create environment
env = gym.make("MathIsFunSokoban-v0")

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test the trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
```

## 🎮 Controls & Features

### Current Features
- ✅ Visual rendering with pygame
- ✅ 60 progressively challenging levels
- ✅ Proper Sokoban game mechanics
- ✅ Episode tracking and statistics
- ✅ Window close detection (click X to exit)

### Keyboard Controls
Currently, the environment uses random actions. To implement keyboard controls:

```python
import pygame

# In your game loop
for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
            action = 0
        elif event.key == pygame.K_RIGHT:
            action = 1
        elif event.key == pygame.K_DOWN:
            action = 2
        elif event.key == pygame.K_LEFT:
            action = 3
```

## 📊 Level Statistics

- **Total Levels**: 60
- **Difficulty**: Progressive (1 = easiest, 60 = hardest)
- **Smallest Level**: 8×8 grid
- **Largest Level**: ~29×27 grid
- **Average Boxes**: 4-8 per level
- **Source**: [MathsIsFun.com/games/sokoban.html](https://www.mathsisfun.com/games/sokoban.html)

## 🐛 Troubleshooting

### Window doesn't appear
```bash
# Test pygame installation
python test_pygame.py

# If that works, test the environment
python test_env.py
```

### Window doesn't close when clicking X
- Updated in latest version - should work now
- Alternatively, press `Ctrl+C` in terminal

### Graphics are too small
- Window size is set to 640px (4x scale)
- Edit `self.scale = 640 // max(max_h, max_w)` in `mathisfun_sokoban.py` to increase

### Import errors
```bash
# Make sure you're in the correct directory
cd /Users/skr3178/Downloads/math_is_fun

# Make sure conda environment is activated
conda activate sokoban
```

## 📝 Level Format

Levels are stored in `levels_clean.json` as ASCII strings:
```
  ##### 
###   # 
#.@$  # 
### $.# 
#.##$ # 
# # . ##
#$ *$$.#
#   .  #
########
```

Where:
- `#` = Wall
- ` ` = Floor
- `.` = Target
- `$` = Box
- `@` = Player
- `*` = Box on Target
- `+` = Player on Target

## 🙏 Credits

- **Original Game**: Created by Hiroyuki Imabayashi (1981)
- **Level Source**: [MathsIsFun.com](https://www.mathsisfun.com/games/sokoban.html)
- **Environment Author**: Created for Gymnasium/RL training
- **License**: Educational use

## 📚 Additional Resources

- [Sokoban Wikipedia](https://en.wikipedia.org/wiki/Sokoban)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Original Game Site](https://www.mathsisfun.com/games/sokoban.html)

## 🔄 Version History

- **v1.0** - Initial release with all 60 levels
- Enhanced graphics with 3D shading
- Proper window close handling
- Full Gymnasium API compatibility

---

**Enjoy solving puzzles! 🎉**

