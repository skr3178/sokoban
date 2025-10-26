# __init__.py
from gymnasium.envs.registration import register
from .mathisfun_sokoban import MathIsFunSokoban

register(
    id="MathIsFunSokoban-v0",
    entry_point="mathisfun_sokoban:MathIsFunSokoban",
    max_episode_steps=300,
)

