#!/usr/bin/env python3
"""
Example script demonstrating video logging functionality for RL training.

This script shows how to configure and use wandb video logging to monitor
agent behavior during training.
"""

import os
from pathlib import Path

import ray
from ray.tune.registry import register_env

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent))

# Import project modules
from src.agents.ppo import get_ppo_config


def env_creator(env_config=None):
    """Create an environment instance."""
    # Import here to avoid circular imports
    training_mode = env_config.get("training_execution_mode", "CTDE")
    
    if training_mode == "CTE":
        from src.environments.reference_model_single_agent import ReferenceModel
    else:
        from src.environments.reference_model_multi_agent import ReferenceModel
    
    return ReferenceModel(env_config)


def main():
    """Run training with video logging enabled."""
    # Initialize Ray
    ray.init()
    
    # Environment configuration
    ENV_NAME = "ReferenceModel-3-1"
    
    env_config = {
        "env_name": ENV_NAME,
        "seed": None,
        "deterministic": False,
        "num_agents": 4,
        "steps_per_episode": 100,
        "sensor_range": 2,
        "training_execution_mode": "CTDE",
        "render_env": False,  # Set to False since we're using video logging
        "video_logging": {
            "enabled": True,  # Enable video logging
            "frequency": 10,  # Record videos every 10 iterations (for demo)
            "max_episodes_per_iteration": 1,  # Record 1 episode per iteration
            "fps": 5,  # 5 frames per second
            "max_frames_per_episode": 150,  # Max 150 frames per episode
        },
    }
    
    # Register environment
    register_env(ENV_NAME, env_creator)
    
    # Get PPO configuration with video logging
    config = get_ppo_config(ENV_NAME, env_config=env_config)
    
    # Build and train the algorithm
    algo = config.build()
    
    print("Starting training with video logging...")
    print(f"Video logging enabled: {env_config['video_logging']['enabled']}")
    print(f"Video frequency: every {env_config['video_logging']['frequency']} iterations")
    
    try:
        # Train for multiple iterations to see video logging in action
        for i in range(25):  # This will trigger video logging at iterations 10 and 20
            result = algo.train()
            
            iteration = result.get("training_iteration", i + 1)
            reward_mean = result.get("env_runners/episode_reward_mean", "N/A")
            
            print(f"Iteration {iteration}: episode_reward_mean = {reward_mean}")
            
            # Log when videos should be recorded
            if iteration % env_config["video_logging"]["frequency"] == 0:
                print(f"  📹 Video recorded and uploaded to wandb for iteration {iteration}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Clean up
        algo.stop()
        ray.shutdown()
        print("Training completed and resources cleaned up")


if __name__ == "__main__":
    # Set wandb project for this example
    os.environ.setdefault('WANDB_PROJECT', 'dl-reference-models-demo')
    
    main()