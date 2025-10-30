"""
PPO Agent for BingleFixIt Business Gym

Trains a Proximal Policy Optimization agent to manage the business.
Requires: pip install stable-baselines3
"""

import sys
import os



# Fix for gym version compatibility issue with stable-baselines3
import gym
if not hasattr(gym, '__version__'):
    gym.__version__ = '0.26.0'  # Fake version to prevent err
    
from business_gym.business_env import BingleFixItBusinessEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime


class DictToMultiDiscreteWrapper(gym.Wrapper):
    """
    Converts a Dict action space to MultiDiscrete for PPO compatibility.
    
    This wrapper takes an environment with a Dict action space containing
    Discrete sub-spaces and converts it to a single MultiDiscrete action space.
    When actions are taken, it automatically converts them back to Dict format.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Verify original action space is Dict
        if not isinstance(env.action_space, spaces.Dict):
            raise ValueError(f"Environment must have Dict action space, got {type(env.action_space)}")
        
        # Extract the discrete sizes from the dict in a consistent order
        self.action_keys = sorted(env.action_space.spaces.keys())
        action_sizes = []
        
        for key in self.action_keys:
            space = env.action_space[key]
            if not isinstance(space, spaces.Discrete):
                raise ValueError(f"All Dict action spaces must be Discrete, got {type(space)} for key '{key}'")
            action_sizes.append(space.n)
        
        # Create MultiDiscrete action space
        self.action_space = spaces.MultiDiscrete(action_sizes)

    
    def step(self, action):
        """Convert MultiDiscrete action array back to Dict and step"""
        # Convert numpy array to dict
        action_dict = {}
        for i, key in enumerate(self.action_keys):
            action_dict[key] = int(action[i])
        
        # Call original environment with dict action
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        
        # The environment now handles episode info correctly
        # We just pass it through without modification
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Pass through reset"""
        return self.env.reset(**kwargs)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_ratings = []
        self.episode_revenues = []
    
    def _on_step(self) -> bool:
        # Log custom metrics if episode ended
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            
            self.episode_ratings.append(info.get('rating', 0))
            self.episode_revenues.append(info.get('revenue', 0))
            
            # Log to tensorboard
            self.logger.record('business/rating', info.get('rating', 0))
            self.logger.record('business/revenue', info.get('revenue', 0))
            self.logger.record('business/satisfaction', info.get('satisfaction', 0))
            self.logger.record('business/quality', info.get('quality', 0))
        
        return True


def train_ppo_agent(
    total_timesteps=50000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    save_dir='./models',
    log_dir='./logs'
):
    """
    Train PPO agent on the business environment
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
    """
    
    print("\n" + "="*70)
    print("TRAINING PPO AGENT")
    print("BingleFixIt Business Management RL Gym")
    print("="*70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print("="*70 + "\n")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Environment config
    env_config = {
        'max_steps': 50,
        'log_to_db': False  # Disable for training (too many logs)
    }
    
    # Create environment factory with wrapper
    def make_env():
        base_env = BingleFixItBusinessEnv(config=env_config)
        wrapped_env = DictToMultiDiscreteWrapper(base_env)
        return wrapped_env
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=n_envs)
    
    # Create eval environment with wrapper
    eval_env = DictToMultiDiscreteWrapper(
        BingleFixItBusinessEnv(config=env_config)
    )
    
    # Callbacks
    tensorboard_callback = TensorboardCallback()
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{save_dir}/best_model',
        log_path=f'{log_dir}/eval',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    print("Starting training...\n")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[tensorboard_callback, eval_callback],
        progress_bar=True
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f'{save_dir}/ppo_binglefixit_{timestamp}'
    model.save(final_model_path)
    print(f"✓ Model saved to {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model, tensorboard_callback


def evaluate_trained_agent(model_path, num_episodes=10):
    """
    Evaluate a trained agent
    """
    print("\n" + "="*70)
    print("EVALUATING TRAINED AGENT")
    print("="*70 + "\n")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with wrapper
    env_config = {'max_steps': 50, 'log_to_db': False}
    base_env = BingleFixItBusinessEnv(config=env_config)
    env = DictToMultiDiscreteWrapper(base_env)
    
    episode_rewards = []
    final_ratings = []
    final_revenues = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if steps % 10 == 0:
                print(f"  Step {steps}: Rating={info['rating']:.2f}, Revenue=${info['revenue']:,.0f}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        final_ratings.append(info['rating'])
        final_revenues.append(info['revenue'])
        
        print(f"  Final: Reward={episode_reward:.2f}, Rating={info['rating']:.2f}, Revenue=${info['revenue']:,.0f}")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Final Rating: {np.mean(final_ratings):.2f}/5.0")
    print(f"Mean Final Revenue: ${np.mean(final_revenues):,.2f}")
    print("="*70)
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_rating': np.mean(final_ratings),
        'mean_revenue': np.mean(final_revenues),
        'episode_rewards': episode_rewards
    }


def plot_training_metrics(callback, save_path='./results/training_curves.png'):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rating over episodes
    axes[0].plot(callback.episode_ratings)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Final Rating')
    axes[0].set_title('Business Rating Over Training')
    axes[0].grid(True, alpha=0.3)
    
    # Revenue over episodes
    axes[1].plot(callback.episode_revenues)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Final Revenue ($)')
    axes[1].set_title('Revenue Over Training')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Training curves saved to {save_path}")


def main():
    """Main training pipeline"""
    
    # Train agent
    model, callback = train_ppo_agent(
        total_timesteps=50000,
        n_envs=4,
        learning_rate=3e-4
    )
    
    # Plot metrics
    plot_training_metrics(callback)
    
    # Evaluate
    print("\n" + "="*70)
    print("Running final evaluation...")
    model_path = './models/best_model/best_model.zip'
    if os.path.exists(model_path):
        results = evaluate_trained_agent(model_path, num_episodes=10)
        
        # Save results
        os.makedirs('./results', exist_ok=True)
        with open('./results/ppo_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("✓ Results saved to ./results/ppo_evaluation.json")
    else:
        print("model not found, skipping evaluation")
        print("Train for more timesteps to generate best_model checkpoint")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("View training logs: tensorboard --logdir ./logs")



if __name__ == "__main__":
    main()