"""
Improved PPO Agent for BingleFixIt Business Gym

Key improvements:
1. Better hyperparameters for stable learning
2. Proper checkpoint handling
3. Improved reward normalization
4. Better logging
"""

import sys
import os

# Fix for gym version compatibility
import gym
if not hasattr(gym, '__version__'):
    gym.__version__ = '0.26.0'
    
from business_gym.business_env import BingleFixItBusinessEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime


class DictToMultiDiscreteWrapper(gym.Wrapper):
    """
    Converts Dict action space to MultiDiscrete for PPO compatibility
    """
    def __init__(self, env):
        super().__init__(env)
        
        if not isinstance(env.action_space, spaces.Dict):
            raise ValueError(f"Environment must have Dict action space, got {type(env.action_space)}")
        
        self.action_keys = sorted(env.action_space.spaces.keys())
        action_sizes = []
        
        for key in self.action_keys:
            space = env.action_space[key]
            if not isinstance(space, spaces.Discrete):
                raise ValueError(f"All Dict spaces must be Discrete, got {type(space)} for key '{key}'")
            action_sizes.append(space.n)
        
        self.action_space = spaces.MultiDiscrete(action_sizes)
    
    def step(self, action):
        action_dict = {}
        for i, key in enumerate(self.action_keys):
            action_dict[key] = int(action[i])
        
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EnhancedTensorboardCallback(BaseCallback):
    """
    Enhanced callback with better metrics tracking
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_ratings = []
        self.episode_revenues = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log metrics when episode ends
        for idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                info = self.locals.get('infos')[idx]
                
                # Extract episode info
                if 'episode' in info:
                    ep_info = info['episode']
                    self.episode_rewards.append(ep_info['r'])
                    self.episode_lengths.append(ep_info['l'])
                
                # Log business metrics
                if 'rating' in info:
                    self.episode_ratings.append(info['rating'])
                    self.episode_revenues.append(info['revenue'])
                    
                    self.logger.record('business/rating', info['rating'])
                    self.logger.record('business/revenue', info['revenue'])
                    self.logger.record('business/satisfaction', info.get('satisfaction', 0))
                    self.logger.record('business/quality', info.get('quality', 0))
        
        return True


def train_ppo_agent(
    total_timesteps=100000,  # Increased from 20k
    n_envs=4,  # Reduced for stability
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,  # Increased batch size
    n_epochs=10,
    save_dir='./models',
    log_dir='./logs'
):
    """
    Train PPO agent with improved hyperparameters
    """
    
    print("\n" + "="*70)
    print("TRAINING PPO AGENT - IMPROVED VERSION")
    print("BingleFixIt Business Management RL Gym")
    print("="*70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Steps per update: {n_steps * n_envs:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print("="*70 + "\n")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Environment config
    env_config = {
        'max_steps': 50,  # Shorter episodes for faster learning
        'log_to_db': False
    }
    
    # Create environment factory
    def make_env():
        base_env = BingleFixItBusinessEnv(config=env_config)
        wrapped_env = DictToMultiDiscreteWrapper(base_env)
        return wrapped_env
    
    # Create vectorized environment with normalization
    print("Creating training environment...")
    env = make_vec_env(make_env, n_envs=n_envs)
    
    # IMPORTANT: VecNormalize helps stabilize training
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    # Create eval environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during eval
        clip_obs=10.0,
        training=False,
        gamma=0.99
    )
    
    # Callbacks
    tensorboard_callback = EnhancedTensorboardCallback()
    
    # Checkpoint callback - saves model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=f'{save_dir}/checkpoints',
        name_prefix='ppo_checkpoint'
    )
    
    # Eval callback - evaluates and saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{save_dir}/best_model',
        log_path=f'{log_dir}/eval',
        eval_freq=max(5000 // n_envs, 1),  # Eval every 5000 steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create PPO model with improved hyperparameters
    print("Initializing PPO model...")
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
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])  # Smaller network
        ),
        verbose=1
    )
    
    print("\nStarting training...")
    print(f"Updates will occur every {n_steps * n_envs:,} steps")
    print(f"Total updates: {total_timesteps // (n_steps * n_envs)}")
    print("-" * 70 + "\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[tensorboard_callback, checkpoint_callback, eval_callback],
            progress_bar=True,
            log_interval=1  # Log every update
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f'{save_dir}/ppo_binglefixit_{timestamp}'
    model.save(final_model_path)
    
    # Save normalization statistics
    env.save(f'{save_dir}/vec_normalize_{timestamp}.pkl')
    
    print(f"✓ Model saved to {final_model_path}")
    print(f"✓ Normalization stats saved to vec_normalize_{timestamp}.pkl")
    
    # Save training summary
    summary = {
        'total_timesteps': total_timesteps,
        'n_envs': n_envs,
        'learning_rate': learning_rate,
        'final_episodes': len(tensorboard_callback.episode_rewards),
        'mean_episode_reward': float(np.mean(tensorboard_callback.episode_rewards[-10:])) if tensorboard_callback.episode_rewards else 0,
        'mean_episode_length': float(np.mean(tensorboard_callback.episode_lengths[-10:])) if tensorboard_callback.episode_lengths else 0,
        'timestamp': timestamp
    }
    
    with open(f'./results/training_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training summary saved")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model, tensorboard_callback


def evaluate_trained_agent(model_path, vec_normalize_path=None, num_episodes=20):
    """
    Evaluate trained agent with proper normalization
    """
    print("\n" + "="*70)
    print("EVALUATING TRAINED AGENT")
    print("="*70 + "\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment
    env_config = {'max_steps': 100, 'log_to_db': False}
    
    def make_env():
        base_env = BingleFixItBusinessEnv(config=env_config)
        return DictToMultiDiscreteWrapper(base_env)
    
    env = make_vec_env(make_env, n_envs=1)
    
    # Load normalization if available
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Loading normalization from {vec_normalize_path}...")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    episode_rewards = []
    final_ratings = []
    final_revenues = []
    episode_lengths = []
    
    print(f"\nRunning {num_episodes} evaluation episodes...\n")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            steps += 1
            
            if done[0]:
                final_ratings.append(info[0].get('rating', 0))
                final_revenues.append(info[0].get('revenue', 0))
                episode_lengths.append(steps)
                break
        
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Rating={final_ratings[-1]:.2f}, "
              f"Revenue=${final_revenues[-1]:,.0f}, "
              f"Length={steps}")
    
    # Calculate statistics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Episodes Completed:    {num_episodes}")
    print(f"Mean Reward:           {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Final Rating:     {np.mean(final_ratings):.2f}/5.0 ± {np.std(final_ratings):.2f}")
    print(f"Mean Final Revenue:    ${np.mean(final_revenues):,.2f} ± ${np.std(final_revenues):,.2f}")
    print(f"Mean Episode Length:   {np.mean(episode_lengths):.1f} steps")
    print(f"Success Rate:          {sum(1 for l in episode_lengths if l >= 100) / num_episodes:.1%}")
    print("="*70)
    
    env.close()
    
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_rating': float(np.mean(final_ratings)),
        'std_rating': float(np.std(final_ratings)),
        'mean_revenue': float(np.mean(final_revenues)),
        'std_revenue': float(np.std(final_revenues)),
        'mean_length': float(np.mean(episode_lengths)),
        'episode_rewards': [float(r) for r in episode_rewards],
        'final_ratings': [float(r) for r in final_ratings],
        'final_revenues': [float(r) for r in final_revenues]
    }
    
    return results


def plot_training_metrics(callback, save_path='./results/training_curves.png'):
    """Plot training metrics"""
    if not callback.episode_rewards:
        print("⚠ No episode data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    axes[0, 0].plot(callback.episode_rewards)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards Over Training')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(callback.episode_lengths)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Episode Lengths Over Training')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Business rating
    if callback.episode_ratings:
        axes[1, 0].plot(callback.episode_ratings)
        axes[1, 0].axhline(y=3.5, color='r', linestyle='--', alpha=0.3, label='Starting Rating')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Final Rating')
        axes[1, 0].set_title('Business Rating Over Training')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Revenue
    if callback.episode_revenues:
        axes[1, 1].plot(callback.episode_revenues)
        axes[1, 1].axhline(y=10000, color='r', linestyle='--', alpha=0.3, label='Starting Revenue')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Final Revenue ($)')
        axes[1, 1].set_title('Revenue Over Training')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    
    # Train agent with better hyperparameters
    model, callback = train_ppo_agent(
        total_timesteps=100000,  # Train longer
        n_envs=4,
        learning_rate=3e-4,
        batch_size=256
    )
    
    # Plot metrics if available
    if callback.episode_rewards:
        plot_training_metrics(callback)
    
    # Evaluate best model
    print("\n" + "="*70)
    print("Running final evaluation...")
    print("="*70)
    
    best_model_path = './models/best_model/best_model.zip'
    if os.path.exists(best_model_path):
        # Find latest vec_normalize file
        vec_norm_files = [f for f in os.listdir('./models') if f.startswith('vec_normalize_')]
        vec_norm_path = None
        if vec_norm_files:
            latest_vec_norm = sorted(vec_norm_files)[-1]
            vec_norm_path = f'./models/{latest_vec_norm}'
        
        results = evaluate_trained_agent(
            best_model_path,
            vec_normalize_path=vec_norm_path,
            num_episodes=20
        )
        
        # Save results
        with open('./results/ppo_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✓ Results saved to ./results/ppo_evaluation.json")
    else:
        print("⚠ Best model not found - train for more timesteps")
        print(f"   Looking for: {best_model_path}")
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print("\nTo view training logs, run:")
    print("  tensorboard --logdir ./logs")
    print("\nTo compare with random baseline, run:")
    print("  python baseline_random_agent.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()