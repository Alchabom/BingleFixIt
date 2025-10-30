"""
Baseline Random Agent for Comparison

Runs a completely random agent to establish baseline performance.
This shows what the business would look like without intelligent management.
"""

import sys
import numpy as np
import json
from business_gym.business_env import BingleFixItBusinessEnv
from ppo_agent import DictToMultiDiscreteWrapper


def evaluate_random_agent(num_episodes=20):
    """
    Evaluate random decision-making baseline
    
    This agent makes completely random choices for:
    - Review responses (random 0-3)
    - Pricing changes (random 0-2)
    - Quality investments (random 0-2)
    - Promotions (random 0-1)
    
    Used to establish baseline performance for comparison.
    """
    
    print("\n" + "="*70)
    print("BASELINE EVALUATION: RANDOM AGENT")
    print("="*70)
    print("Testing completely random business decisions...")
    print(f"Episodes: {num_episodes}")
    print("="*70 + "\n")
    
    # Create environment
    env_config = {'max_steps': 100, 'log_to_db': False}
    base_env = BingleFixItBusinessEnv(config=env_config)
    env = DictToMultiDiscreteWrapper(base_env)
    
    episode_rewards = []
    final_ratings = []
    final_revenues = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Initial: Rating={info['rating']:.2f}, Revenue=${info['revenue']:,.0f}")
        
        while True:
            # Completely random action selection
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if steps % 20 == 0:
                print(f"  Step {steps}: Rating={info['rating']:.2f}, Revenue=${info['revenue']:,.0f}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        final_ratings.append(info['rating'])
        final_revenues.append(info['revenue'])
        episode_lengths.append(steps)
        
        termination_reason = "Business Failed" if terminated else "Max Steps"
        print(f"  Final ({termination_reason}): Reward={episode_reward:.2f}, Rating={info['rating']:.2f}, Revenue=${info['revenue']:,.0f}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_rating = np.mean(final_ratings)
    std_rating = np.std(final_ratings)
    mean_revenue = np.mean(final_revenues)
    std_revenue = np.std(final_revenues)
    mean_length = np.mean(episode_lengths)
    
    # Summary
    print("\n" + "="*70)
    print("RANDOM AGENT BASELINE RESULTS")
    print("="*70)
    print(f"Mean Episode Reward:   {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Final Rating:     {mean_rating:.2f}/5.0 ± {std_rating:.2f}")
    print(f"Mean Final Revenue:    ${mean_revenue:,.2f} ± ${std_revenue:,.2f}")
    print(f"Mean Episode Length:   {mean_length:.1f} steps")
    print(f"Business Failure Rate: {sum(1 for l in episode_lengths if l < 100)/num_episodes:.1%}")
    print("="*70)
    
    # Save results
    results = {
        'agent_type': 'random',
        'num_episodes': num_episodes,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_rating': float(mean_rating),
        'std_rating': float(std_rating),
        'mean_revenue': float(mean_revenue),
        'std_revenue': float(std_revenue),
        'mean_episode_length': float(mean_length),
        'failure_rate': float(sum(1 for l in episode_lengths if l < 100)/num_episodes),
        'episode_rewards': episode_rewards,
        'final_ratings': final_ratings,
        'final_revenues': final_revenues,
        'episode_lengths': episode_lengths
    }
    
    import os
    os.makedirs('./results', exist_ok=True)
    with open('./results/random_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to ./results/random_baseline.json")
    
    env.close()
    return results


def compare_with_ppo(ppo_results_file='./results/ppo_evaluation.json'):
    """
    Compare random baseline with trained PPO agent
    """
    import os
    
    if not os.path.exists(ppo_results_file):
        print(f"\n⚠ PPO results not found at {ppo_results_file}")
        print("Train PPO agent first to enable comparison")
        return
    
    # Load random baseline
    with open('./results/random_baseline.json', 'r') as f:
        random_results = json.load(f)
    
    # Load PPO results
    with open(ppo_results_file, 'r') as f:
        ppo_results = json.load(f)
    
    print("\n" + "="*70)
    print("AGENT COMPARISON: RANDOM vs PPO")
    print("="*70)
    
    # Reward comparison
    random_reward = random_results['mean_reward']
    ppo_reward = ppo_results['mean_reward']
    reward_improvement = ((ppo_reward - random_reward) / abs(random_reward)) * 100
    
    print(f"\n📊 EPISODE REWARD:")
    print(f"  Random Agent:  {random_reward:.2f}")
    print(f"  PPO Agent:     {ppo_reward:.2f}")
    print(f"  Improvement:   {reward_improvement:+.1f}%")
    
    # Rating comparison
    random_rating = random_results['mean_rating']
    ppo_rating = ppo_results['mean_rating']
    rating_improvement = ((ppo_rating - random_rating) / random_rating) * 100
    
    print(f"\n⭐ FINAL RATING:")
    print(f"  Random Agent:  {random_rating:.2f}/5.0")
    print(f"  PPO Agent:     {ppo_rating:.2f}/5.0")
    print(f"  Improvement:   {rating_improvement:+.1f}%")
    
    # Revenue comparison
    random_revenue = random_results['mean_revenue']
    ppo_revenue = ppo_results['mean_revenue']
    revenue_improvement = ((ppo_revenue - random_revenue) / random_revenue) * 100
    
    print(f"\n💰 FINAL REVENUE:")
    print(f"  Random Agent:  ${random_revenue:,.2f}")
    print(f"  PPO Agent:     ${ppo_revenue:,.2f}")
    print(f"  Improvement:   {revenue_improvement:+.1f}%")
    
    # Survival comparison
    if 'failure_rate' in random_results:
        print(f"\n💀 BUSINESS FAILURE RATE:")
        print(f"  Random Agent:  {random_results['failure_rate']:.1%}")
        print(f"  PPO Agent:     0.0%")
    
    print("\n" + "="*70)
    
    # Determine winner
    if reward_improvement > 50:
        print("🏆 VERDICT: PPO agent SIGNIFICANTLY outperforms random baseline!")
    elif reward_improvement > 20:
        print("✓ VERDICT: PPO agent clearly learned effective strategies")
    elif reward_improvement > 0:
        print("→ VERDICT: PPO agent shows some improvement over baseline")
    else:
        print("⚠ VERDICT: PPO agent needs more training")
    
    print("="*70)


if __name__ == "__main__":
    # Run random baseline
    random_results = evaluate_random_agent(num_episodes=20)
    
    # Compare with PPO if available
    compare_with_ppo()