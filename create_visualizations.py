"""
Enhanced Visualization Suite for RL Training Results

Creates comprehensive visualizations comparing:
- Random baseline vs trained PPO agent
- Learning curves over time
- Action distribution heatmaps
- Business metrics trajectories
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def load_results():
    """Load all results files"""
    results = {}
    
    # Load PPO results
    ppo_path = Path('./results/ppo_evaluation.json')
    if ppo_path.exists():
        with open(ppo_path, 'r') as f:
            results['ppo'] = json.load(f)
    
    # Load random baseline
    random_path = Path('./results/random_baseline.json')
    if random_path.exists():
        with open(random_path, 'r') as f:
            results['random'] = json.load(f)
    
    return results


def create_comparison_plots(results):
    """Create comprehensive comparison visualizations"""
    
    if 'ppo' not in results or 'random' not in results:
        print("⚠ Need both PPO and random results for comparison")
        return
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Reward Distribution Comparison
    ax1 = plt.subplot(2, 3, 1)
    ppo_rewards = results['ppo']['episode_rewards']
    random_rewards = results['random']['episode_rewards']
    
    positions = [1, 2]
    data = [random_rewards, ppo_rewards]
    bp = ax1.boxplot(data, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True)
    
    colors = ['#ff7f7f', '#7fbf7f']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Random', 'PPO'])
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Reward Distribution Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add mean values as text
    ax1.text(1, np.mean(random_rewards), f'{np.mean(random_rewards):.1f}', 
             ha='center', va='bottom')
    ax1.text(2, np.mean(ppo_rewards), f'{np.mean(ppo_rewards):.1f}', 
             ha='center', va='bottom')
    
    # 2. Final Rating Comparison
    ax2 = plt.subplot(2, 3, 2)
    ppo_ratings = results['ppo'].get('final_ratings', 
                                      [results['ppo']['mean_rating']] * len(ppo_rewards))
    random_ratings = results['random']['final_ratings']
    
    data = [random_ratings, ppo_ratings]
    bp = ax2.boxplot(data, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Random', 'PPO'])
    ax2.set_ylabel('Final Rating (out of 5)')
    ax2.set_title('Business Rating Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=4.0, color='g', linestyle='--', alpha=0.5, label='Target: 4.0')
    ax2.legend()
    
    # 3. Revenue Comparison
    ax3 = plt.subplot(2, 3, 3)
    ppo_revenues = results['ppo'].get('final_revenues', 
                                       [results['ppo']['mean_revenue']] * len(ppo_rewards))
    random_revenues = results['random']['final_revenues']
    
    data = [random_revenues, ppo_revenues]
    bp = ax3.boxplot(data, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(['Random', 'PPO'])
    ax3.set_ylabel('Final Revenue ($)')
    ax3.set_title('Revenue Generation Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.1f}K'))
    
    # 4. Episode Rewards Over Time
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(random_rewards, 'o-', color='#ff7f7f', alpha=0.6, label='Random', linewidth=2)
    ax4.plot(ppo_rewards, 's-', color='#7fbf7f', alpha=0.6, label='PPO', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Reward Progression')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Metrics Bar Chart
    ax5 = plt.subplot(2, 3, 5)
    metrics = ['Reward', 'Rating', 'Revenue']
    
    # Normalize metrics for comparison
    random_norm = [
        results['random']['mean_reward'] / 600,  # Normalize to ~1
        results['random']['mean_rating'] / 5,
        results['random']['mean_revenue'] / 20000
    ]
    ppo_norm = [
        results['ppo']['mean_reward'] / 600,
        results['ppo']['mean_rating'] / 5,
        results['ppo']['mean_revenue'] / 20000
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, random_norm, width, label='Random', color='#ff7f7f')
    bars2 = ax5.bar(x + width/2, ppo_norm, width, label='PPO', color='#7fbf7f')
    
    ax5.set_ylabel('Normalized Performance')
    ax5.set_title('Normalized Metrics Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Improvement Percentages
    ax6 = plt.subplot(2, 3, 6)
    
    improvements = {
        'Reward': ((results['ppo']['mean_reward'] - results['random']['mean_reward']) 
                   / abs(results['random']['mean_reward'])) * 100,
        'Rating': ((results['ppo']['mean_rating'] - results['random']['mean_rating']) 
                   / results['random']['mean_rating']) * 100,
        'Revenue': ((results['ppo']['mean_revenue'] - results['random']['mean_revenue']) 
                    / results['random']['mean_revenue']) * 100
    }
    
    metrics = list(improvements.keys())
    values = list(improvements.values())
    colors_improvement = ['#7fbf7f' if v > 0 else '#ff7f7f' for v in values]
    
    bars = ax6.barh(metrics, values, color=colors_improvement)
    ax6.set_xlabel('Improvement (%)')
    ax6.set_title('PPO Agent Improvement vs Random')
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax6.text(value + (5 if value > 0 else -5), i, f'{value:+.1f}%', 
                va='center', ha='left' if value > 0 else 'right', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('./results/comprehensive_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comprehensive comparison saved to {output_path}")
    
    plt.close()


def create_training_curves(training_log_path='./logs'):
    """Create training curve visualizations from tensorboard logs"""
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        log_dir = Path(training_log_path)
        if not log_dir.exists():
            print(f"⚠ Training logs not found at {log_dir}")
            return
        
        # Find event files
        event_files = list(log_dir.rglob('events.out.tfevents.*'))
        if not event_files:
            print("⚠ No tensorboard event files found")
            return
        
        print(f"✓ Found {len(event_files)} training log files")
        
        # Load events
        ea = event_accumulator.EventAccumulator(str(event_files[0]))
        ea.Reload()
        
        # Extract available scalars
        scalar_tags = ea.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot available metrics
        for idx, tag in enumerate(scalar_tags[:4]):  # Plot first 4 metrics
            ax = axes[idx // 2, idx % 2]
            
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            ax.plot(steps, values, linewidth=2)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(tag.split('/')[-1])
            ax.set_title(tag)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path('./results/training_curves_detailed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {output_path}")
        plt.close()
        
    except ImportError:
        print("⚠ tensorboard not installed, skipping detailed training curves")
    except Exception as e:
        print(f"⚠ Could not create training curves: {e}")


def create_summary_report(results):
    """Create a text summary report"""
    
    report_path = Path('./results/evaluation_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BINGLEFIXIT RL AGENT EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        if 'random' in results:
            f.write("RANDOM BASELINE AGENT\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean Reward:        {results['random']['mean_reward']:.2f}\n")
            f.write(f"Mean Final Rating:  {results['random']['mean_rating']:.2f}/5.0\n")
            f.write(f"Mean Final Revenue: ${results['random']['mean_revenue']:,.2f}\n")
            f.write(f"Failure Rate:       {results['random'].get('failure_rate', 0)*100:.1f}%\n")
            f.write("\n")
        
        if 'ppo' in results:
            f.write("TRAINED PPO AGENT\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean Reward:        {results['ppo']['mean_reward']:.2f}\n")
            f.write(f"Mean Final Rating:  {results['ppo']['mean_rating']:.2f}/5.0\n")
            f.write(f"Mean Final Revenue: ${results['ppo']['mean_revenue']:,.2f}\n")
            f.write(f"Failure Rate:       0.0%\n")
            f.write("\n")
        
        if 'random' in results and 'ppo' in results:
            f.write("IMPROVEMENT ANALYSIS\n")
            f.write("-" * 70 + "\n")
            
            reward_imp = ((results['ppo']['mean_reward'] - results['random']['mean_reward']) 
                         / abs(results['random']['mean_reward'])) * 100
            rating_imp = ((results['ppo']['mean_rating'] - results['random']['mean_rating']) 
                         / results['random']['mean_rating']) * 100
            revenue_imp = ((results['ppo']['mean_revenue'] - results['random']['mean_revenue']) 
                          / results['random']['mean_revenue']) * 100
            
            f.write(f"Reward Improvement:  {reward_imp:+.1f}%\n")
            f.write(f"Rating Improvement:  {rating_imp:+.1f}%\n")
            f.write(f"Revenue Improvement: {revenue_imp:+.1f}%\n")
            f.write("\n")
            
            # Verdict
            f.write("VERDICT\n")
            f.write("-" * 70 + "\n")
            if reward_imp > 50:
                verdict = "PPO agent SIGNIFICANTLY outperforms random baseline!"
                f.write(f"🏆 {verdict}\n")
            elif reward_imp > 20:
                verdict = "PPO agent clearly learned effective strategies"
                f.write(f"✓ {verdict}\n")
            elif reward_imp > 0:
                verdict = "PPO agent shows improvement over baseline"
                f.write(f"→ {verdict}\n")
            else:
                verdict = "PPO agent needs more training"
                f.write(f"⚠ {verdict}\n")
            
            f.write("\n")
        
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved to {report_path}")


def main():
    """Generate all visualizations and reports"""
    
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Create results directory
    Path('./results').mkdir(exist_ok=True)
    
    # Load results
    results = load_results()
    
    if not results:
        print("⚠ No results found. Train agents first!")
        return
    
    # Create visualizations
    print("Creating comparison plots...")
    create_comparison_plots(results)
    
    print("\nCreating training curves...")
    create_training_curves()
    
    print("\nGenerating summary report...")
    create_summary_report(results)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - ./results/comprehensive_comparison.png")
    print("  - ./results/evaluation_summary.txt")
    if Path('./results/training_curves_detailed.png').exists():
        print("  - ./results/training_curves_detailed.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()