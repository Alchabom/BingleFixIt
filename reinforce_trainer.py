"""
REINFORCE with comprehensive visualization for demonstrating RL learning.
Shows: reward curves, review quality evolution, exploration metrics, and A/B comparisons.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from review_env import ReviewEnvironment
import re
from collections import deque
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
import json

class VisualizingREINFORCEAgent:
    """REINFORCE agent with comprehensive metrics tracking for demos."""
    
    def __init__(self, model_name='model', learning_rate=3e-6, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        self.gamma = 0.95
        self.entropy_coef = 0.02
        self.max_grad_norm = 0.5
        
        # RL storage
        self.episode_generation_data = []
        self.episode_rewards = []
        self.reward_baseline = deque(maxlen=100)
        
        # === METRICS FOR VISUALIZATION ===
        self.metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'avg_rewards': [],
            'review_lengths': [],
            'review_diversity': [],  # Unique word ratio
            'sentiment_accuracy': [],  # How well sentiment matches rating
            'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'early_reviews': [],  # First 5 episodes
            'late_reviews': [],   # Last 5 episodes
            'exploration_rate': [],  # Measure of randomness
            'avg_rating_over_time': []
        }
        
        self.episode_count = 0
    
    def generate_review(self, prompt, max_new_tokens=50, temperature=0.95, top_p=0.95):
        """Generate review and store data for gradient computation."""
        self.model.eval()
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=400,
            padding=True
        ).to(self.device)
        
        input_len = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        generated_sequence = outputs.sequences[0]
        generated_tokens = generated_sequence[input_len:]
        
        # Calculate entropy for exploration metric
        if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
            entropy = self._calculate_entropy(outputs.scores)
        else:
            entropy = 0.0
        
        # Decode
        full_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        review_text = full_text[len(prompt_text):].strip()
        
        # Store the full sequence for later gradient computation
        generation_data = {
            'input_ids': inputs['input_ids'].cpu(),
            'attention_mask': inputs['attention_mask'].cpu(),
            'generated_tokens': generated_tokens.cpu(),
            'full_sequence': generated_sequence.cpu(),
            'entropy': entropy
        }
        
        return review_text, generation_data
    
    def _calculate_entropy(self, scores):
        """Calculate average entropy across generation for exploration metric."""
        entropies = []
        for score in scores[:10]:  # First 10 tokens
            probs = F.softmax(score[0], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            entropies.append(entropy.item())
        return np.mean(entropies) if entropies else 0.0
    
    def store_outcome(self, generation_data, reward):
        """Store outcome for this step."""
        self.episode_generation_data.append(generation_data)
        self.episode_rewards.append(reward)
    
    def compute_log_prob(self, generation_data):
        """Recompute log probability with gradients enabled."""
        self.model.train()
        
        full_sequence = generation_data['full_sequence'].to(self.device)
        attention_mask = torch.ones_like(full_sequence).to(self.device)
        
        outputs = self.model(
            input_ids=full_sequence.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )
        
        logits = outputs.logits[0]
        input_len = generation_data['input_ids'].shape[1]
        generated_tokens = generation_data['generated_tokens'].to(self.device)
        
        log_probs = []
        for i, token_id in enumerate(generated_tokens):
            token_logits = logits[input_len + i - 1]
            token_log_probs = F.log_softmax(token_logits, dim=-1)
            log_prob = token_log_probs[token_id]
            log_probs.append(log_prob)
        
        total_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0, device=self.device)
        
        return total_log_prob
    
    def update(self):
        """Update with proper gradient handling and metrics tracking."""
        if len(self.episode_rewards) == 0:
            return {'loss': 0.0, 'avg_return': 0.0}
        
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Baseline subtraction
        baseline = np.mean(self.reward_baseline) if len(self.reward_baseline) > 0 else 0.0
        advantages = returns - baseline
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Recompute log probs WITH gradients
        policy_losses = []
        for generation_data, advantage in zip(self.episode_generation_data, advantages):
            log_prob = self.compute_log_prob(generation_data)
            policy_losses.append(-log_prob * advantage)
        
        loss = torch.stack(policy_losses).mean()
        
        # Check for numerical issues
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN or Inf detected in loss. Skipping update.")
            self.episode_generation_data = []
            self.episode_rewards = []
            return {'loss': 0.0, 'avg_return': returns.mean().item()}
        
        # Gradient update with clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update baseline
        for ret in returns:
            self.reward_baseline.append(ret.item())
        
        loss_val = loss.item()
        avg_return = returns.mean().item()
        
        # === TRACK METRICS ===
        self.metrics['episode_losses'].append(loss_val)
        
        # Clear buffers
        self.episode_generation_data = []
        self.episode_rewards = []
        
        return {'loss': loss_val, 'avg_return': avg_return}
    
    def track_review_metrics(self, review_text, rating, avg_rating):
        """Track metrics about generated reviews."""
        words = review_text.split()
        
        # Length
        self.metrics['review_lengths'].append(len(words))
        
        # Diversity (unique word ratio)
        diversity = len(set(words)) / len(words) if len(words) > 0 else 0
        self.metrics['review_diversity'].append(diversity)
        
        # Rating distribution
        self.metrics['rating_distribution'][rating] += 1
        
        # Sentiment accuracy (how close generated rating is to target)
        sentiment_error = abs(rating - avg_rating)
        accuracy = max(0, 5 - sentiment_error) / 5  # 0 to 1 scale
        self.metrics['sentiment_accuracy'].append(accuracy)
        
        # Store examples for A/B comparison
        if self.episode_count < 5:
            self.metrics['early_reviews'].append({
                'episode': self.episode_count,
                'text': review_text,
                'rating': rating
            })
        elif self.episode_count >= 95:  # Last 5 episodes
            self.metrics['late_reviews'].append({
                'episode': self.episode_count,
                'text': review_text,
                'rating': rating
            })
    
    def track_episode_metrics(self, episode_reward, avg_rating, avg_entropy):
        """Track episode-level metrics."""
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['avg_rating_over_time'].append(float(avg_rating))
        self.metrics['exploration_rate'].append(float(avg_entropy))
        
        # Calculate rolling average
        window = 20
        if len(self.metrics['episode_rewards']) >= window:
            avg = np.mean(self.metrics['episode_rewards'][-window:])
            self.metrics['avg_rewards'].append(float(avg))
        else:
            avg = np.mean(self.metrics['episode_rewards'])
            self.metrics['avg_rewards'].append(float(avg))
        
        self.episode_count += 1
    
    def save(self, path):
        """Save model and metrics."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Convert metrics to JSON-serializable format
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save metrics with conversion
        metrics_path = os.path.join(path, 'training_metrics.json')
        serializable_metrics = convert_to_serializable(self.metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"✓ Model and metrics saved to {path}")
    
    def plot_training_progress(self, save_path='./training_plots'):
        """Generate comprehensive training visualizations."""
        os.makedirs(save_path, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Episode Rewards
        ax1 = plt.subplot(3, 3, 1)
        episodes = range(1, len(self.metrics['episode_rewards']) + 1)
        ax1.plot(episodes, self.metrics['episode_rewards'], alpha=0.3, label='Episode Reward')
        if len(self.metrics['avg_rewards']) > 0:
            ax1.plot(episodes, self.metrics['avg_rewards'], linewidth=2, label='Rolling Avg (20)', color='red')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Learning Curve: Rewards Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(episodes, self.metrics['episode_losses'], color='orange')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Policy Loss')
        ax2.set_title('Policy Loss Over Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. Review Length
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(self.metrics['review_lengths'], alpha=0.6, color='green')
        ax3.axhline(y=np.mean(self.metrics['review_lengths']), color='red', linestyle='--', label='Mean')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Word Count')
        ax3.set_title('Review Length Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Diversity (Exploration)
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(self.metrics['review_diversity'], alpha=0.6, color='purple')
        ax4.axhline(y=np.mean(self.metrics['review_diversity']), color='red', linestyle='--', label='Mean')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Unique Word Ratio')
        ax4.set_title('Review Diversity (Exploration)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Sentiment Accuracy
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(self.metrics['sentiment_accuracy'], alpha=0.6, color='blue')
        ax5.axhline(y=np.mean(self.metrics['sentiment_accuracy']), color='red', linestyle='--', label='Mean')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Accuracy (0-1)')
        ax5.set_title('Sentiment Matching Accuracy')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Rating Distribution
        ax6 = plt.subplot(3, 3, 6)
        ratings = list(self.metrics['rating_distribution'].keys())
        counts = list(self.metrics['rating_distribution'].values())
        ax6.bar(ratings, counts, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
        ax6.set_xlabel('Star Rating')
        ax6.set_ylabel('Count')
        ax6.set_title('Generated Review Rating Distribution')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Average Rating Over Time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(episodes, self.metrics['avg_rating_over_time'], color='teal', linewidth=2)
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Average Rating')
        ax7.set_title('Business Rating Evolution')
        ax7.set_ylim([1, 5])
        ax7.grid(True, alpha=0.3)
        
        # 8. Exploration Rate (Entropy)
        ax8 = plt.subplot(3, 3, 8)
        if len(self.metrics['exploration_rate']) > 0:
            ax8.plot(episodes, self.metrics['exploration_rate'], color='brown')
            ax8.set_xlabel('Episode')
            ax8.set_ylabel('Avg Entropy')
            ax8.set_title('Exploration Rate (Policy Entropy)')
            ax8.grid(True, alpha=0.3)
        
        # 9. Text box with summary stats
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_stats = f"""
        TRAINING SUMMARY
        ================
        Total Episodes: {len(self.metrics['episode_rewards'])}
        
        Final Avg Reward: {self.metrics['avg_rewards'][-1]:.2f}
        Best Episode Reward: {max(self.metrics['episode_rewards']):.2f}
        
        Avg Review Length: {np.mean(self.metrics['review_lengths']):.1f} words
        Avg Diversity: {np.mean(self.metrics['review_diversity']):.2f}
        Avg Sentiment Acc: {np.mean(self.metrics['sentiment_accuracy']):.2f}
        
        Final Business Rating: {self.metrics['avg_rating_over_time'][-1]:.2f}
        """
        
        ax9.text(0.1, 0.5, summary_stats, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_overview.png'), dpi=300, bbox_inches='tight')
        print(f"\n✓ Training overview saved to {save_path}/training_overview.png")
        
        # === CREATE A/B COMPARISON FIGURE ===
        self._plot_ab_comparison(save_path)
        
        plt.close('all')
    
    def _plot_ab_comparison(self, save_path):
        """Create A/B comparison of early vs late reviews."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Early reviews
        ax1.axis('off')
        ax1.set_title('EARLY REVIEWS (Episodes 1-5)\n[Before Learning]', fontsize=14, fontweight='bold')
        
        early_text = "\n\n".join([
            f"Ep {r['episode']+1} | {r['rating']}⭐: {r['text'][:100]}..."
            for r in self.metrics['early_reviews'][:10]
        ])
        
        ax1.text(0.05, 0.95, early_text, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                wrap=True)
        
        # Late reviews
        ax2.axis('off')
        ax2.set_title('LATE REVIEWS (Episodes 96-100)\n[After Learning]', fontsize=14, fontweight='bold')
        
        late_text = "\n\n".join([
            f"Ep {r['episode']+1} | {r['rating']}⭐: {r['text'][:100]}..."
            for r in self.metrics['late_reviews'][:10]
        ])
        
        ax2.text(0.05, 0.95, late_text, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                wrap=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'ab_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ A/B comparison saved to {save_path}/ab_comparison.png")


# === HELPER FUNCTIONS ===

def better_sentiment_analysis(text: str) -> int:
    """Improved sentiment analysis for tech repair reviews."""
    strong_pos = ['excellent', 'outstanding', 'amazing', 'fantastic', 'great', 'wonderful', 'perfect', 
                  'fixed', 'repaired', 'working', 'resolved', 'solved']
    pos_words = ['good', 'nice', 'friendly', 'fast', 'professional', 'helpful', 'quality', 'satisfied', 
                 'quick', 'reliable', 'skilled', 'expert', 'recommend', 'phone', 'screen', 'battery',
                 'laptop', 'computer', 'repair', 'service', 'technician', 'efficient']
    
    strong_neg = ['terrible', 'horrible', 'awful', 'worst', 'disaster', 'nightmare', 'useless', 
                  'broken', 'damaged', 'scam', 'incompetent']
    neg_words = ['bad', 'poor', 'slow', 'unprofessional', 'disappointing', 'issue', 'problem', 
                 'frustrated', 'waste', 'avoid', 'regret', 'overpriced', 'expensive', 'waiting',
                 'unresponsive', 'delayed', 'unhelpful']
    
    text_lower = text.lower()
    text_clean = re.sub(r'[^a-z0-9\s]', ' ', text_lower)
    words = text_clean.split()
    
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return 1
    
    short_words = [w for w in words if len(w) <= 2]
    if len(words) > 5 and len(short_words) / len(words) > 0.5:
        return 1
    
    strong_pos_count = sum(1 for w in words if w in strong_pos)
    pos_count = sum(1 for w in words if w in pos_words)
    strong_neg_count = sum(1 for w in words if w in strong_neg)
    neg_count = sum(1 for w in words if w in neg_words)
    
    score = (strong_pos_count * 2) + pos_count - neg_count - (strong_neg_count * 2)
    
    if score <= -3:
        return 1
    elif score <= -1:
        return 2
    elif score <= 1:
        return 3
    elif score <= 3:
        return 4
    else:
        return 5


def validate_review(text: str) -> bool:
    """Check if review is valid (not spam/gibberish)."""
    words = text.split()
    
    if len(words) < 5 or len(words) > 150:
        return False
    
    if len(set(words)) / len(words) < 0.4:
        return False
    
    number_pattern = re.findall(r'\d+', text)
    if len(number_pattern) > 5:
        return False
    
    special_chars = re.findall(r'[^a-zA-Z0-9\s\.,!?\'\-]', text)
    if len(special_chars) > len(text) * 0.2:
        return False
    
    return True


def clean_generated_review(text: str) -> str:
    """Clean up generated review text."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return ""
    
    cleaned = '. '.join(sentences[:2])
    if cleaned and not cleaned.endswith('.'):
        cleaned += '.'
    
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    
    return cleaned.strip()


def build_prompt_from_observation(observation, tone_description):
    """Build prompt using ONLY data from the environment observation."""
    reviews_str = observation['reviews'] or "No reviews yet."
    avg_rating = observation['avg_rating'][0]
    
    if len(reviews_str) > 800:
        reviews_str = reviews_str[-800:]
    
    prompt = f"""BingleFixIt - Phone & Computer Repair Reviews
Current average rating: {avg_rating:.1f}/5 stars

Recent customer reviews:
{reviews_str}

Write a {tone_description} review about your repair experience at BingleFixIt (2-3 sentences):"""
    
    return prompt


def main():
    load_dotenv()
    
    # Load settings from .env
    model_name = os.getenv('GEMINI_MODEL', 'distilgpt2')
    max_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', 50))
    
    print(f"--- Using model: {model_name} ---")
    print(f"--- Using max_output_tokens: {max_tokens} ---")

    env_config = {
        'model_name': model_name,
        'max_length': 512,
        'max_steps_per_episode': 5
    }
    env = ReviewEnvironment(env_config)
    
    agent = VisualizingREINFORCEAgent(model_name=model_name, learning_rate=3e-6)
    
    num_episodes = 100
    
    print("\n" + "="*70)
    print("REINFORCEMENT LEARNING DEMONSTRATION")
    print("Training Agent to Generate Contextual Reviews")
    print("="*70 + "\n")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_entropies = []
        
        # Get avg_rating once for the whole episode
        avg_rating = observation['avg_rating'][0]
        
        for step in range(env_config['max_steps_per_episode']):
            
            # Determine tone based on the episode's starting rating
            if avg_rating < 2.5:
                tone = "critical 1-2 star"
            elif avg_rating < 3.5:
                tone = "mixed 3 star"
            elif avg_rating < 4.5:
                tone = "positive 4 star"
            else:
                tone = "excellent 5 star"
            
            # Build prompt
            prompt = build_prompt_from_observation(observation, tone)
            
            # --- START: Corrected Training Logic ---
            is_fallback = False

            # 1. Generate review using the agent
            review_text, generation_data = agent.generate_review(
                prompt, 
                max_new_tokens=max_tokens,
                temperature=0.95, 
                top_p=0.95
            )
            
            # 2. Store entropy for metrics
            episode_entropies.append(generation_data['entropy'])
            
            # 3. Clean and validate
            review_text = clean_generated_review(review_text)
            
            if not validate_review(review_text):
                is_fallback = True
                # Use fallback text if generation is invalid/gibberish
                if avg_rating < 3.0:
                    review_text = "Disappointing experience. The repair took longer than expected and communication was poor."
                elif avg_rating < 4.0:
                    review_text = "Service was acceptable. My device was repaired but nothing exceptional."
                else:
                    review_text = "Good service overall. My phone was repaired professionally and works well."
            
            # 4. Get sentiment-based rating
            rating_1_5 = better_sentiment_analysis(review_text)
            rating_action = rating_1_5 - 1
            
            # 5. Track metrics for the review (either generated or fallback)
            agent.track_review_metrics(review_text, rating_1_5, avg_rating)
            
            # 6. Take action in the environment
            action = {'text': review_text, 'rating': int(rating_action)}
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 7. Accumulate episode reward
            episode_reward += reward
            
            # 8. *** CRITICAL ***
            #    Only store the outcome for training if it was NOT a fallback.
            if not is_fallback:
                agent.store_outcome(generation_data, reward)
            # --- END: Corrected Training Logic ---
            
            # Display
            stars = "⭐" * rating_1_5
            print(f"  Step {step+1}: {stars} ({rating_1_5}) | R: {reward:+6.2f} | {review_text[:70]}...")
            
            if terminated or truncated:
                break
        
        # Update policy
        update_info = agent.update()
        
        # Track episode metrics
        avg_entropy = np.mean(episode_entropies) if episode_entropies else 0
        agent.track_episode_metrics(episode_reward, avg_rating, avg_entropy)
        
        # Display episode summary
        avg_reward_20 = agent.metrics['avg_rewards'][-1]
        
        print(f"\n{'='*70}")
        print(f"Episode {episode+1:3d} | Reward: {episode_reward:7.2f} | Avg(20): {avg_reward_20:7.2f} | Loss: {update_info['loss']:8.4f}")
        print(f"{'='*70}\n")
        
        # Save checkpoints with visualizations
        if (episode + 1) % 25 == 0:
            checkpoint_path = f'./checkpoints/episode_{episode+1}'
            agent.save(checkpoint_path)
            agent.plot_training_progress(f'./training_plots/episode_{episode+1}')
            print(f"✓ Checkpoint and plots saved at episode {episode+1}\n")
    
    print("\n🎉 Training complete!")
    
    # Final save and comprehensive visualization
    agent.save('./trained_review_agent')
    agent.plot_training_progress('./training_plots/final')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Check ./training_plots/ for visualizations!")
    print("="*70)
    
if __name__ == "__main__":
    main()