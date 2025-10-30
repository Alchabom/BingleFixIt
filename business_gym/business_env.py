"""
BingleFixIt Business Management RL Environment

A Gymnasium-compatible environment where agents learn to manage a repair business
by responding to reviews, adjusting prices, and making quality investments.

Author: Omar Alchab
"""

import gymnasium as gym
import numpy as np
import mysql.connector
from typing import Dict, Any, Tuple, Optional, List
import random
from datetime import datetime
import json


class BingleFixItBusinessEnv(gym.Env):
    """
    Reinforcement Learning Environment for Business Management
    
    The agent manages a repair business by making strategic decisions:
    - Responding to customer reviews (builds reputation)
    - Adjusting pricing (affects revenue and customer volume)
    - Investing in service quality (improves future ratings)
    - Running marketing campaigns (increases visibility)
    
    Observation Space:
        - avg_rating: Current business rating (1.0-5.0)
        - revenue: Total accumulated revenue
        - satisfaction: Customer satisfaction score (0-1)
        - service_quality: Current quality level (0-1)
        - price: Current service price ($)
        - pending_reviews: Number of reviews awaiting response
        - recent_reviews_text: Text of recent reviews
    
    Action Space:
        - response_type: How to respond to reviews (0-3)
            0: Ignore
            1: Apologetic (for negative reviews)
            2: Professional/Neutral
            3: Grateful (for positive reviews)
        - pricing_change: Price adjustment (0-2)
            0: Decrease 10%
            1: Maintain
            2: Increase 10%
        - quality_investment: Quality improvements (0-2)
            0: None
            1: Moderate ($500)
            2: High ($1500)
        - run_promotion: Marketing campaign (0-1)
            0: No
            1: Yes ($300)
    
    Reward:
        Composite reward balancing:
        - Appropriate review responses
        - Revenue generation
        - Customer satisfaction
        - Long-term business health
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        
        # Database configuration
        self.db_config = {
            'host': self.config.get('db_host', 'localhost'),
            'port': self.config.get('db_port', 3307),
            'user': self.config.get('db_user', 'ai_agent_user'),
            'password': self.config.get('db_password', 'frince101'),
            'database': self.config.get('db_name', 'mobile_repair')
        }
        
        # Business state variables
        self.current_rating = 3.5
        self.revenue = 10000.0
        self.customer_satisfaction = 0.7
        self.service_quality = 0.7
        self.base_price = 50.0
        self.current_price = 50.0
        self.days_elapsed = 0
        self.current_step = 0
        self.reviews_pending_response: List[Dict] = []
        self.promotion_active = False
        
        # Episode tracking for stable-baselines3
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # Episode configuration
        self.max_steps = self.config.get('max_steps', 100)
        self.episode_num = 0
        
        # Define observation space
        self.observation_space = gym.spaces.Dict({
            'avg_rating': gym.spaces.Box(low=1.0, high=5.0, shape=(1,), dtype=np.float32),
            'revenue': gym.spaces.Box(low=0, high=100000, shape=(1,), dtype=np.float32),
            'satisfaction': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'service_quality': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'price': gym.spaces.Box(low=10, high=200, shape=(1,), dtype=np.float32),
            'pending_reviews': gym.spaces.Box(low=0, high=50, shape=(1,), dtype=np.float32),
        })
        
        # Define action space
        self.action_space = gym.spaces.Dict({
            'response_type': gym.spaces.Discrete(4),
            'pricing_change': gym.spaces.Discrete(3),
            'quality_investment': gym.spaces.Discrete(3),
            'run_promotion': gym.spaces.Discrete(2)
        })
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.episode_num += 1
        self.current_step = 0
        self.days_elapsed = 0
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # Reset business state
        self.current_rating = 3.5
        self.revenue = 10000.0
        self.customer_satisfaction = 0.7
        self.service_quality = 0.7
        self.current_price = 50.0
        self.reviews_pending_response = []
        self.promotion_active = False
        
        # Generate initial customer activity
        self._simulate_customer_activity()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: Dict[str, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one day of business operations with agent's decisions
        
        Args:
            action: Dictionary with agent's decisions for the day
            
        Returns:
            observation: Current business state
            reward: Reward for this step
            terminated: Whether episode is over (business failed)
            truncated: Whether max steps reached
            info: Additional information
        """
        self.current_step += 1
        self.days_elapsed += 1
        self.episode_length += 1
        
        reward_components = {}
        
        # 1. Handle review responses
        reward_components['response'] = self._handle_review_responses(action['response_type'])
        
        # 2. Apply pricing changes
        reward_components['pricing'] = self._handle_pricing(action['pricing_change'])
        
        # 3. Quality investments
        reward_components['quality'] = self._handle_quality_investment(action['quality_investment'])
        
        # 4. Marketing/promotions
        reward_components['marketing'] = self._handle_marketing(action['run_promotion'])
        
        # 5. Simulate customer behavior
        reward_components['customer'] = self._simulate_customer_activity()
        
        # 6. Calculate business health (long-term metric)
        reward_components['health'] = self._calculate_business_health()

        total_reward = sum(reward_components.values())
    
        self.episode_reward += total_reward
        
        # Get new observation
        obs = self._get_observation()
        
        # Termination conditions
        terminated = self.current_rating < 2.0 or self.revenue < 1000
        truncated = self.current_step >= self.max_steps
        
        # Info - don't include 'episode' key unless episode is done
        info = {
            'episode_num': self.episode_num,
            'day': self.days_elapsed,
            'rating': float(self.current_rating),
            'revenue': float(self.revenue),
            'satisfaction': float(self.customer_satisfaction),
            'quality': float(self.service_quality),
            'price': float(self.current_price),
            'pending_reviews': len(self.reviews_pending_response),
            'reward_breakdown': reward_components
        }
        
        # ONLY add 'episode' key when episode ends (REQUIRED by stable-baselines3)
        if terminated or truncated:
            info['episode'] = {
                'r': float(self.episode_reward),
                'l': int(self.episode_length),
                't': float(self.episode_length)
            }
        
        # Log to database if configured
        if self.config.get('log_to_db', True):
            self._log_step_to_db(action, total_reward, reward_components)
        
        return obs, total_reward, terminated, truncated, info
    
    def _handle_review_responses(self, response_type: int) -> float:
        """
        Agent chooses how to respond to the most recent pending review
        
        Returns appropriate reward based on whether response matches review sentiment
        """
        if len(self.reviews_pending_response) == 0:
            return 0.0
        
        reward = 0.0
        review = self.reviews_pending_response[0]
        rating = review['rating']
        
        # Negative review (1-2 stars)
        if rating <= 2:
            if response_type == 0: 
                reward = -3.0
                self.customer_satisfaction -= 0.05
            elif response_type == 1: 
                reward = 5.0
                self.customer_satisfaction += 0.05
                self.current_rating += 0.02
            elif response_type == 2:  
                reward = 2.0
                self.customer_satisfaction += 0.02
            elif response_type == 3:  
                reward = -2.0
                self.customer_satisfaction -= 0.02
        
        # Mixed review (3 stars)
        elif rating == 3:
            if response_type == 0:  
                reward = -1.0
            elif response_type == 1: 
                reward = 1.0
            elif response_type == 2:  
                reward = 3.0
                self.customer_satisfaction += 0.03
            elif response_type == 3: 
                reward = 1.0
        
        # Positive review (4-5 stars)
        else:
            if response_type == 0:  
                reward = -1.0
            elif response_type == 1:  
                reward = -1.0
            elif response_type == 2:  
                reward = 2.0
                self.customer_satisfaction += 0.02
            elif response_type == 3:  
                reward = 4.0
                self.customer_satisfaction += 0.04
        
        # Remove processed review
        self.reviews_pending_response.pop(0)
        
        return reward
    
    def _handle_pricing(self, pricing_change: int) -> float:
        """
        Adjust pricing strategy
        
        Lower prices = more customers but less revenue per customer
        Higher prices = fewer customers but more revenue per customer
        """
        reward = 0.0
        
        if pricing_change == 0:  
            self.current_price *= 0.90
            reward = 0.2 
        elif pricing_change == 2:  
            self.current_price *= 1.30
            reward = -0.1  
        elif pricing_change == 1:
            self.current_price *= 1.0
            reward += 0.2
        

        self.current_price = np.clip(self.current_price, 50, 300.0)
        
        return reward
    
    def _handle_quality_investment(self, investment_level: int) -> float:
        """
        Invest in service quality (training, equipment, etc.)
        
        Costs money now, improves quality and future ratings
        """
        reward = 0.0
        
        if investment_level == 1:  # Moderate investment
            cost = 500
            if self.revenue >= cost:
                self.revenue -= cost
                self.service_quality = min(1.0, self.service_quality + 0.05)
                reward = 2.0  
            else:
                reward = -1.0  
        
        elif investment_level == 2:
            cost = 1500
            if self.revenue >= cost:
                self.revenue -= cost
                self.service_quality = min(1.0, self.service_quality + 0.10)
                reward = 4.0 
            else:
                reward = -2.0  
        
        return reward
    
    def _handle_marketing(self, run_promotion: int) -> float:
        """
        Run marketing campaign
        
        Costs money now, increases customer volume temporarily
        """
        if run_promotion == 1:
            cost = 300
            if self.revenue >= cost:
                self.revenue -= cost
                self.promotion_active = True
                return 1.0  #
            else:
                return -1.0  
        
        self.promotion_active = False
        return 0.0
    
    def _simulate_customer_activity(self) -> float:
        """
        Simulate customers coming to business and leaving reviews
        
        Customer volume depends on:
        - Current rating (higher = more customers)
        - Price (lower = more customers)
        - Promotions (more customers if active)
        
        Review quality depends on:
        - Service quality (higher = better reviews)
        """
        # Base customer volume
        base_customers = 10
        
        # Rating factor (higher rating = more customers)
        rating_factor = self.current_rating / 5.0
        
        # Price factor (lower price = more customers)
        price_ratio = self.current_price / self.base_price
        price_factor = 1.0 - (price_ratio - 1.0) * 0.3
        price_factor = max(0.3, min(1.5, price_factor))
        
        # Promotion boost
        promotion_factor = 1.5 if self.promotion_active else 1.0
        
        # Calculate number of customers
        num_customers = int(base_customers * rating_factor * price_factor * promotion_factor)
        num_customers = max(1, num_customers)
        
        # Revenue from customers
        daily_revenue = num_customers * self.current_price
        self.revenue += daily_revenue
        
        # Generate reviews (about 1/3 of customers leave reviews)
        num_reviews = max(1, num_customers // 3)
        
        for _ in range(num_reviews):
            # Review rating based on service quality
            if random.random() < self.service_quality:
                # Good experience
                rating = random.choices([4, 5], weights=[0.4, 0.6])[0]
            else:
                # Poor experience
                rating = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]
            
            # Create review
            review_texts = {
                1: "Terrible service. Would not recommend.",
                2: "Disappointing experience. Expected better.",
                3: "Service was okay, nothing special.",
                4: "Good service overall. Would come back.",
                5: "Excellent work! Very professional and fast."
            }
            
            review = {
                'rating': rating,
                'text': review_texts[rating],
                'day': self.days_elapsed
            }
            
            self.reviews_pending_response.append(review)
            
            # Update rolling average rating (slow-moving average)
            self.current_rating = 0.95 * self.current_rating + 0.05 * rating
        
        # Reward proportional to revenue generated
        return daily_revenue / 200.0  # Normalize
    
    def _calculate_business_health(self) -> float:
        """
        Calculate overall business health metric
        
        Balances multiple objectives:
        - Revenue (making money)
        - Rating (reputation)
        - Customer satisfaction (retention)
        """
        # Normalize each component to [0, 1]
        revenue_score = np.clip(self.revenue / 30000, 0, 1)
        rating_score = (self.current_rating - 1) / 4.0  # Map [1,5] to [0,1]
        satisfaction_score = self.customer_satisfaction
        
        # Weighted average
        health = (
            0.4 * revenue_score +
            0.4 * rating_score +
            0.2 * satisfaction_score
        )
        
        # Map to [-1, 1] for reward
        return (health * 2.0) - 1.0
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current state observation"""
        return {
            'avg_rating': np.array([self.current_rating], dtype=np.float32),
            'revenue': np.array([self.revenue], dtype=np.float32),
            'satisfaction': np.array([self.customer_satisfaction], dtype=np.float32),
            'service_quality': np.array([self.service_quality], dtype=np.float32),
            'price': np.array([self.current_price], dtype=np.float32),
            'pending_reviews': np.array([len(self.reviews_pending_response)], dtype=np.float32),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information"""
        return {
            'episode_num': self.episode_num,  # Changed from 'episode' to 'episode_num'
            'day': self.days_elapsed,
            'rating': float(self.current_rating),
            'revenue': float(self.revenue),
            'satisfaction': float(self.customer_satisfaction),
            'quality': float(self.service_quality),
            'price': float(self.current_price),
            'pending_reviews': len(self.reviews_pending_response)
        }
    
    def _log_step_to_db(self, action: Dict, reward: float, reward_breakdown: Dict):
        """Log step data to database for analysis"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Log to agent_actions table
            cursor.execute("""
                INSERT INTO agent_actions
                (episode_id, step_number, response_type, pricing_change, 
                 quality_investment, run_promotion, reward, reward_breakdown,
                 rating, revenue, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                self.episode_num,
                self.current_step,
                action['response_type'],
                action['pricing_change'],
                action['quality_investment'],
                action['run_promotion'],
                reward,
                json.dumps(reward_breakdown),
                self.current_rating,
                self.revenue
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            # Don't crash if DB logging fails
            print(f"Warning: Could not log to database: {e}")
    
    def render(self, mode='human'):
        """Display current business state"""
        if mode == 'human':
            print("\n" + "="*70)
            print(f"🏪 BingleFixIt Business Dashboard - Day {self.days_elapsed}")
            print("="*70)
            print(f"⭐ Rating: {self.current_rating:.2f}/5.0")
            print(f"💰 Revenue: ${self.revenue:,.2f}")
            print(f"😊 Customer Satisfaction: {self.customer_satisfaction:.1%}")
            print(f"🔧 Service Quality: {self.service_quality:.1%}")
            print(f"💵 Current Price: ${self.current_price:.2f}")
            print(f"📝 Reviews Pending Response: {len(self.reviews_pending_response)}")
            if self.promotion_active:
                print(f"📢 Promotion: ACTIVE")
            print("="*70)
        
        elif mode == 'ansi':
            return (
                f"Day {self.days_elapsed} | "
                f"Rating: {self.current_rating:.2f} | "
                f"Revenue: ${self.revenue:,.0f} | "
                f"Pending: {len(self.reviews_pending_response)}"
            )
    
    def close(self):
        """Clean up resources"""
        pass


# Register environment
if __name__ != "__main__":
    try:
        from gymnasium.envs.registration import register
        register(
            id='BingleFixIt-v0',
            entry_point='business_gym.business_env:BingleFixItBusinessEnv',
            max_episode_steps=100,
        )
    except:
        pass