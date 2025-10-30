"""
BingleFixIt Business Management RL Environment - Enhanced Version

ENHANCEMENTS:
- Initializes from real database reviews
- More realistic customer behavior based on historical data
- Better reward shaping
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
    Enhanced RL Environment with Real Data Integration
    
    Improvements over base version:
    - Loads real reviews from database on initialization
    - Calculates initial rating from actual customer feedback
    - Review distribution matches historical patterns
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
        
        # Load historical reviews for realistic initialization
        self.historical_reviews = self._load_historical_reviews()
        self.review_distribution = self._analyze_review_distribution()
        
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
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
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
    
    def _load_historical_reviews(self) -> List[Dict]:
        """Load real reviews from database for realistic simulation"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT rating, review_content 
                FROM comments 
                ORDER BY created_at DESC 
                LIMIT 500
            """)
            reviews = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if reviews and len(reviews) > 0:
                print(f"✓ Loaded {len(reviews)} historical reviews from database")
                return reviews
            else:
                print("No reviews found in database, using defaults")
                return []
                
        except Exception as e:
            print(f" Could not load reviews from database: {e}")
            return []
    
    def _analyze_review_distribution(self) -> Dict[int, float]:
        """Analyze distribution of ratings from historical data"""
        if not self.historical_reviews:
            # Default distribution if no data
            return {1: 0.15, 2: 0.15, 3: 0.40, 4: 0.20, 5: 0.10}
        
        rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for review in self.historical_reviews:
            rating = review['rating']
            if 1 <= rating <= 5:
                rating_counts[rating] += 1
        
        total = sum(rating_counts.values())
        distribution = {k: v/total for k, v in rating_counts.items()}
        
        print(f"  Rating distribution: {distribution}")
        return distribution
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment with realistic initialization from historical data"""
        super().reset(seed=seed)
        
        self.episode_num += 1
        self.current_step = 0
        self.days_elapsed = 0
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # Initialize from historical data if available
        if self.historical_reviews:
            # Calculate actual average rating from database
            avg_rating = np.mean([r['rating'] for r in self.historical_reviews])
            self.current_rating = float(avg_rating)
            
            # Initial satisfaction based on rating distribution
            positive_ratio = (self.review_distribution.get(4, 0) + 
                            self.review_distribution.get(5, 0))
            self.customer_satisfaction = np.clip(positive_ratio + 0.3, 0.4, 0.9)
            
            # Initial quality estimate from positive reviews
            self.service_quality = np.clip((avg_rating - 1) / 4.0, 0.5, 0.9)
        else:
            # Default initialization
            self.current_rating = 3.5
            self.customer_satisfaction = 0.7
            self.service_quality = 0.7
        
        # Reset other business metrics
        self.revenue = 10000.0
        self.current_price = 50.0
        self.reviews_pending_response = []
        self.promotion_active = False
        
        # Generate initial customer activity
        self._simulate_customer_activity()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: Dict[str, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one day of business operations"""
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
        
        # 6. Business health (long-term metric)
        reward_components['health'] = self._calculate_business_health()

        total_reward = sum(reward_components.values())
        self.episode_reward += total_reward
        
        # Get new observation
        obs = self._get_observation()
        
        # Termination conditions
        terminated = self.current_rating < 2.0 or self.revenue < 1000
        truncated = self.current_step >= self.max_steps
        
        # Info dictionary
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
        
        # Add episode key only when done (required by stable-baselines3)
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
        """Handle review responses with improved reward shaping"""
        if len(self.reviews_pending_response) == 0:
            return 0.0
        
        reward = 0.0
        review = self.reviews_pending_response[0]
        rating = review['rating']
        
        # Negative review (1-2 stars) - apologetic response is best
        if rating <= 2:
            if response_type == 0:  # Ignore
                reward = -5.0  # Stronger penalty
                self.customer_satisfaction -= 0.08
            elif response_type == 1:  # Apologetic
                reward = 8.0  # Higher reward
                self.customer_satisfaction += 0.06
                self.current_rating += 0.03
            elif response_type == 2:  # Professional
                reward = 3.0
                self.customer_satisfaction += 0.03
            elif response_type == 3:  # Grateful
                reward = -3.0
                self.customer_satisfaction -= 0.03
        
        # Mixed review (3 stars) - professional response is best
        elif rating == 3:
            if response_type == 0:  # Ignore
                reward = -2.0
            elif response_type == 1:  # Apologetic
                reward = 2.0
            elif response_type == 2:  # Professional
                reward = 5.0
                self.customer_satisfaction += 0.04
            elif response_type == 3:  # Grateful
                reward = 2.0
        
        # Positive review (4-5 stars) - grateful response is best
        else:
            if response_type == 0:  # Ignore
                reward = -2.0
            elif response_type == 1:  # Apologetic
                reward = -2.0
            elif response_type == 2:  # Professional
                reward = 3.0
                self.customer_satisfaction += 0.03
            elif response_type == 3:  # Grateful
                reward = 7.0  # Higher reward
                self.customer_satisfaction += 0.05
        
        # Remove processed review
        self.reviews_pending_response.pop(0)
        
        # Clamp satisfaction
        self.customer_satisfaction = np.clip(self.customer_satisfaction, 0.0, 1.0)
        
        return reward
    
    def _handle_pricing(self, pricing_change: int) -> float:
        """Improved pricing mechanics"""
        reward = 0.0
        old_price = self.current_price
        
        if pricing_change == 0:  # Decrease 10%
            self.current_price *= 0.90
            # Small reward for strategic price reduction
            reward = 1.0
        elif pricing_change == 2:  # Increase 10%
            self.current_price *= 1.10
            # Penalty for price increase (reduces customer volume)
            reward = -2.0
        else:  # Maintain
            reward = 0.5  # Small reward for stability
        
        # Clamp price to reasonable range
        self.current_price = np.clip(self.current_price, 30.0, 150.0)
        
        return reward
    
    def _handle_quality_investment(self, investment_level: int) -> float:
        """Quality investment with better balance"""
        reward = 0.0
        
        if investment_level == 1:  # Moderate investment ($500)
            cost = 500
            if self.revenue >= cost:
                self.revenue -= cost
                self.service_quality = min(1.0, self.service_quality + 0.05)
                reward = 3.0  # Long-term investment reward
            else:
                reward = -2.0  # Penalty for overextending
        
        elif investment_level == 2:  # High investment ($1500)
            cost = 1500
            if self.revenue >= cost:
                self.revenue -= cost
                self.service_quality = min(1.0, self.service_quality + 0.10)
                reward = 6.0  # Higher long-term reward
            else:
                reward = -3.0  # Bigger penalty
        
        return reward
    
    def _handle_marketing(self, run_promotion: int) -> float:
        """Marketing campaign handling"""
        if run_promotion == 1:
            cost = 300
            if self.revenue >= cost:
                self.revenue -= cost
                self.promotion_active = True
                return 2.0  # Reward for growth strategy
            else:
                return -2.0  # Penalty for bad timing
        
        self.promotion_active = False
        return 0.0
    
    def _simulate_customer_activity(self) -> float:
        """
        Simulate customers using realistic review distribution
        """
        # Base customer volume
        base_customers = 10
        
        # Rating factor (higher rating = more customers)
        rating_factor = self.current_rating / 5.0
        
        # Price factor (lower price = more customers)
        price_ratio = self.current_price / self.base_price
        price_factor = 1.0 - (price_ratio - 1.0) * 0.5  # Stronger price sensitivity
        price_factor = max(0.2, min(1.8, price_factor))
        
        # Promotion boost
        promotion_factor = 1.5 if self.promotion_active else 1.0
        
        # Calculate number of customers
        num_customers = int(base_customers * rating_factor * price_factor * promotion_factor)
        num_customers = max(1, num_customers)
        
        # Revenue from customers
        daily_revenue = num_customers * self.current_price
        self.revenue += daily_revenue
        
        # Generate reviews using historical distribution
        num_reviews = max(1, num_customers // 3)
        
        for _ in range(num_reviews):
            # Use historical distribution if available
            if self.review_distribution:
                ratings = list(self.review_distribution.keys())
                weights = list(self.review_distribution.values())
                rating = random.choices(ratings, weights=weights)[0]
            else:
                # Quality-based rating
                if random.random() < self.service_quality:
                    rating = random.choices([4, 5], weights=[0.4, 0.6])[0]
                else:
                    rating = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]
            
            # Use real review text if available
            review_text = "Customer feedback"
            if self.historical_reviews:
                matching_reviews = [r for r in self.historical_reviews if r['rating'] == rating]
                if matching_reviews:
                    review_text = random.choice(matching_reviews)['review_content']
            else:
                review_texts = {
                    1: "Terrible service. Would not recommend.",
                    2: "Disappointing experience. Expected better.",
                    3: "Service was okay, nothing special.",
                    4: "Good service overall. Would come back.",
                    5: "Excellent work! Very professional and fast."
                }
                review_text = review_texts[rating]
            
            review = {
                'rating': rating,
                'text': review_text,
                'day': self.days_elapsed
            }
            
            self.reviews_pending_response.append(review)
            
            # Update rolling average rating
            self.current_rating = 0.95 * self.current_rating + 0.05 * rating
        
        # Reward proportional to revenue (normalized)
        return daily_revenue / 150.0
    
    def _calculate_business_health(self) -> float:
        """Calculate balanced business health metric"""
        # Normalize components
        revenue_score = np.clip(self.revenue / 30000, 0, 1)
        rating_score = (self.current_rating - 1) / 4.0
        satisfaction_score = self.customer_satisfaction
        quality_score = self.service_quality
        
        # Weighted average emphasizing long-term metrics
        health = (
            0.3 * revenue_score +
            0.3 * rating_score +
            0.2 * satisfaction_score +
            0.2 * quality_score
        )
        
        # Map to reward range [-2, 2]
        return (health * 4.0) - 2.0
    
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
            'episode_num': self.episode_num,
            'day': self.days_elapsed,
            'rating': float(self.current_rating),
            'revenue': float(self.revenue),
            'satisfaction': float(self.customer_satisfaction),
            'quality': float(self.service_quality),
            'price': float(self.current_price),
            'pending_reviews': len(self.reviews_pending_response)
        }
    
    def _log_step_to_db(self, action: Dict, reward: float, reward_breakdown: Dict):
        """Log step data to database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
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
            pass
    
    def render(self, mode='human'):
        """Display current business state"""
        if mode == 'human':
            print("\n" + "="*70)
            print(f" BingleFixIt Business Dashboard - Day {self.days_elapsed}")
            print("="*70)
            print(f"Rating: {self.current_rating:.2f}/5.0")
            print(f"Revenue: ${self.revenue:,.2f}")
            print(f"Customer Satisfaction: {self.customer_satisfaction:.1%}")
            print(f"Service Quality: {self.service_quality:.1%}")
            print(f"Current Price: ${self.current_price:.2f}")
            print(f"Reviews Pending Response: {len(self.reviews_pending_response)}")
            if self.promotion_active:
                print(f"Promotion: ACTIVE")
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