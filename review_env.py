import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple
import mysql.connector


class ReviewEnvironment(gym.Env):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
     
        self.db_config = {
            'host': 'localhost',
            'port': 3307,
            'user': 'root',
            'password': '',
            'database': 'mobile_repair'
        }
        
      
        self.tokenizer = None
        if config.get('use_tokenizer', False):
            model_for_tokenizer = config.get('model_name', 'gpt2')
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer)
            except Exception as e:
              
                print(f"Warning: could not load tokenizer for '{model_for_tokenizer}': {e}")
        self.max_length = config.get('max_length', 512)
        
        # Define action and observation spaces
        # Action space: Generated review text and rating
        # NOTE: gym.spaces.Discrete(5) produces values 0..4. The environment
        # maps 0..4 -> 1..5 internally. The environment also accepts external
        # ratings already in 1..5 (e.g. from the training script).
        self.action_space = gym.spaces.Dict({
            'text': gym.spaces.Text(max_length=self.max_length),
            'rating': gym.spaces.Discrete(5)
        })
        
        # Observation space: Current state of reviews
        self.observation_space = gym.spaces.Dict({
            'reviews': gym.spaces.Text(max_length=self.max_length * 10),
            'avg_rating': gym.spaces.Box(low=1, high=5, shape=(1,), dtype=np.float32)
        })
        
        self.reset()
    
    def _get_reviews(self) -> Tuple[str, float]:
        """Fetch reviews from database and calculate average rating"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            
            cursor.execute("""
                SELECT review_content, rating 
                FROM comments 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            
            reviews = []
            ratings = []
            
            for review, rating in cursor.fetchall():
                reviews.append(review)
                ratings.append(rating)
            
            cursor.close()
            conn.close()
            
           
            avg_rating = float(np.mean(ratings)) if ratings else 3.0
            reviews_text = " | ".join(reviews) if reviews else ""
            
            return reviews_text, float(avg_rating)
            
        except Exception as e:
            print(f"Database error: {e}")
            return "", 0.0
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
       
        super().reset(seed=seed)
        
       
        reviews, avg_rating = self._get_reviews()
        
        observation = {
            'reviews': reviews,
            'avg_rating': np.array([avg_rating], dtype=np.float32)
        }
        
        return observation, {}
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Dict containing 'text' (review content) and 'rating' (1-5)
        
        Returns:
            observation: Next state
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        review_text = str(action.get('text', '')).strip()
        raw_rating = action.get('rating', 1)

        # Normalize rating values: accept either 0..4 (Discrete) or 1..5 (external)
        try:
            rating_int = int(np.asarray(raw_rating).item())
        except Exception:
            rating_int = 1

        if 0 <= rating_int <= 4:
            rating = rating_int + 1
        else:
           
            rating = max(1, min(5, rating_int))

      
        _, avg_before = self._get_reviews()

      
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO comments 
                (customer_name, email, rating, review_content, created_at, updated_at)
                VALUES 
                ('RL_Agent', 'agent@rl.ai', %s, %s, NOW(), NOW())
            """, (rating, review_text))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Database error: {e}")
           
            obs, _ = self.reset()
            return obs, -1.0, True, False, {'error': str(e)}

        
        reviews_after, avg_after = self._get_reviews()

        observation = {
            'reviews': reviews_after,
            'avg_rating': np.array([avg_after], dtype=np.float32)
        }

      
        reward = self._calculate_reward(review_text, rating, avg_before, avg_after)

        return observation, reward, False, False, {}
    
    def _calculate_reward(self, review_text: str, rating: int, avg_before: float, avg_after: float) -> float:
        """
        Calculate reward for the action
        - Length of review (longer reviews might be better)
        - Sentiment alignment (rating should match sentiment)
        - Impact on average rating
        - Text quality metrics
        """
        reward = 0.0

        review_length = len(review_text.split())
        if review_length > 10:
            reward += 0.5
        if review_length > 30:
            reward += 0.5

      
        rating_diff = abs(rating - avg_before)
        if rating_diff < 1.0:
            reward += 1.0

        # Reward the agent for improving the average rating (delta)
        # delta > 0 means the average rating increased after the agent's submission
        delta = float(avg_after - avg_before)
        reward += delta

        return float(reward)

    def render(self):
       
        reviews, avg_rating = self._get_reviews()
        print("\nCurrent Environment State:")
        print(f"Average Rating: {avg_rating:.2f}")
        print("\nLatest Reviews:")
        for review in reviews.split(" | "):
            if review:
                print(f"- {review}")
        print("-" * 50)

    def close(self):
        pass