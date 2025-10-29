import gymnasium as gym
import numpy as np
import re  
from typing import List, Optional, Dict, Any, Tuple
import mysql.connector
import json
import random 

class ReviewEnvironment(gym.Env):
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment to an initial state and returns an initial
        observation and info.
        """
        # Set the seed, a required part of the Gym API
        super().reset(seed=seed)
        
        self.current_episode += 1
        self.current_step = 0
        
        obs = {}
        info = {}
        
        try:
            # Get the initial observation
            obs = self._get_obs()
            
            # Get initial info
            info = {
                'episode': self.current_episode,
                'step': self.current_step,
                'avg_rating': obs['avg_rating'][0]
            }
            
            return obs, info

        except Exception as e:
            print(f"Error during environment reset: {e}")
            
            # If reset fails, we MUST still return the correct structure
            # to avoid the TypeError, even if the observation is "empty".
            obs = {
                'reviews': "",
                'avg_rating': np.array([0.0], dtype=np.float32)
            }
            info = {
                'episode': self.current_episode,
                'step': self.current_step,
                'error': str(e)
            }
            
            return obs, info
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.env_config = config
        
        self.db_config = {
            'host': 'localhost',
            'port': 3307,
            'user': 'ai_agent_user',  
            'password': 'frince101', 
            'database': 'mobile_repair'
        }
        
        self.current_episode = 0
        self.current_step = 0
        
        self.tokenizer = None
        if config.get('use_tokenizer', False):
            model_for_tokenizer = config.get('model_name', 'gpt2')
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer)
            except Exception as e:
                print(f"Warning: could not load tokenizer for '{model_for_tokenizer}': {e}")
        self.max_length = config.get('max_length', 512)
        
        self.action_space = gym.spaces.Dict({
            'text': gym.spaces.Text(max_length=self.max_length),
            'rating': gym.spaces.Discrete(5)  # 0-4, which we map to 1-5
        })
        
        self.observation_space = gym.spaces.Dict({
            'reviews': gym.spaces.Text(max_length=self.max_length * 10),
            'avg_rating': gym.spaces.Box(low=1, high=5, shape=(1,), dtype=np.float32)
        })
        

    def _get_reviews(self) -> Tuple[List[str], float]:
        """
        Fetches the 10 most recent agent reviews and calculates the
        *combined* average rating from both user and agent tables.
        """
      
        reviews_list = []
        avg_rating = 3.0  
            
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)


            cursor.execute("""
                SELECT AVG(rating) as avg_rating FROM (
                    SELECT rating FROM comments
                    UNION ALL
                    SELECT rating FROM agent_comments
                ) as combined_reviews
            """)
            result = cursor.fetchone()
            if result and result['avg_rating']:
                avg_rating = float(result['avg_rating'])
                

            cursor.execute("""
                SELECT review_content FROM comments
                ORDER BY created_at DESC
                LIMIT 10
            """)
          
            
            reviews = cursor.fetchall()
            reviews_list = [r['review_content'] for r in reviews]

            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Database error in _get_reviews: {e}")
                
        return reviews_list, avg_rating
    
    def _get_obs(self):
        """
        Gets the current observation from the environment.
        """
        reviews_list, avg_rating = self._get_reviews()
        
        reviews_str = "\n---\n".join(reviews_list)        

        max_obs_len = self.observation_space['reviews'].max_length
        if len(reviews_str) > max_obs_len:
            reviews_str = reviews_str[-max_obs_len:]
            
        return {
            'reviews': reviews_str,
            'avg_rating': np.array([avg_rating], dtype=np.float32)
        }
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        review_text = action['text']
        raw_rating = action['rating'] 
        
        # 1-5 star rating
        rating = int(raw_rating) + 1  
        
    
       
        _, avg_before = self._get_reviews()
 
        target_rating_category = 3 # Default to neutral
        if avg_before < 2.5:
            target_rating_category = 2 
        elif avg_before < 3.5:
            target_rating_category = 3 
        elif avg_before < 4.5:
            target_rating_category = 4 
        else:
            target_rating_category = 5 
            

        internal_reward = self._calculate_reward(review_text, rating)

        # 4. Calculate the "Context" reward:
        #    How well did the agent's action (text + rating) 
        #    match the *target tone category*?
        

        if target_rating_category == 2: 
            rating_match_error = min(abs(rating - 1), abs(rating - 2))
        else:
            rating_match_error = abs(rating - target_rating_category)
            
        context_reward_rating = (2.0 - rating_match_error) 

        text_sentiment = self._predict_rating_from_text(review_text)
        if target_rating_category == 2:
            text_match_error = min(abs(text_sentiment - 1), abs(text_sentiment - 2))
        else:
            text_match_error = abs(text_sentiment - target_rating_category)
            
        context_reward_text = (2.0 - text_match_error) # Max reward +2
        

        reward = internal_reward + context_reward_rating + context_reward_text

        
        
        avg_after = avg_before 

        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()

            action_metadata = {
                'text_length': len(review_text.split()),
                'rating_normalized': bool(0 <= raw_rating <= 4),
                'rating_original': raw_rating,
                'target_category': target_rating_category,
                'reward_internal': internal_reward,
                'reward_ctx_rating': context_reward_rating,
                'reward_ctx_text': context_reward_text
            }
            action_metadata_json = json.dumps(action_metadata) 

            cursor.execute("""
                INSERT INTO agent_comments
                (rating, review_content, episode_id, step_number, reward, action_metadata)
                VALUES
                (%s, %s, %s, %s, %s, %s)
            """, (rating, review_text, self.current_episode, self.current_step,
                  reward, action_metadata_json))
            
            conn.commit() 
            
            # Get the true state after the action
            _, avg_after = self._get_reviews() 

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Database error: {e}")
            reward = -10.0 # Give a large negative reward for DB errors
            if conn and conn.is_connected():
                cursor.close()
                conn.close()


        self.current_step += 1
        
        obs = self._get_obs()
        
        max_steps = self.env_config.get('max_steps_per_episode', 10) 
        terminated = self.current_step >= max_steps
        truncated = False 
        
        info = {
            'episode': self.current_episode,
            'step': self.current_step,
            'avg_rating_before': avg_before,
            'avg_rating_after': avg_after,
            'reward': reward,
            'target_rating_category': target_rating_category
        }
        
        return obs, reward, terminated, truncated, info
    
    
    def _predict_rating_from_text(self, text: str) -> int:
        """
        More robust sentiment prediction that's harder to game.
        """
        strong_pos = ['excellent', 'outstanding', 'amazing', 'fantastic', 'perfect']

        pos_words = ['good', 'great', 'friendly', 'fast', 'professional', 
                    'quality', 'helpful', 'satisfied', 'recommend']
        
        strong_neg = ['terrible', 'horrible', 'awful', 'worst', 'useless']

        neg_words = ['bad', 'poor', 'slow', 'disappointing', 'broken', 'avoid',
                    'problem', 'issue', 'frustrated', 'waste']
        
        neutral_words = ['okay', 'average', 'acceptable', 'fine']
        
        text_l = text.lower()
        text_l = re.sub(r"[^a-z0-9\s]", ' ', text_l)
        words = text_l.split()
        
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return 1  # Penalize spam as 1-star
        
        strong_pos_count = sum(1 for w in words if w in strong_pos)
        pos_count = sum(1 for w in words if w in pos_words)
        strong_neg_count = sum(1 for w in words if w in strong_neg)
        neg_count = sum(1 for w in words if w in neg_words)
        neutral_count = sum(1 for w in words if w in neutral_words)
        
    
        score = (strong_pos_count * 2) + pos_count - neg_count - (strong_neg_count * 2)
        
    
        if neutral_count > 2:
            score -= 1
        
    
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

  
    def _calculate_reward(self, review_text: str, rating: int) -> float:
        """
        Calculate a reward for the agent's generated review.
        1.  Detail (length)
        2.  Honesty (sentiment of text matches the star rating)
        3.  Helpfulness (detailed strong opinions)
        """
        reward = 0.0 

        words = review_text.split()
        word_count = len(words)

        if word_count < 5:
            reward -= 5.0  
        elif 5 <= word_count <= 15:
            reward += 1.0 
        elif 15 < word_count <= 30:
            reward += 2.0  
        elif 30 < word_count <= 50:
            reward += 1.5  
        else:
            reward += 0.5


        unique_words = len(set(words))
        
        if word_count > 0:
            uniqueness_ratio = unique_words / word_count
        
            if uniqueness_ratio < 0.4:
                reward -= 5.0
            elif uniqueness_ratio > 0.7:
                reward += 1.0

        if 'star' in review_text.lower():
             star_count = review_text.lower().count('star')
             if star_count > 3:
                    reward -= 10.0

    
        sentiment_rating = self._predict_rating_from_text(review_text)
        alignment_error = abs(sentiment_rating - rating)

        if alignment_error == 0:
            reward += 3.0  
        elif alignment_error == 1:
            reward += 1.0  
        else:
            reward -= 2.0
        
        if '.' in review_text or '!' in review_text or '?' in review_text:
            reward += 0.5
    
        return float(reward)

    def render(self):
        reviews, avg_rating = self._get_reviews()
        print("\nCurrent Environment State:")
        print(f"Average Rating: {avg_rating:.2f}")
        print("\nLatest Reviews:")
        
        for review in reviews:
            if review:
                print(f"- {review}")
        print("-" * 50)

    def close(self):
        pass
