from review_env import ReviewEnvironment
import numpy as np
import mysql.connector
from typing import List, Dict, Any

def get_agent_reviews(db_config: Dict[str, Any], episode_id: int) -> List[Dict[str, Any]]:
    """Fetch all reviews and their rewards for a specific episode"""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT rating, review_content, step_number, reward, action_metadata
        FROM agent_comments
        WHERE episode_id = %s
        ORDER BY step_number ASC
    """, (episode_id,))
    
    reviews = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return reviews

def print_episode_summary(episode_id: int, reviews: List[Dict[str, Any]]):
    """Print a summary of an episode's reviews and rewards"""
    print(f"\n=== Episode {episode_id} Summary ===")
    total_reward = 0
    
    for review in reviews:
        print(f"\nStep {review['step_number']}:")
        print(f"Review: {review['review_content']}")
        print(f"Rating: {review['rating']}/5")
        print(f"Reward: {review['reward']}")
        print(f"Metadata: {review['action_metadata']}")
        total_reward += review['reward']
    
    print(f"\nTotal Episode Reward: {total_reward}")
    print("=" * 50)

def main():
    # Initialize environment with test configuration
    env_config = {
        'model_name': 'test-model',
        'max_length': 100
    }
    env = ReviewEnvironment(env_config)
    
    # Run 2 episodes with 3 steps each
    for episode in range(2):
        observation, info = env.reset()
        print(f"\nStarting Episode {episode + 1}")
        print(f"Initial Observation: {observation}")
        
        for step in range(3):
            # Generate a test review
            review_text = f"Test review for episode {episode + 1}, step {step + 1}"
            rating = np.random.randint(0, 5)  # 0-4 for testing rating normalization
            
            action = {
                'text': review_text,
                'rating': rating
            }
            
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"\nStep {step + 1}:")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"New Observation: {observation}")
            
            if terminated or truncated:
                break
        
        # Get and print episode summary from database
        reviews = get_agent_reviews(env.db_config, episode + 1)
        print_episode_summary(episode + 1, reviews)

if __name__ == "__main__":
    main()