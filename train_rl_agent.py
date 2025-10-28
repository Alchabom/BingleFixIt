from review_env import ReviewEnvironment
import numpy as np
import re
import google.generativeai as palm
import os
from dotenv import load_dotenv
import time

def configure_api():
    load_dotenv() 
    api_key = os.getenv('PALM_API_KEY')
    if not api_key:
        raise ValueError(
            "Google PaLM API key not found. Please create a .env file with your PALM_API_KEY. "
            "Get a free key from https://makersuite.google.com/app/apikey"
        )
    palm.configure(api_key=api_key)
    return palm

def main():
    palm = configure_api()
    model_name = os.getenv('GEMINI_MODEL', os.getenv('PALM_MODEL', 'models/chat-bison-001'))

    env_config = {
        'model_name': model_name,
        'max_length': 512
    }
    env = ReviewEnvironment(env_config)
    max_output_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', '2048'))
    max_review_chars = int(os.getenv('MAX_REVIEW_CHARS', '60000'))
    
    
    num_episodes = 100
    max_steps_per_episode = 10

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
           
           
            prior_reviews = observation.get('reviews', '') or ''
            if len(prior_reviews) > max_review_chars:
                prior_reviews = prior_reviews[-max_review_chars:]
                length_of_reviews = len(prior_reviews)
                print(f"length of reviews:  {length_of_reviews}")
                print(f"(Note) Truncated prior reviews to last {max_review_chars} chars to fit model context")

            prompt = f"""You are an expert at writing helpful product reviews.
Based on the following context, write a detailed, honest review:

Current reviews (truncated): {prior_reviews}
Average rating: {observation['avg_rating'][0]}

Write a helpful, detailed review that contributes meaningful feedback:"""
            
            # for rate limits 
            time.sleep(1)
            
            
            model = palm.GenerativeModel(model_name)

            
            generation_config = {
                'temperature': 0.7,
                    'max_output_tokens': max_output_tokens,
                'top_p': 0.9,
            }

            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }


            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config, 
                    safety_settings= safety_settings
                )

                if response.candidates and response.candidates[0].finish_reason == 1:
                    generated_text = response.text.strip()
                else:
                    reason = response.candidates[0].finish_reason.name if (response.candidates and response.candidates[0].finish_reason.name) else "UNKNOWN_REASON"
                    print(f"Warning: Generation stopped. Finish Reason: {reason}")
                    generated_text = "Service was okay."

                
            except Exception as e:
                print(f"Warning: API call failed. {e}")
                generated_text = "Service was okay." 

            # --- END: MODIFIED BLOCK ---
            

            def predict_rating_from_text(text: str) -> int:
                
                pos_words = ['good', 'great', 'excellent', 'awesome', 'love', 'recommend', 'fast', 
                           'friendly', 'helpful', 'perfect', 'best', 'amazing', 'satisfied', 'professional']
                neg_words = ['bad', 'terrible', 'awful', 'hate', 'slow', 'rude', 'poor', 'broken', 
                           'problem', 'damage', 'fault', 'disappointed', 'waste', 'unprofessional']
                
                text_l = text.lower()
                text_l = re.sub(r"[^a-z0-9\s]", ' ', text_l)
                text_l = ' '.join(text_l.split())
                
                tokens = text_l.split()
                pos = sum(1 for t in tokens if t in pos_words)
                neg = sum(1 for t in tokens if t in neg_words)
                score = pos - neg
                if score <= -2:
                    rating = 1
                elif score == -1:
                    rating = 2
                elif score == 0:
                    rating = 3
                elif score == 1:
                    rating = 4
                else:
                    rating = 5
                return rating

            rating_1_5 = predict_rating_from_text(generated_text)
            rating_action = rating_1_5 - 1

            action = {
                'text': generated_text,
                'rating': int(rating_action)
            }
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} finished with reward: {episode_reward}")
        env.render() 

if __name__ == "__main__":
    main()