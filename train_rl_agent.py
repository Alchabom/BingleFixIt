from review_env import ReviewEnvironment
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re

def main():
    if not torch.backends.mps.is_available():
        print("WARNING: MPS (Apple GPU) not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        print("MPS is available. Running on Apple GPU.")
        device = torch.device("mps")

    
    env_config = {
        'model_name': 'meta-llama/Llama-2-7b-chat-hf',
        'max_length': 512
    }
    env = ReviewEnvironment(env_config)
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    # --- MODIFICATION: 2. Use standard transformers to load model ---
    # We load in float16 (half-precision) since 4-bit is not supported on Mac.
    # This will require ~14GB of memory.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # trust_remote_code=True  # Uncomment if the model requires it
    )
    
    
    model.to(device)
    
    # Training parameters
    num_episodes = 100
    max_steps_per_episode = 10
    
    # Training loop
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
           
            context = f"Current reviews: {observation['reviews']}\nAverage rating: {observation['avg_rating'][0]}\nGenerate a helpful review:"
            
            inputs = tokenizer(context, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # We decode on the CPU
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Derive a rating from generated text using a simple sentiment heuristic.
            # This maps to 1..5, then we convert to 0..4 to match the env's Discrete(5).
            # Replace this with a learned classifier or sentiment model for better results.
            def predict_rating_from_text(text: str) -> int:
                # Very small heuristic: count positive vs negative words
                pos_words = ['good', 'great', 'excellent', 'awesome', 'love', 'recommend', 'fast', 'friendly', 'helpful', 'perfect', 'best', 'amazing']
                neg_words = ['bad', 'terrible', 'awful', 'hate', 'slow', 'rude', 'poor', 'broken', 'problem', 'damage', 'fault']
                text_l = text.lower()
                # strip punctuation for simple matching
                text_l = re.sub(r"[^a-z0-9\s]", ' ', text_l)
                tokens = text_l.split()
                pos = sum(1 for t in tokens if t in pos_words)
                neg = sum(1 for t in tokens if t in neg_words)
                score = pos - neg
                # map score to rating 1..5
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
            # Map into 0..4 for the Discrete(5) action space used by the env
            rating_action = rating_1_5 - 1

            # Take action in environment
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