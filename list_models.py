import google.generativeai as palm
import os
from dotenv import load_dotenv

# --- Load API Key from .env file ---
load_dotenv()
api_key = os.getenv('PALM_API_KEY')
if not api_key:
    print("Error: PALM_API_KEY not found in .env file.")
    exit()
palm.configure(api_key=api_key)
# --- End of setup ---


print("Finding available models...\n")

# Call the 'list_models' function
for m in palm.list_models():
    # Check if 'generateContent' is a supported method for the model
    if 'generateContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")
        print(f"Display Name: {m.display_name}")
        print("-" * 30)