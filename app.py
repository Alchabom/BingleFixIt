import streamlit as st
import subprocess
import json
from pathlib import Path
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="PPO Business Agent Demo",
    page_icon="🤖",
    layout="wide"
)

# --- Functions (Copy these from your scripts) ---
# We need the evaluation functions. Ideally, refactor your
# ppo_agent.py and baseline_random_agent.py so you can 
# import:
# from ppo_agent import evaluate_trained_agent
# from baseline_random_agent import evaluate_random_agent
# from create_visualizations import create_comparison_plots

# For this example, we'll just call them as scripts.
def run_full_evaluation():
    """Runs all evaluation scripts."""
    st.info("Starting evaluation... (This may take a minute)")
    
    # 1. Run the PPO agent evaluation
    # (This assumes your ppo_agent.py saves its results)
    subprocess.run(["python", "ppo_agent.py"], capture_output=True, text=True)
    
    # 2. Run the baseline agent evaluation
    # (This assumes baseline_random_agent.py saves its results)
    subprocess.run(["python", "baseline_random_agent.py"], capture_output=True, text=True)
    
    # 3. Generate visualizations
    subprocess.run(["python", "create_visualizations.py"], capture_output=True, text=True)
    st.success("Evaluation complete! Results below.")


# --- Main App ---
st.title("🤖 PPO Agent for Business Management")
st.write("""
This project demonstrates a Reinforcement Learning agent (using PPO) trained to manage a "BingleFixIt" service business. 
The agent learns to make optimal decisions on pricing, review responses, and quality investments to maximize revenue while maintaining a high customer rating.
""")

if st.button("► Run Live Agent vs. Random Baseline", type="primary"):
    with st.spinner("Running simulations... Please wait."):
        # 1. Run the scripts
        run_full_evaluation()
        
        # 2. Load the results (as defined in your scripts)
        ppo_results_path = Path('./results/ppo_evaluation.json')
        random_results_path = Path('./results/random_baseline.json')
        plot_path = Path('./results/comprehensive_comparison.png')

        if not ppo_results_path.exists() or not random_results_path.exists():
            st.error("Error: Could not find result files after running scripts.")
        else:
            with open(ppo_results_path, 'r') as f:
                ppo_data = json.load(f)
            with open(random_results_path, 'r') as f:
                random_data = json.load(f)

            # 3. Display Key Metrics (from your README.md)
            st.subheader("🏆 Results at a Glance")
            col1, col2, col3 = st.columns(3)
            
            reward_imp = (ppo_data['mean_reward'] / random_data['mean_reward'] - 1) * 100
            revenue_imp = (ppo_data['mean_revenue'] / random_data['mean_revenue'] - 1) * 100

            col1.metric("Reward Improvement", f"{reward_imp:+.1f}%")
            col2.metric("Revenue Improvement", f"{revenue_imp:+.1f}%")
            col3.metric("PPO Final Rating", f"{ppo_data['mean_rating']:.2f} / 5.0 ⭐")
            
            st.metric("PPO Avg. Revenue", f"${ppo_data['mean_revenue']:,.2f}")
            st.metric("Random Avg. Revenue", f"${random_data['mean_revenue']:,.2f}")

            # 4. Display Plot
            if plot_path.exists():
                st.subheader("Visual Comparison")
                image = Image.open(plot_path)
                st.image(image, caption="PPO vs. Random Agent Performance")
            else:
                st.warning("Could not find visualization image.")
else:
    st.info("Click the button to run the evaluation.")