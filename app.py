import streamlit as st
import subprocess
import json
import sys
from pathlib import Path
from PIL import Image
import time
import os

# --- Page Config ---
st.set_page_config(
    page_title="PPO Business Agent Demo",
    page_icon="",
    layout="wide"
)

# --- Helper Functions ---
def check_model_exists():
    """Check if a trained model exists"""
    best_model_path = Path('./models/best_model/best_model.zip')
    return best_model_path.exists()

def check_results_exist():
    """Check if evaluation results exist"""
    ppo_path = Path('./results/ppo_evaluation.json')
    random_path = Path('./results/random_baseline.json')
    return ppo_path.exists() and random_path.exists()

def run_training(progress_bar, status_text):
    """Run the training script"""
    status_text.text("Training PPO agent... This may take 5-10 minutes.")
    
    try:
        # Run training script
        process = subprocess.Popen(
            [sys.executable, "ppo_agent.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            if "Episode" in line or "Step" in line:
                status_text.text(line.strip())
        
        process.wait()
        
        if process.returncode == 0:
            progress_bar.progress(0.33)
            return True, "Training completed successfully!"
        else:
            stderr = process.stderr.read()
            return False, f"Training failed: {stderr}"
            
    except Exception as e:
        return False, f"Training error: {str(e)}"

def run_baseline(progress_bar, status_text):
    """Run the random baseline evaluation"""
    status_text.text("Evaluating random baseline agent...")
    
    try:
        result = subprocess.run(
            [sys.executable, "baseline_random_agent.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            progress_bar.progress(0.66)
            return True, "Baseline evaluation completed!"
        else:
            return False, f"Baseline failed: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Baseline evaluation timed out"
    except Exception as e:
        return False, f"Baseline error: {str(e)}"

def run_visualizations(progress_bar, status_text):
    """Generate comparison visualizations"""
    status_text.text("Creating visualizations...")
    
    try:
        result = subprocess.run(
            [sys.executable, "create_visualizations.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            progress_bar.progress(1.0)
            return True, "Visualizations created!"
        else:
            return False, f"Visualization failed: {result.stderr}"
            
    except Exception as e:
        return False, f"Visualization error: {str(e)}"

def load_results():
    """Load evaluation results from JSON files"""
    ppo_path = Path('./results/ppo_evaluation.json')
    random_path = Path('./results/random_baseline.json')
    
    if not ppo_path.exists() or not random_path.exists():
        return None, None
    
    with open(ppo_path, 'r') as f:
        ppo_data = json.load(f)
    with open(random_path, 'r') as f:
        random_data = json.load(f)
    
    return ppo_data, random_data

def display_results(ppo_data, random_data):
    """Display evaluation results with metrics and visualizations"""
    
    st.subheader("🏆 Performance Comparison")
    
    # Calculate improvements
    reward_imp = ((ppo_data['mean_reward'] - random_data['mean_reward']) / 
                  abs(random_data['mean_reward'])) * 100
    revenue_imp = ((ppo_data['mean_revenue'] - random_data['mean_revenue']) / 
                   random_data['mean_revenue']) * 100
    rating_imp = ((ppo_data['mean_rating'] - random_data['mean_rating']) / 
                  random_data['mean_rating']) * 100
    
    # Display key metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Reward Improvement",
            f"{reward_imp:+.1f}%",
            delta=f"{ppo_data['mean_reward']:.1f} vs {random_data['mean_reward']:.1f}"
        )
    
    with col2:
        st.metric(
            "Revenue Improvement",
            f"{revenue_imp:+.1f}%",
            delta=f"${ppo_data['mean_revenue']:,.0f} vs ${random_data['mean_revenue']:,.0f}"
        )
    
    with col3:
        st.metric(
            "Rating Improvement",
            f"{rating_imp:+.1f}%",
            delta=f"{ppo_data['mean_rating']:.2f} vs {random_data['mean_rating']:.2f}"
        )
    
    # Detailed metrics
    st.subheader("Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###PPO Agent")
        st.write(f"**Mean Reward:** {ppo_data['mean_reward']:.2f} ± {ppo_data['std_reward']:.2f}")
        st.write(f"**Final Rating:** {ppo_data['mean_rating']:.2f}/5.0 ")
        st.write(f"**Final Revenue:** ${ppo_data['mean_revenue']:,.2f}")
        st.write(f"**Episode Length:** {ppo_data.get('mean_length', 0):.1f} steps")
    
    with col2:
        st.markdown("###Random Baseline")
        st.write(f"**Mean Reward:** {random_data['mean_reward']:.2f} ± {random_data['std_reward']:.2f}")
        st.write(f"**Final Rating:** {random_data['mean_rating']:.2f}/5.0")
        st.write(f"**Final Revenue:** ${random_data['mean_revenue']:,.2f}")
        st.write(f"**Failure Rate:** {random_data.get('failure_rate', 0):.1%}")
    
    # Visualization
    st.subheader("Visual Comparison")
    plot_path = Path('./results/comprehensive_comparison.png')
    
    if plot_path.exists():
        image = Image.open(plot_path)
        st.image(image, caption="PPO vs Random Agent Performance", use_column_width=True)
    else:
        st.warning("Visualization not found. Run visualizations again.")
    
    # Verdict
    st.subheader("Verdict")
    if reward_imp > 50:
        st.success("PPO agent SIGNIFICANTLY outperforms random baseline!")
    elif reward_imp > 20:
        st.success("PPO agent clearly learned effective strategies")
    elif reward_imp > 0:
        st.info("PPO agent shows improvement over baseline")
    else:
        st.warning("PPO agent needs more training")

# --- Main App ---
st.title("PPO Agent for Business Management")

st.markdown("""
This project demonstrates a **Reinforcement Learning agent** (using PPO) trained to manage 
a "BingleFixIt" service business. The agent learns to make optimal decisions on:
- Review response strategies
- Pricing optimization
- Quality investments
- Marketing campaigns

The goal is to maximize revenue while maintaining high customer ratings.
""")

# Check current status
model_exists = check_model_exists()
results_exist = check_results_exist()

st.divider()

# Training Section
st.header("Training Phase")

if model_exists:
    st.success(" Trained model found!")
    model_path = Path('./models/best_model/best_model.zip')
    st.caption(f"Model location: `{model_path}`")
else:
    st.info("No trained model found. Train a new agent below.")

if st.button(" Train New PPO Agent", type="primary", disabled=False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Training in progress..."):
        success, message = run_training(progress_bar, status_text)
        
        if success:
            st.success(message)
            st.rerun()  # Refresh to update model status
        else:
            st.error(message)

st.divider()

# Evaluation Section
st.header(" Evaluation Phase")

if not model_exists:
    st.warning("Train a model first before running evaluation.")
elif results_exist:
    st.success("Evaluation results available!")
    
    if st.button("Re-run Evaluation"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Running evaluation..."):
            # Run baseline
            success1, msg1 = run_baseline(progress_bar, status_text)
            if not success1:
                st.error(msg1)
            
            # Generate visualizations
            success2, msg2 = run_visualizations(progress_bar, status_text)
            if not success2:
                st.error(msg2)
            
            if success1 and success2:
                st.success("Evaluation complete!")
                st.rerun()
else:
    st.info("Run evaluation to compare PPO vs Random baseline.")
    
    if st.button("Run Evaluation", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Running evaluation..."):
            # Run baseline
            success1, msg1 = run_baseline(progress_bar, status_text)
            if not success1:
                st.error(msg1)
            
            # Generate visualizations
            success2, msg2 = run_visualizations(progress_bar, status_text)
            if not success2:
                st.error(msg2)
            
            if success1 and success2:
                st.success("Evaluation complete!")
                st.rerun()

st.divider()

# Results Section
st.header("Results Dashboard")

if results_exist:
    ppo_data, random_data = load_results()
    if ppo_data and random_data:
        display_results(ppo_data, random_data)
    else:
        st.error("Could not load results files.")
else:
    st.info("Results will appear here after running evaluation.")

# Footer
st.divider()
st.caption("""
**Note:** Training can take 5-10 minutes depending on your hardware. 
The agent learns through trial and error, optimizing business decisions over 100,000 timesteps.
""")