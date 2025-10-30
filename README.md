# BingleFixIt: Business Management RL Gym 🏪🤖

A realistic reinforcement learning environment for training AI agents to manage a service business. Built on a real PHP/MySQL storefront with actual customer review mechanics and 286 seeded realistic reviews. 
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)

## What This Is

An **OpenAI Gym-compatible environment** where RL agents learn to:
- Respond to customer reviews appropriately
- Balance pricing vs. customer satisfaction  
- Make strategic quality investment decisions
- Manage business reputation over time
- Maximize long-term revenue while maintaining ratings

**Key Feature:** Trained on **286 seeded customer reviews** from actual database, replicating realistic customer behavior patterns.

## 🏆 Results at a Glance

```
┌────────────────────────────────────────────────────┐
│  PPO Agent vs Random Baseline                      │
├────────────────────────────────────────────────────┤
│  Reward Improvement:     +437.3% 📊               │
│  Revenue Improvement:    +3,224.4% 💰             │
│  Final Rating:           3.03/5.0 ⭐              │
│  Business Survival:      100% (vs 0% baseline) ✅  │
└────────────────────────────────────────────────────┘
```

The trained PPO agent generates **$25,351 average revenue** compared to **$763** for random decisions, while maintaining customer satisfaction and never failing.

## 🤔 Why This Matters

**The AI industry needs realistic training environments.** Most RL research uses:
- Simple games (CartPole, Atari)
- Simulated robotics (MuJoCo)
- Abstract tasks

**BingleFixIt provides:**
1. Real-world business simulation with actual economic outcomes
2. Historical data integration (286 reviews from MySQL database)
3. Multi-objective optimization (revenue + reputation + satisfaction)
4. Practical AI application demonstrating business value

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│   RL Agent (PPO/DQN/A3C)           │
│   - Observes business state         │
│   - Decides actions                 │
│   - Learns from rewards             │
└──────────────┬──────────────────────┘
               │ actions
               ↓
┌─────────────────────────────────────┐
│   BingleFixItBusinessGym            │
│   (OpenAI Gym Environment)          │
│                                     │
│   State:                            │
│   - Rating, Revenue, Satisfaction   │
│   - Quality, Price, Pending Reviews │
│                                     │
│   Actions:                          │
│   - Review responses (4 types)      │
│   - Pricing changes (3 options)     │
│   - Quality investment (3 levels)   │
│   - Marketing (on/off)              │
│                                     │
│   Rewards:                          │
│   - Customer satisfaction           │
│   - Revenue generation              │
│   - Long-term business health       │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   MySQL Database                    │
│   - comments (286 reviews)          │
│   - agent_actions (training logs)   │
│   - Business metrics tracking       │
└─────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   PHP Storefront (Optional)         │
│   - Real website interface          │
│   - Live review submission          │
└─────────────────────────────────────┘
```

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/binglefixit-rl-gym
cd binglefixit-rl-gym

# Install dependencies
pip3 install -r requirements.txt

# Setup database (requires MySQL)

# Update db_config with your credentials
```

### Train PPO Agent

```bash
# Train agent (100K timesteps, ~10 minutes)
python3 ppo_agent.py

# Outputs:
# - Trained model: ./models/ppo_binglefixit_TIMESTAMP.zip
# - Training curves: ./results/training_curves.png
# - Logs for TensorBoard: ./logs/
```

### Run Baseline Comparison

```bash
# Evaluate random baseline and compare with PPO
python3 baseline_random_agent.py

# Outputs:
# - Random agent results: ./results/random_baseline.json
# - PPO comparison: ./results/ppo_evaluation.json
# - Performance comparison printed to terminal
```

### Create Visualizations

```bash
# Generate comprehensive analysis plots
python3 create_visualizations.py

# Outputs:
# - Comparison charts: ./results/agent_comparison.png
# - Training metrics: ./results/training_analysis.png
```

### View Training Progress

```bash
# Launch TensorBoard (while or after training)
tensorboard --logdir ./logs

# Open browser to http://localhost:6006
```

## 📊 State Space

The environment provides a **Dict observation space** with these features:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `avg_rating` | Box | [1.0, 5.0] | Current average business rating |
| `revenue` | Box | [0, 100000] | Cumulative revenue in dollars |
| `satisfaction` | Box | [0, 1] | Customer satisfaction score |
| `service_quality` | Box | [0, 1] | Service quality level |
| `price` | Box | [10, 200] | Current service price ($) |
| `pending_reviews` | Box | [0, 50] | Number of reviews awaiting response |

**Note:** State uses historical review distribution from database for realistic customer behavior.

## 🎮 Action Space

The environment uses a **Dict action space** with discrete choices:

| Action | Type | Options | Description |
|--------|------|---------|-------------|
| `response_type` | Discrete(4) | 0: Ignore<br>1: Apologetic<br>2: Professional<br>3: Grateful | How to respond to reviews |
| `pricing_change` | Discrete(3) | 0: Decrease 10%<br>1: Maintain<br>2: Increase 10% | Pricing strategy |
| `quality_investment` | Discrete(3) | 0: None ($0)<br>1: Moderate ($500)<br>2: High ($1500) | Training & equipment investment |
| `run_promotion` | Discrete(2) | 0: No<br>1: Yes ($300 cost) | Marketing campaign |

**Total action space:** 4 × 3 × 3 × 2 = **72 possible actions**

## Reward Structure

The reward function balances six components to encourage holistic business management:

```python
total_reward = (
    response_reward +      # +8 for appropriate responses (e.g., apologetic to 1★)
    pricing_reward +       # +1 for strategic pricing
    quality_reward +       # +3 to +6 for quality investments
    marketing_reward +     # +2 for growth campaigns
    customer_reward +      # Revenue/150 (normalized daily income)
    health_reward          # -2 to +2 for long-term business health
)
```

**Reward signals:**
- Apologize to negative reviews: +8
- Thank positive reviews: +7  
- Ignore negative reviews: -5
- Quality investment (when affordable): +6
- Business health (rating >4, revenue >$20K): +2

##  Benchmark Results

### Training Results (100K timesteps)

| Metric | Value |
|--------|-------|
| **Training Episodes** | 2,719 |
| **Mean Episode Reward** | 367.5 ± 42.5 |
| **Mean Episode Length** | 50 steps (max) |
| **Training Time** | ~10 minutes (4 parallel envs) |

### Agent Comparison (20 evaluation episodes)

| Agent Type | Avg Reward | Final Rating | Final Revenue | Survival Rate |
|------------|------------|--------------|---------------|---------------|
| **Random** | 162.1 ± 41.3 | 2.99 ± 0.18 | $763 ± $159 | 0% |
| **PPO (trained)** | **870.7 ± 42.5** | **3.03 ± 0.21** | **$25,352 ± $1,243** | **100%** |
| **Improvement** | **+437.3%** | **+1.4%** | **+3,224.4%** | **+100%** |

### What the Agent Learned

Based on evaluation rollouts, the trained agent exhibits these strategies:

1. **Review Response Strategy:**
   - Negative reviews (1-2★): Apologetic responses (~92%)
   - Neutral reviews (3★): Professional responses (~78%)
   - Positive reviews (4-5★): Grateful responses (~87%)

2. **Pricing Strategy:**
   - Maintains stable pricing when rating >3.0
   - Reduces prices when revenue drops
   - Avoids aggressive price increases

3. **Investment Strategy:**
   - Invests in quality when revenue >$1,500
   - Prioritizes survival over growth when struggling
   - Balances reinvestment vs. cash reserves

4. **Marketing Strategy:**
   - Runs promotions during stable periods
   - Avoids marketing when cash-constrained

## Research Applications

This environment enables research on:

1. **Multi-objective RL** - Balance profit vs. reputation vs. customer satisfaction
2. **Long-term Planning** - Investment decisions pay off over multiple timesteps
3. **Human-AI Interaction** - Learning appropriate response strategies
4. **Safe RL** - Test strategies in simulation before real deployment
5. **Sim-to-Real Transfer** - Train in environment with real data distribution
6. **Curriculum Learning** - Progressive difficulty with review complexity
7. **Reward Shaping** - Six-component reward for complex optimization

## 🛠️ Customization

### Modify Environment Parameters

```python
env_config = {
    'max_steps': 100,           # Episode length
    'log_to_db': False,         # Database logging
    'db_host': 'localhost',
    'db_port': 3307,
    'db_user': 'ai_agent_user',
    'db_password': 'your_password',
    'db_name': 'mobile_repair'
}

env = BingleFixItBusinessEnv(config=env_config)
```

### Try Different Algorithms

```python
from stable_baselines3 import PPO, A2C, DQN

# PPO (best performance)
model = PPO("MultiInputPolicy", env)

# A2C (faster training)
model = A2C("MultiInputPolicy", env)

# Note: DQN requires Box action space, use wrapper
```



## Dependencies

- **Python 3.8+**
- **gymnasium** - RL environment interface
- **stable-baselines3** - PPO implementation
- **numpy** - Numerical computations
- **matplotlib** - Visualization
- **mysql-connector-python** - Database integration
- **tensorboard** - Training monitoring

See `requirements.txt` for exact versions.

## Future Enhancements

- [ ] Multi-agent competition (businesses competing)
- [ ] Seasonal demand variations
- [ ] Employee management (hiring/firing)
- [ ] Competitor pricing dynamics
- [ ] Social media reputation system
- [ ] Real-time deployment (agent manages live website)
- [ ] Integration with Yelp/Google Reviews APIs
- [ ] Customer personas with different preferences
- [ ] Economic recession/boom cycles

## Contributing

Contributions welcome! Areas of interest:
- Additional baseline algorithms (A2C, SAC, TD3)
- Reward function improvements  
- Integration with real review APIs
- More realistic customer behavior models
- Hyperparameter optimization studies
- Curriculum learning approaches

## Author

**Omar Alchab**  
📧 Email: alchabomar@gmail.com  
🐙 GitHub: [@alchabom](https://github.com/alchabom)  



---
