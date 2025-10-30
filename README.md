# BingleFixIt: Business Management RL Gym

A realistic reinforcement learning environment for training AI agents to manage a service business. Built on a real PHP/MySQL storefront with actual review mechanics.

##  What This Is

An **OpenAI Gym-compatible environment** where RL agents learn to:
- Respond to customer reviews appropriately
- Balance pricing vs. customer satisfaction
- Make quality investment decisions
- Manage business reputation over time
- Maximize long-term revenue and ratings

## Why 

**The AI industry needs realistic training environments.** Most RL research uses:
- Simple game environments (CartPole, Atari)
- Simulated robotics (MuJoCo)
- Abstract tasks

**Real-world business simulation** 
1. It connects to actual economic outcomes
2. It has real data from a MySQL database
3. It tests multi-objective optimization (revenue + reputation)
4. It demonstrates practical AI applications

## Architecture

```
┌─────────────────────────────────────┐
│   RL Agent         │
│   - PPO, DQN, A3C, etc.             │
└──────────────┬──────────────────────┘
               │ actions
               ↓
┌─────────────────────────────────────┐
│   BingleFixItBusinessGym            │
│   (OpenAI Gym Environment)          │
│   - State: ratings, revenue, etc.   │
│   - Actions: respond, price, invest │
│   - Rewards: business health        │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   MySQL Database                    │
│   - comments (real user reviews)    │
│   - agent_comments (AI responses)   │
│   - Business metrics                │
└─────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   PHP Storefront (Vercel/Localhost) │
│   - Actual website users can visit  │
│   - Real review submission          │
└─────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install gymnasium numpy mysql-connector-python torch

# Clone repo
git clone https://github.com/yourusername/binglefixit-rl-gym
cd binglefixit-rl-gym

# Setup database (see database/schema.sql)
mysql -u root -p < database/schema.sql
```

### Basic Usage

```python
import gymnasium as gym
from binglefixit_gym import BingleFixItBusinessGym

# Create environment
env = BingleFixItBusinessGym(config={})

# Reset
obs, info = env.reset()

# Take actions
for _ in range(100):
    action = {
        'response_type': 1,      # Apologetic response
        'pricing_change': 1,     # Maintain price
        'quality_investment': 1, # Moderate investment
        'run_promotion': 0       # No promotion
    }
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### Train an Agent

```python
from stable_baselines3 import PPO

# Create and wrap environment
env = BingleFixItBusinessGym(config={})

# Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save model
model.save("business_manager_agent")
```

## State Space

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `avg_rating` | float | [1.0, 5.0] | Current business rating |
| `revenue` | float | [0, 100000] | Total revenue ($) |
| `satisfaction` | float | [0, 1] | Customer satisfaction |
| `service_quality` | float | [0, 1] | Quality level |
| `price` | float | [10, 200] | Current service price |
| `pending_reviews` | int | [0, 50] | Reviews needing response |
| `recent_reviews` | string | text | Last 5 reviews |

## Action Space

| Action | Type | Options | Description |
|--------|------|---------|-------------|
| `response_type` | Discrete(4) | ignore/apologetic/professional/grateful | How to respond to reviews |
| `pricing_change` | Discrete(3) | decrease/maintain/increase | Pricing strategy |
| `quality_investment` | Discrete(3) | none/moderate/high | Training & equipment |
| `run_promotion` | Discrete(2) | no/yes | Marketing campaign |

## Reward Structure

The reward function balances multiple objectives:

```python
reward = (
    response_reward +      # Appropriate review responses
    pricing_reward +       # Revenue optimization
    quality_reward +       # Service improvements
    marketing_reward +     # Customer acquisition
    customer_reward +      # Customer behavior
    health_reward         # Long-term business health
)
```


## Baselines

| Agent | Avg Reward | Final Rating | Final Revenue |
|-------|------------|--------------|---------------|
| Random | -12.3 | 2.8/5.0 | $8,500 |
| Greedy (price max) | -8.7 | 2.1/5.0 | $15,200 |
| Quality-first | +15.2 | 4.2/5.0 | $18,900 |
| PPO (trained) | **+23.8** | **4.5/5.0** | **$22,300** |

## Applications

This gym enables research on:

1. **Multi-objective RL** - Balance competing goals (profit vs. reputation)
2. **Long-term planning** - Investment decisions pay off over time
3. **Human-AI interaction** - Appropriate responses to reviews
4. **Sim-to-real transfer** - Train in simulation, deploy to real business
5. **Safe RL** - Can't test aggressive strategies on real business


## Future

- [ ] Add multi-agent competition (multiple businesses)
- [ ] Seasonal demand variations
- [ ] Employee hiring/firing decisions
- [ ] Competitor pricing dynamics
- [ ] Social media reputation system
- [ ] Real-time deployment (agent manages live site)

## Contributing

Contributions welcome! Areas of interest:
- Additional baseline algorithms
- Reward function improvements  
- Integration with real review APIs (Yelp, Google)
- More realistic customer behavior models

Omar Alchab - alchabomar@gmail.com
GitHub: [@alchabom](https://github.com/alchabom)

---