import numpy as np
import pandas as pd
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================================================================
# MODULE 4: MULTI-AGENT REINFORCEMENT LEARNING FOR ADAPTIVE COLLABORATION
# ============================================================================

class Village:
    """Represents a village with resource needs"""
    
    def __init__(self, name, urgency, food_need, medicine_need):
        self.name = name
        self.urgency = urgency
        self.food_need = food_need
        self.medicine_need = medicine_need
        self.food_delivered = 0
        self.medicine_delivered = 0
    
    def get_reward(self, food_qty, medicine_qty):
        """Calculate reward based on need satisfaction"""
        food_satisfaction = min(food_qty / max(self.food_need, 1), 1.0)
        medicine_satisfaction = min(medicine_qty / max(self.medicine_need, 1), 1.0)
        
        # Higher urgency villages give more reward
        urgency_multiplier = {'High': 3.0, 'Medium': 2.0, 'Low': 1.0}[self.urgency]
        
        base_reward = (food_satisfaction + medicine_satisfaction) / 2
        return base_reward * urgency_multiplier * 100
    
    def is_satisfied(self):
        """Check if village needs are met"""
        return (self.food_delivered >= self.food_need and 
                self.medicine_delivered >= self.medicine_need)


class ReliefAgent:
    """Relief agent using Q-learning"""
    
    def __init__(self, agent_id, resource_type, initial_capacity=100):
        self.agent_id = agent_id
        self.resource_type = resource_type  # 'food' or 'medicine'
        self.capacity = initial_capacity
        self.remaining = initial_capacity
        
        # Q-learning parameters
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.3  # Exploration rate
        
        self.total_reward = 0
        self.deliveries = []
    
    def get_state(self, villages, current_location):
        """Get state representation"""
        # State: (current_location, remaining_capacity, high_urgency_villages)
        high_urgency_count = sum(1 for v in villages 
                                if v.urgency == 'High' and not v.is_satisfied())
        
        capacity_level = 'High' if self.remaining > 60 else 'Medium' if self.remaining > 20 else 'Low'
        
        return (current_location, capacity_level, high_urgency_count)
    
    def choose_action(self, state, available_villages):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            # Explore: random village
            return np.random.choice(available_villages)
        else:
            # Exploit: best Q-value
            q_values = {v: self.q_table[state][v] for v in available_villages}
            if not q_values or all(v == 0 for v in q_values.values()):
                return np.random.choice(available_villages)
            return max(q_values, key=q_values.get)
    
    def update_q_value(self, state, action, reward, next_state, next_actions):
        """Q-learning update rule"""
        current_q = self.q_table[state][action]
        
        if next_actions:
            max_next_q = max(self.q_table[next_state][a] for a in next_actions)
        else:
            max_next_q = 0
        
        # Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def deliver(self, village, quantity):
        """Deliver resources to village"""
        delivery_qty = min(quantity, self.remaining)
        
        if self.resource_type == 'food':
            village.food_delivered += delivery_qty
        else:
            village.medicine_delivered += delivery_qty
        
        self.remaining -= delivery_qty
        self.deliveries.append((village.name, delivery_qty))
        
        return delivery_qty


class MultiAgentMDP:
    """Multi-Agent MDP for disaster relief coordination"""
    
    def __init__(self, villages, agents):
        self.villages = villages
        self.agents = agents
        self.episode_rewards = []
        self.coordination_scores = []
    
    def calculate_coordination_penalty(self):
        """Penalize redundant deliveries"""
        penalty = 0
        
        for village in self.villages:
            # Over-delivery penalty
            if village.food_delivered > village.food_need:
                penalty += (village.food_delivered - village.food_need) * 5
            if village.medicine_delivered > village.medicine_need:
                penalty += (village.medicine_delivered - village.medicine_need) * 5
        
        return penalty
    
    def run_episode(self, max_steps=20):
        """Run one episode of multi-agent coordination"""
        # Reset villages
        for village in self.villages:
            village.food_delivered = 0
            village.medicine_delivered = 0
        
        # Reset agents
        for agent in self.agents:
            agent.remaining = agent.capacity
            agent.deliveries = []
        
        episode_reward = 0
        
        for step in range(max_steps):
            all_satisfied = all(v.is_satisfied() for v in self.villages)
            if all_satisfied:
                break
            
            # Each agent takes action
            for agent in self.agents:
                if agent.remaining <= 0:
                    continue
                
                # Get available villages (not fully satisfied)
                available = [v.name for v in self.villages if not v.is_satisfied()]
                
                if not available:
                    break
                
                # Get state and choose action
                current_location = agent.deliveries[-1][0] if agent.deliveries else 'Base'
                state = agent.get_state(self.villages, current_location)
                action = agent.choose_action(state, available)
                
                # Execute delivery
                village = next(v for v in self.villages if v.name == action)
                
                if agent.resource_type == 'food':
                    needed = village.food_need - village.food_delivered
                else:
                    needed = village.medicine_need - village.medicine_delivered
                
                delivery_qty = min(needed, agent.remaining, 30)  # Max 30 per delivery
                agent.deliver(village, delivery_qty)
                
                # Calculate reward
                reward = village.get_reward(village.food_delivered, village.medicine_delivered)
                
                # Get next state
                next_available = [v.name for v in self.villages if not v.is_satisfied()]
                next_state = agent.get_state(self.villages, action)
                
                # Update Q-value
                agent.update_q_value(state, action, reward, next_state, next_available)
                
                agent.total_reward += reward
                episode_reward += reward
        
        # Apply coordination penalty
        penalty = self.calculate_coordination_penalty()
        episode_reward -= penalty
        
        self.episode_rewards.append(episode_reward)
        
        # Calculate coordination score (0-100)
        satisfied_count = sum(1 for v in self.villages if v.is_satisfied())
        coordination_score = (satisfied_count / len(self.villages)) * 100
        self.coordination_scores.append(coordination_score)
        
        return episode_reward, coordination_score
    
    def train(self, num_episodes=100):
        """Train agents over multiple episodes"""
        print("\n" + "="*70)
        print("MULTI-AGENT REINFORCEMENT LEARNING TRAINING")
        print("="*70)
        print(f"\nEpisodes: {num_episodes}")
        print(f"Villages: {len(self.villages)}")
        print(f"Agents: {len(self.agents)}")
        print(f"Total Resources: Food={sum(a.capacity for a in self.agents if a.resource_type=='food')}, "
              f"Medicine={sum(a.capacity for a in self.agents if a.resource_type=='medicine')}")
        print("\nTraining progress:")
        
        for episode in range(num_episodes):
            reward, coord_score = self.run_episode()
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(self.episode_rewards[-20:])
                avg_coord = np.mean(self.coordination_scores[-20:])
                print(f"  Episode {episode+1:3d}: Avg Reward={avg_reward:7.2f}, "
                      f"Coordination={avg_coord:5.1f}%")
        
        print("\n✓ Training complete!")
        
        # Final statistics
        final_avg_reward = np.mean(self.episode_rewards[-20:])
        final_avg_coord = np.mean(self.coordination_scores[-20:])
        
        print(f"\n📊 Final Performance (last 20 episodes):")
        print(f"  Average Reward: {final_avg_reward:.2f}")
        print(f"  Average Coordination: {final_avg_coord:.1f}%")
        print(f"  Max Coordination: {max(self.coordination_scores[-20:]):.1f}%")


def visualize_learning_curves(mdp):
    """Visualize learning progress"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Agent Reinforcement Learning Results', fontsize=14, weight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    ax.plot(mdp.episode_rewards, color='blue', alpha=0.3, label='Episode Reward')
    
    # Smooth with moving average
    window = 10
    if len(mdp.episode_rewards) >= window:
        smoothed = np.convolve(mdp.episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(mdp.episode_rewards)), smoothed, 
               color='red', linewidth=2, label=f'{window}-Episode Average')
    
    ax.set_xlabel('Episode', fontsize=11, weight='bold')
    ax.set_ylabel('Total Reward', fontsize=11, weight='bold')
    ax.set_title('Learning Curve: Episode Rewards', fontsize=11, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Coordination Score
    ax = axes[0, 1]
    ax.plot(mdp.coordination_scores, color='green', alpha=0.3, label='Coordination Score')
    
    if len(mdp.coordination_scores) >= window:
        smoothed = np.convolve(mdp.coordination_scores, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(mdp.coordination_scores)), smoothed,
               color='orange', linewidth=2, label=f'{window}-Episode Average')
    
    ax.set_xlabel('Episode', fontsize=11, weight='bold')
    ax.set_ylabel('Coordination Score (%)', fontsize=11, weight='bold')
    ax.set_title('Coordination Efficiency Over Time', fontsize=11, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 3. Agent Performance Distribution
    ax = axes[1, 0]
    agent_rewards = [agent.total_reward for agent in mdp.agents]
    agent_labels = [f"{agent.agent_id}\n({agent.resource_type})" for agent in mdp.agents]
    colors = ['#FF6B6B' if a.resource_type == 'food' else '#4ECDC4' for a in mdp.agents]
    
    bars = ax.bar(agent_labels, agent_rewards, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Total Reward Collected', fontsize=11, weight='bold')
    ax.set_title('Individual Agent Performance', fontsize=11, weight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, agent_rewards):
        ax.text(bar.get_x() + bar.get_width()/2, val + 50, f'{val:.0f}',
               ha='center', va='bottom', weight='bold', fontsize=9)
    
    # 4. Q-table heatmap for first agent
    ax = axes[1, 1]
    ax.axis('off')
    
    # Extract Q-values
    agent = mdp.agents[0]
    states = list(agent.q_table.keys())[:8]  # First 8 states
    
    if states:
        table_data = [['State', 'Best Action', 'Q-Value']]
        
        for state in states:
            if agent.q_table[state]:
                best_action = max(agent.q_table[state], key=agent.q_table[state].get)
                best_q = agent.q_table[state][best_action]
                state_str = f"{state[0][:10]}, {state[1]}, {state[2]}"
                table_data.append([state_str, best_action[:12], f'{best_q:.2f}'])
        
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.45, 0.35, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows
        for i in range(1, len(table_data)):
            for j in range(3):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title(f'Sample Q-Table Entries ({agent.agent_id})', 
                    fontsize=11, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('MARL_Results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: MARL_Results.png")
    plt.show()


def main():
    print("\n" + "="*80)
    print("MODULE 4: MULTI-AGENT REINFORCEMENT LEARNING")
    print("="*80)
    
    # Create villages
    villages = [
        Village('Rayagada', 'High', food_need=80, medicine_need=60),
        Village('Balangir', 'Medium', food_need=50, medicine_need=40),
        Village('Koraput', 'High', food_need=70, medicine_need=50),
        Village('Bhadrak', 'Low', food_need=30, medicine_need=20),
        Village('Jagatsinghpur', 'Medium', food_need=40, medicine_need=35),
    ]
    
    # Create agents
    agents = [
        ReliefAgent('Agent-F1', 'food', initial_capacity=100),
        ReliefAgent('Agent-F2', 'food', initial_capacity=100),
        ReliefAgent('Agent-M1', 'medicine', initial_capacity=80),
        ReliefAgent('Agent-M2', 'medicine', initial_capacity=80),
    ]
    
    print(f"\n🏘️  Villages: {len(villages)}")
    for v in villages:
        print(f"  • {v.name}: {v.urgency} urgency, needs Food={v.food_need}, Medicine={v.medicine_need}")
    
    print(f"\n🤖 Agents: {len(agents)}")
    for a in agents:
        print(f"  • {a.agent_id}: Type={a.resource_type}, Capacity={a.capacity}")
    
    # Create MDP and train
    mdp = MultiAgentMDP(villages, agents)
    mdp.train(num_episodes=100)
    
    # Visualize results
    print("\n📊 Generating visualizations...")
    visualize_learning_curves(mdp)
    
    # Save results
    results = {
        'episodes': len(mdp.episode_rewards),
        'final_avg_reward': float(np.mean(mdp.episode_rewards[-20:])),
        'final_avg_coordination': float(np.mean(mdp.coordination_scores[-20:])),
        'max_coordination': float(max(mdp.coordination_scores)),
        'agent_performance': [
            {
                'agent_id': a.agent_id,
                'resource_type': a.resource_type,
                'total_reward': float(a.total_reward),
                'deliveries': len(a.deliveries)
            }
            for a in agents
        ]
    }
    
    with open('marl_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 Results saved to: marl_results.json")
    
    print("\n" + "="*80)
    print("✅ MODULE 4: MULTI-AGENT RL COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()