# 🛰️ Multi-Agent AI System for Disaster Relief Resource Allocation

### _A Complete AI Framework for Probabilistic Assessment, Planning, Search, MARL, Negotiation, & Integration_

## Overview

This repository contains a **complete multi-agent AI system** designed for **disaster relief coordination** after a cyclone in coastal Odisha.
Multiple autonomous agents (Government, NGOs, Local Volunteers) cooperate to:

- Assess urgency using **Bayesian reasoning**
- Optimize routes using **UCS & A\***
- Generate coordinated plans using **GraphPlan & POP**
- Learn adaptive collaboration using **Multi-Agent Reinforcement Learning**
- Conduct LLM-based negotiation for conflict-free allocation
- Fully integrate all modules into a complete end-to-end system

This project corresponds to **6 major modules**, each satisfying the academic project requirements.

---

# Repository Structure

```
├── module1_bayesian.py
├── module1_visualize.py
├── module2_search.py
├── module2_visualize.py
├── module3_planning.py
├── module4_marl.py
├── module5_llm_dialogue.py
├── module6_integration.py
├── run_all_modules.py
└── README.md   (this file)
```

Each module is standalone and also connects with the integrated system.

---

# Module Summary

## **Module 1 — Bayesian Network for Urgency Assessment**

📄 _File:_ `module1_bayesian.py`
_Visualization:_ `module1_visualize.py`

This module models how each relief agent maintains beliefs about village conditions.

### **Key Features**

- Probabilistic modelling of **Population Density**, **Water Level**, **Medical Signals**, **Prev Deliveries**, etc.
- 4+ Evidence nodes & intermediate nodes.
- Complete **CPT generation** and **posterior inference**.
- Outputs urgency probabilities per village.
- Visualization:

  - Bayesian Network structure
  - Inference workflow
  - d-separation analysis

---

## **Module 2 — Search-Based Route & Task Allocation**

_File:_ `module2_search.py`
_Visualization:_ `module2_visualize.py`

A real geographical network of Odisha villages is modelled using road & boat accessibility.

### **Implemented Algorithms**

- **Uniform Cost Search (UCS)**
- **A\* Search** with an admissible geographic heuristic

### **Outputs**

- Optimal path, cost, fuel used, time required
- Comparison of UCS vs A\*
- A\* achieves faster convergence with fewer node expansions

---

## **Module 3 — Automated Planning (GraphPlan + POP)**

📄 _File:_ `module3_planning.py`

Simulates multi-agent coordination using classical planning.

### **Implemented**

- Precondition–effect based operators:

  - DeliverFood, DeliverMedicine, RequestSupport, AvoidRedundancy…

- GraphPlan with:

  - Fact levels
  - Action levels
  - Mutex relations

- POP (Partial Order Planning)

  - Causal links
  - Threat resolution

### **Output**

A coordinated plan from “Start” to “MissionComplete”.

---

## **Module 4 — Multi-Agent Reinforcement Learning**

📄 _File:_ `module4_marl.py`

Agents learn adaptive cooperation to maximize resource distribution efficiency.

### **Implemented**

- Q-Learning for each agent
- State space: `(location, capacity level, #high urgency villages)`
- Reward:

  - Higher for urgent villages
  - Penalty for duplicate deliveries

- Tracks:

  - Episode rewards
  - Coordination percentage
  - Learning curve graphs

---

## **Module 5 — LLM-Based Negotiation System**

📄 _File:_ `module5_llm_dialogue.py`

Agents negotiate and resolve conflicts using natural-language reasoning.

### **Includes**

- Personality-based behaviours (cooperative, analytical, assertive)
- Multi-turn dialogue simulation
- Conflict detection & resolution
- Final commitment generation per organization

---

## **Module 6 — Full System Integration**

📄 _File:_ `module6_integration.py`

Runs all modules together:

1. Bayesian urgency assessment
2. A\* route optimization
3. GraphPlan coordination
4. MARL training
5. LLM-based negotiation
6. Outputs final, conflict-free resource allocation

---

# Running the Entire System

### **Run all 6 modules sequentially**

```bash
python run_all_modules.py
```

This executes inference → search → planning → RL → negotiation → integration.

### **Run modules individually**

```bash
python module1_bayesian.py
python module2_search.py
python module3_planning.py
python module4_marl.py
python module5_llm_dialogue.py
python module6_integration.py
```

---

# Installation & Requirements

## **1. Clone the repository**

```bash
git clone https://github.com/nagendra9271/multiagent-disaster-relief.git
cd multiagent-disaster-relief
```

## **2. Install dependencies**

```bash
pip install -r requirements.txt
```

## **Requirements (major libraries)**

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Heapq
- JSON
- Datetime

---

# Example Outputs

### ✔ Bayesian Network

- Urgency Posterior: `High: 0.85`, `Medium: 0.12`, `Low: 0.03`

### ✔ A\* Route Comparison

- A\*: 47% fewer nodes than UCS
- A\*: 2.3× faster
- Same optimal path cost

### ✔ GraphPlan Plan

```
AnalyzeUrgency → DeliverFood → DeliverMedicine → VerifyDelivery → MissionComplete
```

### ✔ MARL Final Score

- Final coordination: **87%**
- Rewards stabilize after ~20 episodes

### ✔ Negotiation Example

Agents resolve conflicts and finalize which villages each will serve.

---

# Test Cases

Each module includes non-trivial test cases:

- Bayesian inference: High flood + No prior delivery
- A\*: Blocked paths + fuel constraints
- POP: multiple causal link threats
- MARL: varying urgency across episodes
- Negotiation: multi-agent conflict detection

---

---

# Final Deliverables (Included in this Repo)

✔ Complete Code
✔ Module-wise working outputs
✔ Visualizations (PNG)
✔ Integrated system log
✔ PDF report-ready content

---

---

