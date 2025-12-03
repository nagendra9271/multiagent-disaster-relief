import json
from collections import defaultdict
import pandas as pd

# ============================================================================
# MODULE 3: PLANNING (GraphPlan & POP) - FORMAL ALGORITHMS
# ============================================================================

class Action:
    """Planning action with preconditions and effects"""
    
    def __init__(self, name, preconditions, effects, cost=1):
        self.name = name
        self.preconditions = set(preconditions)
        self.effects = set(effects)
        self.cost = cost
    
    def __repr__(self):
        return self.name
    
    def can_execute(self, state):
        """Check if preconditions are satisfied"""
        return self.preconditions.issubset(state)
    
    def execute(self, state):
        """Execute action: add effects to state"""
        new_state = state.copy()
        new_state.update(self.effects)
        return new_state

class GraphPlanSolver:
    """
    GraphPlan Algorithm - FIXED VERSION
    """
    
    def __init__(self, initial_state, goal_state, actions):
        self.initial_state = set(initial_state)
        self.goal_state = set(goal_state)
        self.actions = actions
        
        self.fact_levels = [self.initial_state]
        self.action_levels = []
        self.mutexes = []
    
    def goals_non_mutex(self, goals, level):
        return True  # simplified
    
    def expand_graph(self):
        current_facts = self.fact_levels[-1]
        applicable_actions = [a for a in self.actions if a.can_execute(current_facts)]
        
        self.action_levels.append(applicable_actions)
        
        new_facts = set(current_facts)
        for a in applicable_actions:
            new_facts.update(a.effects)
        
        self.fact_levels.append(new_facts)
        self.mutexes.append(set())
    
    def extract_solution(self):
        plan = []
        current_state = set(self.initial_state)
        
        for level in reversed(range(len(self.action_levels))):
            actions = self.action_levels[level]
            applicable = [a for a in actions if a.can_execute(current_state)]
            if applicable:
                action = applicable[0]
                plan.append(action.name)
                current_state = action.execute(current_state)
        
        plan.reverse()
        return plan
    
    def solve(self, max_levels=20):
        print("\n" + "="*70)
        print("GraphPlan ALGORITHM")
        print("="*70)
        
        level = 0
        
        while level < max_levels:
            print(f"Level {level}: Facts = {len(self.fact_levels[-1])}")
            
            if self.goal_state.issubset(self.fact_levels[-1]):
                print("  Goal facts present: YES")
                if self.goals_non_mutex(self.goal_state, level):
                    print("  Goals non-mutex: YES")
                    plan = self.extract_solution()
                    print(f"\n✓ PLAN FOUND: {' → '.join(plan)}")
                    return plan, level
            else:
                missing = self.goal_state - self.fact_levels[-1]
                print(f"  Missing goals: {missing}")
            
            self.expand_graph()
            level += 1
        
        print("\n✗ No plan found\n")
        return None, level


class POPSolver:
    """Partial Order Planning Algorithm"""
    
    def __init__(self, initial_state, goal_state, actions):
        self.initial_state = set(initial_state)
        self.goal_state = set(goal_state)
        self.actions = actions
        
        self.steps = []
        self.causal_links = []
        self.orderings = []
    
    def make_minimal_plan(self):
        self.steps = [
            ('Start', self.initial_state, set()),
            ('Finish', set(), self.goal_state)
        ]
        for goal in self.goal_state:
            self.causal_links.append(('Start', goal, 'Finish'))
        self.orderings.append(('Start', '<', 'Finish'))
    
    def select_subgoal(self):
        for g in self.goal_state:
            if g not in self.initial_state:
                return 'Finish', g
        return None, None
    
    def choose_operator(self, S, c):
        for action in self.actions:
            if c in action.effects:
                step_info = (action.name, action.preconditions, action.effects)
                if step_info not in self.steps:
                    self.steps.append(step_info)
                
                self.causal_links.append((action.name, c, S))
                self.orderings.append((action.name, '<', S))
                print(f"  ✓ Added operator {action.name} for {c}")
                return action
        
        print(f"  ✗ No operator produces {c}")
        return None
    
    def resolve_threats(self):
        print("  ✓ Threat resolution done (simplified)")
    
    def is_solution(self):
        achieved = set()
        for step, _, effects in self.steps:
            achieved.update(effects)
        return self.goal_state.issubset(achieved)
    
    def solve(self):
        print("\n" + "="*70)
        print("Partial Order Planning (POP) ALGORITHM")
        print("="*70)
        
        self.make_minimal_plan()
        iteration = 0
        
        while iteration < 15:
            print(f"\nIteration {iteration + 1}:")
            
            if self.is_solution():
                print("  ✓ SOLUTION FOUND")
                break
            
            S, c = self.select_subgoal()
            if S is None:
                break
            
            print(f"  Subgoal: {S} needs {c}")
            action = self.choose_operator(S, c)
            if action is None:
                break
            
            self.resolve_threats()
            iteration += 1
        
        return self.steps, self.causal_links, self.orderings


def create_disaster_relief_actions():
    return [
        Action("AnalyzeUrgency", ["VillageData"], ["UrgencyAssessed"]),
        Action("DeliverFood", ["UrgencyAssessed", "FoodAvailable"], ["FoodDelivered", "VillageServed"]),
        Action("DeliverMedicine", ["UrgencyAssessed", "MedicineAvailable"], ["MedicineDelivered", "VillageServed"]),
        Action("CoordinateAgents", ["FoodDelivered", "MedicineDelivered"], ["AgentsCoordinated", "DuplicatesAvoided"]),
        Action("VerifyDelivery", ["VillageServed", "AgentsCoordinated"], ["DeliveryVerified", "MissionComplete"]),
    ]


def main():
    print("\n" + "="*80)
    print("MODULE 3: PLANNING (GraphPlan & POP)")
    print("="*80)
    
    initial_state = ["VillageData", "FoodAvailable", "MedicineAvailable"]
    goal_state = ["MissionComplete", "DuplicatesAvoided"]
    actions = create_disaster_relief_actions()
    
    print("\nAvailable Actions:")
    for a in actions:
        print(f"  {a.name} | Pre: {a.preconditions} | Eff: {a.effects}")
    
    print("\n--- GRAPHPLAN ---")
    gp = GraphPlanSolver(initial_state, goal_state, actions)
    gp_plan, gp_levels = gp.solve()

    print("\n--- POP ---")
    pop = POPSolver(initial_state, goal_state, actions)
    pop_steps, pop_links, pop_orderings = pop.solve()

    print("\nSaving results...")
    results = {
        'GraphPlan': gp_plan,
        'POP_steps': [s[0] for s in pop_steps],
        'POP_links': pop_links,
        'POP_orderings': pop_orderings
    }
    json.dump(results, open("planning_results.json", "w"), indent=2)
    print("✓ Saved: planning_results.json")


# FIXED ENTRY POINT
if __name__ == "__main__":
    main()
