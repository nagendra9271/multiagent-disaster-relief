import json
import pandas as pd
from datetime import datetime
import re

# ============================================================================
# MODULE 5: LLM-BASED NEGOTIATION / COORDINATION DIALOGUE GENERATION
# ============================================================================

class NegotiationAgent:
    """Agent with personality and negotiation strategy"""
    
    def __init__(self, agent_id, organization, personality, resource_type, capacity):
        self.agent_id = agent_id
        self.organization = organization
        self.personality = personality  # 'cooperative', 'assertive', 'analytical'
        self.resource_type = resource_type
        self.capacity = capacity
        self.assigned_villages = []
        self.commitments = []
    
    def generate_proposal(self, villages, other_agents):
        """Generate initial proposal based on personality"""
        
        # Sort villages by urgency
        high_urgency = [v for v in villages if v['urgency'] == 'High']
        medium_urgency = [v for v in villages if v['urgency'] == 'Medium']
        
        if self.personality == 'cooperative':
            # Cooperate: focus on high urgency, willing to share
            proposal = {
                'agent': self.agent_id,
                'stance': 'cooperative',
                'message': f"I suggest we prioritize high-urgency villages. I can cover {', '.join([v['name'] for v in high_urgency[:2]])} with {self.resource_type}. What areas can your teams support?",
                'assigned': high_urgency[:2],
                'tone': 'collaborative'
            }
        
        elif self.personality == 'assertive':
            # Assertive: claim territory confidently
            proposal = {
                'agent': self.agent_id,
                'stance': 'assertive',
                'message': f"My team from {self.organization} will handle {', '.join([v['name'] for v in high_urgency])}. We have the capacity and expertise for rapid {self.resource_type} delivery.",
                'assigned': high_urgency,
                'tone': 'confident'
            }
        
        else:  # analytical
            # Analytical: data-driven approach
            total_need = sum(v['food_need'] + v['medicine_need'] for v in villages)
            proposal = {
                'agent': self.agent_id,
                'stance': 'analytical',
                'message': f"Based on capacity analysis, total need is {total_need} units. Given our {self.capacity} {self.resource_type} capacity, I propose an allocation by distance and urgency. Suggest using optimization algorithms.",
                'assigned': high_urgency[:1] + medium_urgency[:1],
                'tone': 'logical'
            }
        
        return proposal
    
    def respond_to_conflict(self, conflict_village, competing_agent):
        """Generate response when there's resource conflict"""
        
        if self.personality == 'cooperative':
            response = {
                'agent': self.agent_id,
                'action': 'yield',
                'message': f"I see {competing_agent} is also targeting {conflict_village}. To avoid duplication, I'll redirect to another location. Coordination is key in disaster response.",
                'tone': 'accommodating'
            }
        
        elif self.personality == 'assertive':
            response = {
                'agent': self.agent_id,
                'action': 'negotiate',
                'message': f"We're both targeting {conflict_village}. My team has specialized {self.resource_type} equipment already en route. Perhaps you can focus on adjacent areas?",
                'tone': 'firm'
            }
        
        else:  # analytical
            response = {
                'agent': self.agent_id,
                'action': 'analyze',
                'message': f"Conflict detected at {conflict_village}. Let's compute resource sufficiency: if both deliver, we'll have surplus. I propose: whichever team has shorter ETA delivers, other redirects to next priority village.",
                'tone': 'rational'
            }
        
        return response
    
    def generate_final_commitment(self, assigned_villages):
        """Generate final commitment message"""
        self.assigned_villages = assigned_villages
        
        village_names = ', '.join([v['name'] for v in assigned_villages])
        
        message = {
            'agent': self.agent_id,
            'villages': [v['name'] for v in assigned_villages],
            'message': f"Confirmed: {self.organization} will deliver {self.resource_type} to {village_names}. ETA within 24 hours. Will update coordination center upon arrival.",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return message


class DialogueCoordinator:
    """Coordinates multi-agent negotiation dialogue"""
    
    def __init__(self, agents, villages):
        self.agents = agents
        self.villages = villages
        self.dialogue_log = []
        self.conflicts = []
        self.final_allocation = {}
    
    def simulate_negotiation(self):
        """Simulate multi-turn negotiation dialogue"""
        
        print("\n" + "="*70)
        print("MULTI-AGENT NEGOTIATION SIMULATION")
        print("="*70)
        print(f"\nParticipants: {len(self.agents)} relief organizations")
        print(f"Target Villages: {len(self.villages)}")
        print(f"Negotiation Goal: Avoid duplication, maximize coverage\n")
        
        print("━" * 70)
        print("ROUND 1: INITIAL PROPOSALS")
        print("━" * 70)
        
        proposals = []
        
        # Generate proposals
        for agent in self.agents:
            proposal = agent.generate_proposal(self.villages, self.agents)
            proposals.append(proposal)
            
            print(f"\n[{agent.agent_id}] ({agent.organization}, {agent.personality}):")
            print(f"  \"{proposal['message']}\"")
            
            self.dialogue_log.append({
                'round': 1,
                'agent': agent.agent_id,
                'type': 'proposal',
                'content': proposal['message']
            })
        
        # Detect conflicts
        print("\n" + "━" * 70)
        print("CONFLICT DETECTION")
        print("━" * 70)
        
        assignment_count = {}
        for proposal in proposals:
            for village in proposal['assigned']:
                village_name = village['name']
                if village_name not in assignment_count:
                    assignment_count[village_name] = []
                assignment_count[village_name].append(proposal['agent'])
        
        conflicts_found = {v: agents for v, agents in assignment_count.items() 
                          if len(agents) > 1}
        
        if conflicts_found:
            print(f"\n⚠️  Conflicts detected: {len(conflicts_found)} villages")
            for village, competing_agents in conflicts_found.items():
                print(f"  • {village}: {', '.join(competing_agents)}")
                self.conflicts.append({'village': village, 'agents': competing_agents})
        else:
            print("\n✓ No conflicts detected")
        
        # Conflict resolution
        if conflicts_found:
            print("\n" + "━" * 70)
            print("ROUND 2: CONFLICT RESOLUTION")
            print("━" * 70)
            
            for conflict in self.conflicts:
                village = conflict['village']
                competing_agents = conflict['agents']
                
                print(f"\n📍 Resolving conflict at {village}:")
                
                for agent_id in competing_agents:
                    agent = next(a for a in self.agents if a.agent_id == agent_id)
                    other_agent = [a for a in competing_agents if a != agent_id][0]
                    
                    response = agent.respond_to_conflict(village, other_agent)
                    
                    print(f"\n[{agent.agent_id}]:")
                    print(f"  \"{response['message']}\"")
                    
                    self.dialogue_log.append({
                        'round': 2,
                        'agent': agent.agent_id,
                        'type': 'conflict_resolution',
                        'content': response['message'],
                        'action': response['action']
                    })
        
        # Final allocation
        print("\n" + "━" * 70)
        print("ROUND 3: FINAL COMMITMENTS")
        print("━" * 70)
        
        # Simple allocation: distribute villages round-robin
        allocated_villages = {agent.agent_id: [] for agent in self.agents}
        
        sorted_villages = sorted(self.villages, 
                                key=lambda v: {'High': 3, 'Medium': 2, 'Low': 1}[v['urgency']], 
                                reverse=True)
        
        for i, village in enumerate(sorted_villages):
            agent = self.agents[i % len(self.agents)]
            
            # Match resource type
            if village['food_need'] > 0 and agent.resource_type == 'food':
                allocated_villages[agent.agent_id].append(village)
            elif village['medicine_need'] > 0 and agent.resource_type == 'medicine':
                allocated_villages[agent.agent_id].append(village)
            else:
                # Assign to any available agent
                allocated_villages[agent.agent_id].append(village)
        
        for agent in self.agents:
            commitment = agent.generate_final_commitment(allocated_villages[agent.agent_id])
            
            print(f"\n[{agent.agent_id}] - FINAL COMMITMENT:")
            print(f"  \"{commitment['message']}\"")
            
            self.dialogue_log.append({
                'round': 3,
                'agent': agent.agent_id,
                'type': 'commitment',
                'content': commitment['message'],
                'villages': commitment['villages']
            })
            
            self.final_allocation[agent.agent_id] = allocated_villages[agent.agent_id]
        
        print("\n" + "="*70)
        print("NEGOTIATION COMPLETE")
        print("="*70)
        
        self.generate_summary()
    
    def generate_summary(self):
        """Generate negotiation summary"""
        
        print("\n" + "="*70)
        print("📋 NEGOTIATION SUMMARY")
        print("="*70)
        
        total_villages = len(self.villages)
        allocated_count = sum(len(villages) for villages in self.final_allocation.values())
        coverage = (allocated_count / total_villages) * 100 if total_villages > 0 else 0
        
        print(f"\n✅ Villages Covered: {allocated_count}/{total_villages} ({coverage:.1f}%)")
        print(f"🤝 Total Conflicts Resolved: {len(self.conflicts)}")
        print(f"💬 Dialogue Turns: {len(self.dialogue_log)}")
        
        print(f"\n📊 Final Allocation:")
        for agent_id, villages in self.final_allocation.items():
            agent = next(a for a in self.agents if a.agent_id == agent_id)
            village_names = [v['name'] for v in villages]
            print(f"  • {agent_id} ({agent.organization}): {', '.join(village_names) if village_names else 'None'}")
        
        # Personality effectiveness
        print(f"\n🎭 Negotiation Style Analysis:")
        for agent in self.agents:
            assigned_count = len(self.final_allocation[agent.agent_id])
            print(f"  • {agent.personality.capitalize()} ({agent.agent_id}): {assigned_count} villages assigned")


def analyze_dialogue_quality(dialogue_log):
    """Analyze quality metrics of generated dialogue"""
    
    print("\n" + "="*70)
    print("📈 DIALOGUE QUALITY ANALYSIS")
    print("="*70)
    
    # Count messages by type
    type_counts = {}
    for entry in dialogue_log:
        msg_type = entry['type']
        type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
    
    print(f"\n📝 Message Distribution:")
    for msg_type, count in type_counts.items():
        print(f"  • {msg_type.replace('_', ' ').title()}: {count}")
    
    # Average message length
    avg_length = sum(len(entry['content'].split()) for entry in dialogue_log) / len(dialogue_log)
    print(f"\n📏 Average Message Length: {avg_length:.1f} words")
    
    # Tone analysis
    tones = []
    for entry in dialogue_log:
        content = entry['content'].lower()
        if any(word in content for word in ['suggest', 'propose', 'perhaps', 'could']):
            tones.append('polite')
        elif any(word in content for word in ['will', 'must', 'need to', 'have to']):
            tones.append('directive')
        else:
            tones.append('neutral')
    
    tone_dist = {tone: tones.count(tone) / len(tones) * 100 for tone in set(tones)}
    print(f"\n🎵 Tone Distribution:")
    for tone, pct in tone_dist.items():
        print(f"  • {tone.capitalize()}: {pct:.1f}%")
    
    # Cooperation indicators
    cooperation_words = ['cooperate', 'together', 'collaborate', 'coordinate', 'share']
    cooperation_count = sum(1 for entry in dialogue_log 
                           if any(word in entry['content'].lower() 
                           for word in cooperation_words))
    
    cooperation_rate = (cooperation_count / len(dialogue_log)) * 100
    print(f"\n🤝 Cooperation Indicators: {cooperation_rate:.1f}% of messages")
    
    return {
        'total_messages': len(dialogue_log),
        'avg_message_length': avg_length,
        'cooperation_rate': cooperation_rate,
        'tone_distribution': tone_dist
    }


def main():
    print("\n" + "="*80)
    print("MODULE 5: LLM-BASED NEGOTIATION & DIALOGUE GENERATION")
    print("="*80)
    
    # Create relief agents with different personalities
    agents = [
        NegotiationAgent('Agent-A', 'Red Cross India', 'cooperative', 'food', 150),
        NegotiationAgent('Agent-B', 'WHO Emergency Response', 'analytical', 'medicine', 120),
        NegotiationAgent('Agent-C', 'NDRF Odisha', 'assertive', 'food', 180),
        NegotiationAgent('Agent-D', 'MSF Medical Team', 'cooperative', 'medicine', 100),
    ]
    
    # Village data
    villages = [
        {'name': 'Rayagada', 'urgency': 'High', 'food_need': 80, 'medicine_need': 60},
        {'name': 'Balangir', 'urgency': 'Medium', 'food_need': 50, 'medicine_need': 40},
        {'name': 'Koraput', 'urgency': 'High', 'food_need': 70, 'medicine_need': 50},
        {'name': 'Bhadrak', 'urgency': 'Low', 'food_need': 30, 'medicine_need': 20},
        {'name': 'Jagatsinghpur', 'urgency': 'Medium', 'food_need': 40, 'medicine_need': 35},
    ]
    
    print(f"\n🤖 Negotiation Participants:")
    for agent in agents:
        print(f"  • {agent.agent_id}: {agent.organization}")
        print(f"    - Personality: {agent.personality}")
        print(f"    - Resource: {agent.resource_type} (capacity: {agent.capacity})")
    
    # Run negotiation
    coordinator = DialogueCoordinator(agents, villages)
    coordinator.simulate_negotiation()
    
    # Analyze dialogue
    quality_metrics = analyze_dialogue_quality(coordinator.dialogue_log)
    
    # Save results
    results = {
        'agents': [
            {
                'agent_id': a.agent_id,
                'organization': a.organization,
                'personality': a.personality,
                'resource_type': a.resource_type
            }
            for a in agents
        ],
        'dialogue_log': coordinator.dialogue_log,
        'final_allocation': {
            agent_id: [v['name'] for v in villages]
            for agent_id, villages in coordinator.final_allocation.items()
        },
        'conflicts_resolved': len(coordinator.conflicts),
        'quality_metrics': quality_metrics
    }
    
    with open('negotiation_dialogue.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary dataframe
    summary_data = []
    for agent in agents:
        allocated = coordinator.final_allocation[agent.agent_id]
        summary_data.append({
            'Agent': agent.agent_id,
            'Organization': agent.organization,
            'Personality': agent.personality,
            'Resource Type': agent.resource_type,
            'Villages Assigned': len(allocated),
            'Village Names': ', '.join([v['name'] for v in allocated]) if allocated else 'None'
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv('negotiation_results.csv', index=False)
    
    print(f"\n💾 Results saved:")
    print(f"  • negotiation_dialogue.json")
    print(f"  • negotiation_results.csv")
    
    print("\n" + "="*80)
    print("✅ MODULE 5: LLM NEGOTIATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()