import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_network():
    """Draw Bayesian Network Structure - SIMPLE"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'BAYESIAN NETWORK STRUCTURE', fontsize=16, weight='bold', ha='center')
    
    # Define positions
    nodes = {
        'PopDensity': (1, 7),
        'PrevDelivery': (2.5, 7),
        'MedicalSignals': (6, 7),
        'DiseaseOutbreak': (7.5, 7),
        'WaterLevel': (4, 7),
        
        'NeedForSupplies': (1.75, 5),
        'MedicalEmergency': (6.75, 5),
        'RoadAccess': (4, 5),
        
        'Urgency': (4, 2)
    }
    
    # Colors
    colors = {
        'root': '#FFD700',      # Gold
        'intermediate': '#87CEEB',  # Sky blue
        'target': '#FFB6C6'     # Pink
    }
    
    # Draw nodes
    for node, (x, y) in nodes.items():
        if node in ['PopDensity', 'PrevDelivery', 'MedicalSignals', 'DiseaseOutbreak', 'WaterLevel']:
            color = colors['root']
        elif node in ['NeedForSupplies', 'MedicalEmergency', 'RoadAccess']:
            color = colors['intermediate']
        else:
            color = colors['target']
        
        circle = patches.FancyBboxPatch((x-0.3, y-0.2), 0.6, 0.4,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', fontsize=9, weight='bold')
    
    # Draw edges
    edges = [
        ((1, 6.8), (1.75, 5.2)),          # PopDensity -> NeedForSupplies
        ((2.5, 6.8), (1.75, 5.2)),        # PrevDelivery -> NeedForSupplies
        ((6, 6.8), (6.75, 5.2)),          # MedicalSignals -> MedicalEmergency
        ((7.5, 6.8), (6.75, 5.2)),        # DiseaseOutbreak -> MedicalEmergency
        ((4, 6.8), (4, 5.2)),             # WaterLevel -> RoadAccess
        ((1.75, 4.8), (4, 2.2)),          # NeedForSupplies -> Urgency
        ((6.75, 4.8), (4, 2.2)),          # MedicalEmergency -> Urgency
        ((4, 4.8), (4, 2.2)),             # RoadAccess -> Urgency
    ]
    
    for start, end in edges:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.15, head_length=0.15, fc='darkblue', ec='darkblue', linewidth=2)
    
    # Legend
    ax.text(0.5, 0.5, '● Gold = Root Nodes (Evidence)', fontsize=10, color='gold', weight='bold')
    ax.text(0.5, 0.1, '● Blue = Intermediate Nodes', fontsize=10, color='skyblue', weight='bold')
    ax.text(5.5, 0.5, '● Pink = Target Node (Query)', fontsize=10, color='pink', weight='bold')
    
    plt.tight_layout()
    plt.savefig('BN_Structure.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: BN_Structure.png")
    plt.show()


def draw_inference():
    """Draw Inference Process - SIMPLE"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('BAYESIAN INFERENCE PROCESS', fontsize=14, weight='bold')
    
    # Step 1
    ax = axes[0, 0]
    ax.set_title('Step 1: Input Evidence', weight='bold', fontsize=11)
    ax.axis('off')
    text1 = """
PopDensity: High
PrevDelivery: No
MedicalSignals: Present
DiseaseOutbreak: Absent
WaterLevel: High
    """
    ax.text(0.5, 0.5, text1, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Step 2
    ax = axes[0, 1]
    ax.set_title('Step 2: Compute Intermediates', weight='bold', fontsize=11)
    ax.axis('off')
    text2 = """
P(NeedForSupplies|High,No):
  High: 0.85, Medium: 0.12

P(RoadAccess|High):
  Accessible: 0.25, Blocked: 0.75

P(MedicalEmergency|Present,Absent):
  High: 0.55, Medium: 0.35
    """
    ax.text(0.5, 0.5, text2, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            family='monospace')
    
    # Step 3
    ax = axes[1, 0]
    ax.set_title('Step 3: Marginalize', weight='bold', fontsize=11)
    ax.axis('off')
    text3 = """
Sum over all combinations:

P(Urgency) = ∑ P(Need) × 
              P(Medical) × 
              P(Road) × 
              P(Urgency|parents)
    """
    ax.text(0.5, 0.5, text3, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Step 4 - Results
    ax = axes[1, 1]
    ax.set_title('Step 4: Final Result', weight='bold', fontsize=11)
    ax.axis('off')
    
    # Bar chart
    states = ['High', 'Medium', 'Low']
    values = [0.85, 0.12, 0.03]
    colors_bar = ['red', 'orange', 'green']
    
    ax_bar = ax.bar(states, values, color=colors_bar, edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    
    for bar, val in zip(ax_bar, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', weight='bold')
    
    ax.text(0.5, -0.3, 'Result: HIGH URGENCY (0.85)', ha='center', transform=ax.transAxes,
           fontsize=11, weight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('Inference_Process.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Inference_Process.png")
    plt.show()


def draw_dseparation():
    """D-Separation Explanation - SIMPLE"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, 'D-SEPARATION ANALYSIS', fontsize=14, weight='bold', ha='center')
    
    # Query 1
    ax.text(5, 8.5, 'Q1: PopDensity ⊥ MedicalSignals?', fontsize=11, weight='bold', ha='center')
    text1 = """
Path: PopDensity → NeedForSupplies → Urgency ← MedicalEmergency ← MedicalSignals

Urgency is COLLIDER (v-structure - multiple parents converge)

Without conditioning on Urgency:
→ PATH BLOCKED at collider
→ Variables are INDEPENDENT ✓

Result: PopDensity ⊥ MedicalSignals (unconditionally)
    """
    ax.text(5, 6.8, text1, fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), family='monospace')
    
    # Query 2
    ax.text(5, 3.5, 'Q2: WaterLevel ⊥ PopDensity?', fontsize=11, weight='bold', ha='center')
    text2 = """
Path: WaterLevel → RoadAccess → Urgency ← NeedForSupplies ← PopDensity

Urgency is COLLIDER

Without conditioning on Urgency:
→ PATH BLOCKED at collider
→ Variables are INDEPENDENT ✓

Result: WaterLevel ⊥ PopDensity (unconditionally)

KEY: When we OBSERVE Urgency (condition on it), these become DEPENDENT!
    """
    ax.text(5, 1.8, text2, fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5), family='monospace')
    
    plt.tight_layout()
    plt.savefig('D_Separation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: D_Separation.png")
    plt.show()


if __name__ == "__main__":   # FIXED
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    print("\n1. Drawing Network Structure...")
    draw_network()
    
    print("\n2. Drawing Inference Process...")
    draw_inference()
    
    print("\n3. Drawing D-Separation Analysis...")
    draw_dseparation()
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS DONE")
    print("="*60 + "\n")
