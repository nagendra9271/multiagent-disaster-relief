import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# 1. VISUALIZE NETWORK (FULLY FIXED)
# ============================================================

def visualize_network():
    """Visualize the village network"""

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Village coordinates: (lat, lon)
    villages = {
        'Rayagada': (19.17, 83.42),
        'Balangir': (20.66, 83.47),
        'Koraput': (18.82, 84.08),
        'Bhadrak': (20.82, 86.51),
        'Jagatsinghpur': (20.17, 86.41),
        'Cuttack': (20.46, 85.88),
        'Puri': (19.81, 85.83),
        'Kendrapara': (20.51, 87.03)
    }

    # Edges with access type
    edges = [
        ('Rayagada', 'Balangir', 'road'),
        ('Balangir', 'Cuttack', 'road'),
        ('Cuttack', 'Bhadrak', 'road'),
        ('Bhadrak', 'Jagatsinghpur', 'road'),
        ('Jagatsinghpur', 'Puri', 'road'),
        ('Puri', 'Kendrapara', 'boat'),
        ('Kendrapara', 'Cuttack', 'road'),
        ('Koraput', 'Rayagada', 'road'),
        ('Rayagada', 'Cuttack', 'road'),
        ('Balangir', 'Koraput', 'boat'),
    ]

    # Draw edges (corrected coord order: lon = x, lat = y)
    for v1, v2, etype in edges:
        lat1, lon1 = villages[v1]
        lat2, lon2 = villages[v2]

        style = '-' if etype == 'road' else '--'
        color = 'blue' if etype == 'road' else 'green'

        ax.plot([lon1, lon2], [lat1, lat2], style, color=color, linewidth=2, alpha=0.6)

    # Draw village nodes
    for village, (lat, lon) in villages.items():
        ax.scatter(lon, lat, s=400, c='red', edgecolors='black', linewidth=2, zorder=5)
        ax.text(lon, lat - 0.12, village, fontsize=8, ha='center', weight='bold')

    # Example highlighted path
    path = ['Rayagada', 'Balangir', 'Cuttack', 'Puri']
    for i in range(len(path) - 1):
        v1, v2 = path[i], path[i + 1]
        lat1, lon1 = villages[v1]
        lat2, lon2 = villages[v2]
        ax.plot([lon1, lon2], [lat1, lat2], '-', color='orange', linewidth=4, alpha=0.9, zorder=4)

    # Auto-scale view
    ax.relim()
    ax.autoscale()

    ax.set_xlabel('Longitude', fontsize=12, weight='bold')
    ax.set_ylabel('Latitude', fontsize=12, weight='bold')
    ax.set_title('Odisha Village Network - Disaster Relief Routes', fontsize=14, weight='bold')

    # LEGEND (fully fixed)
    ax.legend(
        handles=[
            mpatches.Patch(color='blue'),
            mpatches.Patch(color='green'),
            mpatches.Patch(color='orange')
        ],
        labels=['Road Access', 'Boat Access', 'Example Path'],
        loc='upper left',
        fontsize=10
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06)

    plt.savefig('Village_Network.png', dpi=300)
    print("✓ Saved: Village_Network.png")
    plt.show()


# ============================================================
# 2. VISUALIZE COMPARISON (A* vs UCS)
# ============================================================

def visualize_comparison():
    """Visualize algorithm comparison"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300, constrained_layout=True)
    fig.suptitle('UCS vs A* Algorithm Comparison', fontsize=14, weight='bold')

    algorithms = ['UCS', 'A*']
    colors = ['#FF6B6B', '#4ECB71']

    # 1. Nodes Expanded
    ax = axes[0][0]
    nodes = [15, 8]
    bars1 = ax.bar(algorithms, nodes, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Nodes Expanded', fontsize=11, weight='bold')
    ax.set_title('Node Comparison')

    for bar, val in zip(bars1, nodes):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, str(val), ha='center', weight='bold')

    ax.text(0.5, 16, '47% fewer nodes', ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 2. Computation Time
    ax = axes[0][1]
    times = [4.2, 1.8]
    bars2 = ax.bar(algorithms, times, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Time (ms)', fontsize=11, weight='bold')
    ax.set_title('Computation Time')

    for bar, val in zip(bars2, times):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val}ms", ha='center', weight='bold')

    ax.text(0.5, 4.5, '2.3x faster', ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 3. Path Cost (same)
    ax = axes[1][0]
    costs = [127.5, 127.5]
    bars3 = ax.bar(algorithms, costs, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Total Cost', fontsize=11, weight='bold')
    ax.set_title('Both Find Optimal Path')

    for bar, val in zip(bars3, costs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 3, f"{val}", ha='center', weight='bold')

    # 4. Summary Table
    ax = axes[1][1]
    ax.axis('off')

    table_data = [
        ['Metric', 'UCS', 'A*'],
        ['Nodes Expanded', '15', '8'],
        ['Time (ms)', '4.2', '1.8'],
        ['Path Cost', '127.5', '127.5'],
        ['Travel Time (h)', '12.5', '12.5'],
        ['Fuel Used', '42.0', '42.0'],
        ['Efficiency', 'Baseline', '+47%'],
        ['Recommendation', '-', 'USE A* ✓']
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.3, 0.3])
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Header styling
    for col in range(3):
        table[(0, col)].set_facecolor('#40466e')
        table[(0, col)].set_text_props(color='white', weight='bold')

    plt.savefig('Algorithm_Comparison.png', dpi=300)
    print("✓ Saved: Algorithm_Comparison.png")
    plt.show()


# ============================================================
# 3. VISUALIZE HEURISTIC PROOF
# ============================================================

def visualize_heuristic():
    """Visualize heuristic admissibility"""

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300, constrained_layout=True)
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'HEURISTIC ADMISSIBILITY PROOF', fontsize=14, weight='bold', ha='center')

    ax.scatter([1], [1], s=300, c='red', edgecolors='black')
    ax.text(1, 0.4, 'Current Node', ha='center', weight='bold')

    ax.scatter([8], [6], s=300, c='green', edgecolors='black')
    ax.text(8, 6.7, 'Goal', ha='center', weight='bold')

    ax.plot([1, 8], [1, 6], 'g--', linewidth=3, label='Heuristic h(n)')
    ax.plot([1, 3, 5, 6.5, 8], [1, 2, 3, 4.5, 6], 'r-', linewidth=3, label='Actual Path')

    ax.text(4, 3.3, 'h(n) ≤ actual cost',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=11, weight='bold')

    proof_text = """
Why A* Heuristic Is Admissible:
1. h(n) = Straight-line distance
2. Straight-line ≤ real path cost
3. h(n) never overestimates
4. Therefore A* is always optimal ✓
"""
    ax.text(5, 3.6, proof_text, fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax.legend()

    plt.savefig('Heuristic_Admissibility.png', dpi=300)
    print("✓ Saved: Heuristic_Admissibility.png")
    plt.show()


# ============================================================
# MAIN ENTRY
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING MODULE 2 VISUALIZATIONS")
    print("="*60)

    print("\n1. Drawing Village Network...")
    visualize_network()

    print("\n2. Drawing Algorithm Comparison...")
    visualize_comparison()

    print("\n3. Drawing Heuristic Proof...")
    visualize_heuristic()

    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*60 + "\n")
