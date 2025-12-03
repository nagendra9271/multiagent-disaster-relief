import heapq
import time
import pandas as pd
from collections import defaultdict
import math

# ============================================================
# VILLAGE GRAPH CLASS
# ============================================================

class VillageGraph:
    """Graph representing villages with accessibility & fuel constraints"""
    
    def __init__(self):
        self.villages = {}
        self.edges = []
        self.adjacency = defaultdict(list)
    
    def add_village(self, name, lat, lon, population=100):
        self.villages[name] = {'lat': lat, 'lon': lon, 'population': population}
    
    def add_edge(self, v1, v2, travel_time, access_type='road', fuel_cost=1.0):
        penalty = {'road': 0, 'boat': 5, 'blocked': 1000}.get(access_type, 0)
        cost = travel_time * 10 + fuel_cost * 5 + penalty
        
        self.edges.append((v1, v2, cost, access_type, fuel_cost))
        
        self.adjacency[v1].append({
            'neighbor': v2,
            'cost': cost,
            'time': travel_time,
            'fuel': fuel_cost,
            'type': access_type
        })
        self.adjacency[v2].append({
            'neighbor': v1,
            'cost': cost,
            'time': travel_time,
            'fuel': fuel_cost,
            'type': access_type
        })
    
    def euclidean_distance(self, v1, v2):
        lat1, lon1 = self.villages[v1]['lat'], self.villages[v1]['lon']
        lat2, lon2 = self.villages[v2]['lat'], self.villages[v2]['lon']
        lat_diff = (lat2 - lat1) * 111
        lon_diff = (lon2 - lon1) * 111 * math.cos(math.radians(lat1))
        return math.sqrt(lat_diff**2 + lon_diff**2) / 40
    
    def heuristic_h(self, current, goal):
        return self.euclidean_distance(current, goal)

    # ============================================================
    # UCS SEARCH (CLEAN OUTPUT)
    # ============================================================

    def ucs_search(self, start, goal, fuel_limit=100):
        pq = [(0, start, [start], 0, 0)]
        visited = set()
        nodes_expanded = 0
        start_time = time.time()
        
        while pq:
            cost, current, path, total_time, total_fuel = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1
            
            if current == goal:
                return {
                    'path': path,
                    'cost': cost,
                    'nodes': nodes_expanded,
                    'time': time.time() - start_time,
                    'travel': total_time,
                    'fuel': total_fuel
                }
            
            for e in self.adjacency[current]:
                if e['neighbor'] not in visited:
                    if total_fuel + e['fuel'] <= fuel_limit:
                        heapq.heappush(
                            pq, 
                            (cost + e['cost'], e['neighbor'], path + [e['neighbor']], total_time + e['time'], total_fuel + e['fuel'])
                        )
        
        return None

    # ============================================================
    # A* SEARCH (CLEAN OUTPUT)
    # ============================================================

    def astar_search(self, start, goal, fuel_limit=100):
        pq = [(self.heuristic_h(start, goal), 0, start, [start], 0, 0)]
        visited = set()
        g_scores = {start: 0}
        nodes_expanded = 0
        start_time = time.time()
        
        while pq:
            f_score, g_score, current, path, total_time, total_fuel = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1

            if current == goal:
                return {
                    'path': path,
                    'cost': g_score,
                    'nodes': nodes_expanded,
                    'time': time.time() - start_time,
                    'travel': total_time,
                    'fuel': total_fuel
                }
            
            for e in self.adjacency[current]:
                new_g = g_score + e['cost']
                if new_g < g_scores.get(e['neighbor'], float('inf')) and total_fuel + e['fuel'] <= fuel_limit:
                    g_scores[e['neighbor']] = new_g
                    new_f = new_g + self.heuristic_h(e['neighbor'], goal)
                    heapq.heappush(
                        pq,
                        (new_f, new_g, e['neighbor'], path + [e['neighbor']], total_time + e['time'], total_fuel + e['fuel'])
                    )
        
        return None


# ============================================================
# ODISHA NETWORK
# ============================================================

def create_odisha_network():
    graph = VillageGraph()
    
    villages = {
        'Rayagada': (19.1724, 83.4198, 150),
        'Balangir': (20.6577, 83.4677, 200),
        'Koraput': (18.8165, 84.0805, 120),
        'Bhadrak': (20.8155, 86.5064, 180),
        'Jagatsinghpur': (20.1689, 86.4126, 160),
        'Cuttack': (20.4625, 85.8830, 250),
        'Puri': (19.8135, 85.8312, 220),
        'Kendrapara': (20.5114, 87.0264, 140)
    }
    
    for v, (lat, lon, pop) in villages.items():
        graph.add_village(v, lat, lon, pop)
    
    edges = [
        ('Rayagada', 'Balangir', 3.5, 'road', 15),
        ('Balangir', 'Cuttack', 5.0, 'road', 20),
        ('Cuttack', 'Bhadrak', 3.0, 'road', 12),
        ('Bhadrak', 'Jagatsinghpur', 2.5, 'road', 8),
        ('Jagatsinghpur', 'Puri', 4.0, 'road', 16),
        ('Puri', 'Kendrapara', 3.5, 'boat', 18),
        ('Kendrapara', 'Cuttack', 4.5, 'road', 18),
        ('Koraput', 'Rayagada', 4.0, 'road', 18),
        ('Rayagada', 'Cuttack', 6.0, 'road', 25),
        ('Balangir', 'Koraput', 5.5, 'boat', 22),
    ]
    
    for e in edges:
        graph.add_edge(*e)
    
    return graph


# ============================================================
# CASE-INSENSITIVE INPUT
# ============================================================

def normalize_input(user_text, valid_names):
    mapping = {name.lower(): name for name in valid_names}
    return mapping.get(user_text.lower(), None)


# ============================================================
# ALGORITHM COMPARISON (CLEAN MODE)
# ============================================================

def compare_algorithms(graph, start, goal, fuel_limit):
    print("\n======================================================================")
    print("COMPARING SEARCH ALGORITHMS")
    print("======================================================================\n")
    
    # Run UCS
    ucs = graph.ucs_search(start, goal, fuel_limit)
    print("[UCS]")
    print(f"  Path Found: {' → '.join(ucs['path'])}")
    print(f"  Cost: {ucs['cost']}")
    print(f"  Travel Time: {ucs['travel']} h")
    print(f"  Fuel Used: {ucs['fuel']}")
    print(f"  Nodes Expanded: {ucs['nodes']}\n")
    
    # Run A*
    astar = graph.astar_search(start, goal, fuel_limit)
    print("[A*]")
    print(f"  Path Found: {' → '.join(astar['path'])}")
    print(f"  Cost: {astar['cost']}")
    print(f"  Travel Time: {astar['travel']} h")
    print(f"  Fuel Used: {astar['fuel']}")
    print(f"  Nodes Expanded: {astar['nodes']}\n")
    
    # Table
    df = pd.DataFrame([
        {
            'Algorithm': 'UCS',
            'Path': ' → '.join(ucs['path']),
            'Cost': ucs['cost'],
            'Nodes': ucs['nodes'],
            'Time(ms)': round(ucs['time']*1000, 3),
            'Travel(h)': ucs['travel'],
            'Fuel': ucs['fuel']
        },
        {
            'Algorithm': 'A*',
            'Path': ' → '.join(astar['path']),
            'Cost': astar['cost'],
            'Nodes': astar['nodes'],
            'Time(ms)': round(astar['time']*1000, 3),
            'Travel(h)': astar['travel'],
            'Fuel': astar['fuel']
        }
    ])
    
    return df


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("\n" + "="*70)
    print("MODULE 2: SEARCH-BASED ROUTE OPTIMIZATION FOR DISASTER RELIEF")
    print("="*70)
    
    graph = create_odisha_network()
    
    print("\nAvailable villages:", ', '.join(graph.villages.keys()))
    
    start_input = input("\nStart village (default Rayagada): ").strip() or "Rayagada"
    goal_input = input("Goal village (default Puri): ").strip() or "Puri"
    
    start = normalize_input(start_input, graph.villages.keys())
    goal = normalize_input(goal_input, graph.villages.keys())
    
    if start is None or goal is None:
        print("Invalid village name!")
        return
    
    fuel_limit = int(input("Fuel limit (default 100): ").strip() or 100)
    
    results_df = compare_algorithms(graph, start, goal, fuel_limit)
    
    print("\n======================================================================")
    print("DETAILED RESULTS TABLE")
    print("======================================================================\n")
    print(results_df.to_string(index=False))
    
    results_df.to_csv("route_comparison_results.csv", index=False)
    print("\n✓ Saved to route_comparison_results.csv\n")


if __name__ == "__main__":
    main()
