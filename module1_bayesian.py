import pandas as pd
from itertools import product

class BayesianNetwork:
    """Simple Bayesian Network for Disaster Relief Urgency Assessment"""
    
    def __init__(self):   # FIXED
        self.cpts = {}
    
    # STEP 1: Set Prior Probabilities (Root Nodes)
    def set_priors(self):
        print("\n=== STEP 1: PRIOR PROBABILITIES ===\n")
        
        self.cpts['PopulationDensity'] = {
            'High': 0.30, 'Medium': 0.45, 'Low': 0.25
        }
        self.cpts['PrevDelivery'] = {'Yes': 0.45, 'No': 0.55}
        self.cpts['MedicalSignals'] = {'Present': 0.25, 'Absent': 0.75}
        self.cpts['DiseaseOutbreak'] = {'Present': 0.15, 'Absent': 0.85}
        self.cpts['WaterLevel'] = {'High': 0.35, 'Medium': 0.40, 'Low': 0.25}
        
        for node, probs in self.cpts.items():
            print(f"P({node}): {probs}")
    
    # STEP 2: CPT for NeedForSupplies (depends on PopDensity + PrevDelivery)
    def set_cpt_needforsupplies(self):
        print("\n=== STEP 2: P(NeedForSupplies | PopDensity, PrevDelivery) ===\n")
        
        self.cpts['NeedForSupplies'] = {}
        
        for pop in ['High', 'Medium', 'Low']:
            for prev in ['Yes', 'No']:
                pop_score = {'High': 0.9, 'Medium': 0.5, 'Low': 0.2}[pop]
                prev_score = {'No': 0.9, 'Yes': 0.2}[prev]
                score = 0.55 * pop_score + 0.45 * prev_score
                
                if score >= 0.70:
                    probs = {'High': 0.85, 'Medium': 0.12, 'Low': 0.03}
                elif score >= 0.50:
                    probs = {'High': 0.60, 'Medium': 0.30, 'Low': 0.10}
                else:
                    probs = {'High': 0.25, 'Medium': 0.40, 'Low': 0.35}
                
                self.cpts['NeedForSupplies'][(pop, prev)] = probs
        
        print("(High, No)  -> High: 0.85, Medium: 0.12, Low: 0.03")
        print("(Medium, Yes) -> High: 0.60, Medium: 0.30, Low: 0.10")
        print("(Low, No)   -> High: 0.25, Medium: 0.40, Low: 0.35")
    
    # STEP 3: CPT for RoadAccess (depends on WaterLevel)
    def set_cpt_roadaccess(self):
        print("\n=== STEP 3: P(RoadAccess | WaterLevel) ===\n")
        
        self.cpts['RoadAccess'] = {
            'High': {'Accessible': 0.25, 'Blocked': 0.75},
            'Medium': {'Accessible': 0.60, 'Blocked': 0.40},
            'Low': {'Accessible': 0.90, 'Blocked': 0.10}
        }
        
        for water, probs in self.cpts['RoadAccess'].items():
            print(f"WaterLevel={water} -> {probs}")
    
    # STEP 4: CPT for MedicalEmergency (depends on MedicalSignals + DiseaseOutbreak)
    def set_cpt_medicalemergency(self):
        print("\n=== STEP 4: P(MedicalEmergency | MedicalSignals, DiseaseOutbreak) ===\n")
        
        self.cpts['MedicalEmergency'] = {}
        
        for med in ['Present', 'Absent']:
            for dis in ['Present', 'Absent']:
                med_score = {'Present': 0.85, 'Absent': 0.15}[med]
                dis_score = {'Present': 0.90, 'Absent': 0.10}[dis]
                score = 0.45 * med_score + 0.55 * dis_score
                
                if score >= 0.75:
                    probs = {'High': 0.90, 'Medium': 0.08, 'Low': 0.02}
                elif score >= 0.50:
                    probs = {'High': 0.55, 'Medium': 0.35, 'Low': 0.10}
                else:
                    probs = {'High': 0.10, 'Medium': 0.30, 'Low': 0.60}
                
                self.cpts['MedicalEmergency'][(med, dis)] = probs
        
        print("(Present, Present) -> High: 0.90, Medium: 0.08, Low: 0.02")
        print("(Absent, Absent) -> High: 0.10, Medium: 0.30, Low: 0.60")
    
    # STEP 5: CPT for Urgency
    def set_cpt_urgency(self):
        print("\n=== STEP 5: P(Urgency | NeedForSupplies, MedicalEmergency, RoadAccess) ===\n")
        
        self.cpts['Urgency'] = {}
        
        for need in ['High', 'Medium', 'Low']:
            for med in ['High', 'Medium', 'Low']:
                for road in ['Accessible', 'Blocked']:
                    need_score = {'High': 0.9, 'Medium': 0.5, 'Low': 0.2}[need]
                    med_score = {'High': 0.9, 'Medium': 0.5, 'Low': 0.2}[med]
                    road_score = {'Accessible': 1.0, 'Blocked': 0.3}[road]
                    
                    score = 0.35 * need_score + 0.35 * med_score + 0.30 * road_score
                    
                    if score >= 0.75:
                        probs = {'High': 0.85, 'Medium': 0.12, 'Low': 0.03}
                    elif score >= 0.50:
                        probs = {'High': 0.55, 'Medium': 0.35, 'Low': 0.10}
                    else:
                        probs = {'High': 0.15, 'Medium': 0.35, 'Low': 0.50}
                    
                    self.cpts['Urgency'][(need, med, road)] = probs
        
        print("(High, High, Accessible) -> High: 0.85, Medium: 0.12, Low: 0.03")
        print("(Low, Low, Blocked) -> High: 0.15, Medium: 0.35, Low: 0.50")
    
    # STEP 6: Inference
    def infer_urgency(self, pop_density, prev_delivery, medical_signals, disease_outbreak, water_level):
        print(f"\n=== INFERENCE ===")
        print(f"Evidence: PopDensity={pop_density}, PrevDelivery={prev_delivery}, ")
        print(f"          MedicalSignals={medical_signals}, DiseaseOutbreak={disease_outbreak}, ")
        print(f"          WaterLevel={water_level}\n")
        
        need_probs = self.cpts['NeedForSupplies'][(pop_density, prev_delivery)]
        road_probs = self.cpts['RoadAccess'][water_level]
        med_probs = self.cpts['MedicalEmergency'][(medical_signals, disease_outbreak)]
        
        print(f"P(NeedForSupplies): {need_probs}")
        print(f"P(RoadAccess): {road_probs}")
        print(f"P(MedicalEmergency): {med_probs}\n")
        
        urgency_posterior = {'High': 0.0, 'Medium': 0.0, 'Low': 0.0}
        
        for need_state in need_probs:
            for med_state in med_probs:
                for road_state in road_probs:
                    urgency_cpt = self.cpts['Urgency'][(need_state, med_state, road_state)]
                    
                    for urgency_state in urgency_posterior:
                        urgency_posterior[urgency_state] += (
                            need_probs[need_state] *
                            med_probs[med_state] *
                            road_probs[road_state] *
                            urgency_cpt[urgency_state]
                        )
        
        total = sum(urgency_posterior.values())
        urgency_posterior = {k: v/total for k, v in urgency_posterior.items()}
        
        print(f"P(Urgency | Evidence):")
        for state, prob in sorted(urgency_posterior.items(), key=lambda x: x[1], reverse=True):
            print(f"  {state}: {prob:.4f}")
        
        max_state = max(urgency_posterior.items(), key=lambda x: x[1])
        print(f"\nMost Probable: {max_state[0]} (confidence: {max_state[1]:.4f})")
        
        return urgency_posterior
    
    # STEP 7: D-Separation
    def d_separation(self):
        print("\n=== D-SEPARATION ANALYSIS ===\n")
        print("Q1: PopDensity ⊥ MedicalSignals?")
        print("Path: PopDensity → NeedForSupplies → Urgency ← MedicalEmergency ← MedicalSignals")
        print("Urgency is COLLIDER → path blocked when unconditioned.\n")
        
        print("Q2: WaterLevel ⊥ PopDensity?")
        print("Path: WaterLevel → RoadAccess → Urgency ← NeedForSupplies ← PopDensity")
        print("Again, Urgency is COLLIDER → path blocked.\n")


def main():
    print("="*70)
    print("BAYESIAN NETWORK - DISASTER RELIEF URGENCY ASSESSMENT")
    print("="*70)
    
    bn = BayesianNetwork()
    
    bn.set_priors()
    bn.set_cpt_needforsupplies()
    bn.set_cpt_roadaccess()
    bn.set_cpt_medicalemergency()
    bn.set_cpt_urgency()
    
    bn.d_separation()
    
    test_cases = [
        {
            'name': 'Rayagada (High Urgency)',
            'PopulationDensity': 'High',
            'PrevDelivery': 'No',
            'MedicalSignals': 'Present',
            'DiseaseOutbreak': 'Absent',
            'WaterLevel': 'High'
        },
        {
            'name': 'Balangir (Low Urgency)',
            'PopulationDensity': 'Low',
            'PrevDelivery': 'Yes',
            'MedicalSignals': 'Absent',
            'DiseaseOutbreak': 'Absent',
            'WaterLevel': 'Medium'
        },
        {
            'name': 'Koraput (Medium Urgency)',
            'PopulationDensity': 'Medium',
            'PrevDelivery': 'No',
            'MedicalSignals': 'Present',
            'DiseaseOutbreak': 'Present',
            'WaterLevel': 'Low'
        }
    ]
    
    results = []
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        posterior = bn.infer_urgency(
            case['PopulationDensity'],
            case['PrevDelivery'],
            case['MedicalSignals'],
            case['DiseaseOutbreak'],
            case['WaterLevel']
        )
        
        max_state = max(posterior.items(), key=lambda x: x[1])
        results.append({
            'Village': case['name'],
            'High': f"{posterior['High']:.4f}",
            'Medium': f"{posterior['Medium']:.4f}",
            'Low': f"{posterior['Low']:.4f}",
            'Classification': max_state[0],
            'Confidence': f"{max_state[1]:.4f}"
        })
    
    df = pd.DataFrame(results)
    df.to_csv('urgency_assessment.csv', index=False)
    
    print("\n=== RESULTS TABLE ===")
    print(df.to_string(index=False))
    print("\n✓ Saved to: urgency_assessment.csv")


if __name__ == "__main__":   # FIXED
    main()
