"""
MASTER SCRIPT - Run All Disaster Relief AI Modules
Execute this script to run all 6 modules in sequence
"""

import sys
import os
from datetime import datetime

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_module(module_name, description):
    """Run a single module"""
    print_header(f"Running {module_name}: {description}")
    
    try:
        if module_name == "Module 1 - Bayesian Network":
            from module1_bayesian import main
            main()
            print("\n✓ Module 1 Complete!")
            
            # Run visualization
            print("\nGenerating Module 1 visualizations...")
            from module1_visualize import draw_network, draw_inference, draw_dseparation
            draw_network()
            draw_inference()
            draw_dseparation()
            print("✓ Module 1 Visualizations Complete!")
        
        elif module_name == "Module 2 - Search":
            from module2_search import main
            main()
            print("\n✓ Module 2 Complete!")
            
            # Run visualization
            print("\nGenerating Module 2 visualizations...")
            from module2_visualize import visualize_network, visualize_comparison, visualize_heuristic
            visualize_network()
            visualize_comparison()
            visualize_heuristic()
            print("✓ Module 2 Visualizations Complete!")
        
        elif module_name == "Module 3 - Planning":
            from module3_planning import main
            main()
            print("\n✓ Module 3 Complete!")
        
        elif module_name == "Module 4 - MARL":
            from module4_marl import main
            main()
            print("\n✓ Module 4 Complete!")
        
        elif module_name == "Module 5 - LLM Negotiation":
            from module5_llm_dialogue import main
            main()
            print("\n✓ Module 5 Complete!")
        
        elif module_name == "Module 6 - Integration":
            from module6_integration import main
            main()
            print("\n✓ Module 6 Complete!")
        
        return True
    
    except Exception as e:
        print(f"\n✗ ERROR in {module_name}: {str(e)}")
        print(f"   Make sure all required files are in the same directory.")
        return False

def main():
    """Main execution function"""
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("  MULTI-AGENT AI SYSTEM FOR DISASTER RELIEF")
    print("  Group 4 - Complete System Execution")
    print("="*80)
    print(f"\nStart Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of modules to run
    modules = [
        ("Module 1 - Bayesian Network", "Urgency Assessment using Probabilistic Reasoning"),
        ("Module 2 - Search", "Route Optimization using A* Algorithm"),
        ("Module 3 - Planning", "Task Coordination using GraphPlan & POP"),
        ("Module 4 - MARL", "Adaptive Learning using Multi-Agent RL"),
        ("Module 5 - LLM Negotiation", "Agent Coordination using Natural Language"),
        ("Module 6 - Integration", "Full System Integration & Performance Analysis")
    ]
    
    results = []
    
    # Run each module
    for i, (module_name, description) in enumerate(modules, 1):
        print(f"\n{'▶'*40}")
        print(f"  EXECUTING MODULE {i}/6")
        print(f"{'▶'*40}")
        
        success = run_module(module_name, description)
        results.append((module_name, success))
        
        if not success:
            print(f"\n⚠️  Warning: {module_name} encountered errors")
            response = input("Continue with remaining modules? (y/n): ")
            if response.lower() != 'y':
                print("\n❌ Execution stopped by user")
                break
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("  EXECUTION SUMMARY")
    print("="*80)
    print(f"\nStart Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    print("\n📊 Module Results:")
    for module_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status} - {module_name}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\n🎯 Overall Success Rate: {successful}/{len(results)} modules")
    
    print("\n📁 Generated Files:")
    output_files = [
        "urgency_assessment.csv",
        "BN_Structure.png",
        "Inference_Process.png",
        "D_Separation.png",
        "route_comparison_results.csv",
        "Village_Network.png",
        "Algorithm_Comparison.png",
        "Heuristic_Admissibility.png",
        "planning_results.json",
        "marl_results.json",
        "MARL_Results.png",
        "negotiation_dialogue.json",
        "negotiation_results.csv",
        "integrated_mission_report.json",
        "mission_performance_summary.csv",
        "Integrated_System_Dashboard.png"
    ]
    
    print("\nExpected output files:")
    for file in output_files:
        exists = "✓" if os.path.exists(file) else "○"
        print(f"  {exists} {file}")
    
    print("\n" + "="*80)
    print("  🎉 ALL MODULES EXECUTION COMPLETE!")
    print("="*80 + "\n")
    
    return successful == len(results)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)