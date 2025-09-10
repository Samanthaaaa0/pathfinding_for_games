#!/usr/bin/env python3

import subprocess
import os
import time

def run_experiment(map_file, scen_file, num_agents, output_file):
    cmd = [
        "poetry", "run", "python", "app.py",
        "-m", map_file,
        "-i", scen_file,
        "-N", str(num_agents),
        "-o", output_file
    ]
    
    print(f"Running: {os.path.basename(map_file)} + {os.path.basename(scen_file)} (N={num_agents})")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        runtime = time.time() - start_time
        
        solved = "solved: True" in result.stdout
        print(f"  Result: {'✓ SOLVED' if solved else '✗ FAILED'} ({runtime:.2f}s)")
        
        if result.stderr:
            print(f"  Error: {result.stderr}")
            
        return solved, runtime
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, 0

def main():
    os.makedirs("batch_results", exist_ok=True)
    
    # Define your experiments here
    # experiments = [
    #     # map_file, scen_file, num_agents
    #     ("assets/random-32-32-20.map", "assets/32-32-20-scen/random-32-32-20-random-4.scen", 200),
    #     ("assets/random-32-32-20.map", "assets/32-32-20-scen/random-32-32-20-random-1.scen", 200),
    # ]
    
    # You can also auto-generate experiments by scanning directories:
    # Uncomment the following section if you want automatic detection:
    
    import glob
    experiments = []
    
    # Find all maps
    for map_file in glob.glob("assets/*.map"):
        map_name = os.path.basename(map_file).replace('.map', '')
        
        # Find corresponding scenario directory
        scen_dir = f"assets/scens/{map_name}-*"
        if os.path.exists(scen_dir):
            for scen_file in glob.glob(f"{scen_dir}/*.scen"):
                experiments.append((map_file, scen_file, 200))  
    
    # Run all experiments
    results = []
    total_time = 0
    
    print(f"Starting batch run with {len(experiments)} experiments...")
    print("="*60)
    
    for i, (map_file, scen_file, num_agents) in enumerate(experiments, 1):
        # Create output filename
        map_name = os.path.basename(map_file).replace('.map', '')
        scen_name = os.path.basename(scen_file).replace('.scen', '')
        output_file = f"batch_results/{map_name}_{scen_name}_N{num_agents}.txt"
        
        print(f"[{i}/{len(experiments)}]", end=" ")
        solved, runtime = run_experiment(map_file, scen_file, num_agents, output_file)
        
        results.append({
            'map': map_name,
            'scenario': scen_name,
            'agents': num_agents,
            'solved': solved,
            'runtime': runtime
        })
        total_time += runtime
    
    # Print summary
    print("="*60)
    print("BATCH RUN SUMMARY:")
    print(f"Total experiments: {len(results)}")
    print(f"Solved: {sum(1 for r in results if r['solved'])}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per experiment: {total_time/len(results):.2f}s")
    
    # Save summary to file
    with open("batch_results/summary.txt", "w") as f:
        f.write("Batch Run Summary\n")
        f.write("="*50 + "\n")
        for r in results:
            f.write(f"{r['map']} + {r['scenario']} (N={r['agents']}): "
                   f"{'SOLVED' if r['solved'] else 'FAILED'} ({r['runtime']:.2f}s)\n")
    
    print("Results saved to batch_results/")

if __name__ == "__main__":
    main()