#!/usr/bin/env python3
"""
Enhanced batch runner script with automatic max timestep calculation
and failure reason detection.
"""

import math
import subprocess
import os
import time
import glob

def get_map_dimensions(map_file):
    """Parse map file to get dimensions (height, width)"""
    try:
        with open(map_file, 'r') as f:
            lines = f.readlines()
        
        # Find height and width from map file
        height = None
        width = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('height'):
                height = int(line.split()[-1])
            elif line.startswith('width'):
                width = int(line.split()[-1])
        
        if height is None or width is None:
            # Fallback: count actual map lines
            map_lines = [line for line in lines if line.strip() and not line.startswith(('type', 'height', 'width', 'map'))]
            height = len(map_lines)
            width = len(map_lines[0].strip()) if map_lines else 0
        
        return height, width
    except Exception as e:
        print(f"Error reading map {map_file}: {e}")
        return None, None

def analyze_failure_reason(stdout_content, max_timestep):
    """
    Analyze the failure reason based on output.
    Returns: 'insufficient_time', 'livelock', 'other', or 'unknown'
    """
    # Look for specific patterns in the output
    if "max timestep reached" in stdout_content.lower():
        return 'insufficient_time'
    elif "livelock" in stdout_content.lower():
        return 'livelock'
    elif "deadlock" in stdout_content.lower():
        return 'deadlock'
    
    # Check if we can infer from plan length (if available in output)
    # This would require modifications to your driver code to output more info
    lines = stdout_content.split('\n')
    for line in lines:
        if 'timestep' in line.lower() and 'reached' in line.lower():
            return 'insufficient_time'
        elif 'cycle' in line.lower() or 'loop' in line.lower():
            return 'livelock'
    
    return 'unknown'

def run_experiment(map_file, scen_file, num_agents, output_file, max_timestep=None):
    """Run a single experiment"""
    
    # Calculate max timestep if not provided
    if max_timestep is None:
        height, width = get_map_dimensions(map_file)
        if height and width:
            max_timestep = height * width * num_agents 
            print(f"  Map dimensions: {height}x{width}, max timestep: {max_timestep}")
        else:
            max_timestep = 1000  # fallback
            print(f"  Could not determine map size, using fallback: {max_timestep}")
    
    cmd = [
        "poetry", "run", "python", "app.py",
        "-m", map_file,
        "-i", scen_file,
        "-N", str(num_agents),
        "-o", output_file,
        "--max-timestep", str(max_timestep)
    ]
    
    print(f"Running: {os.path.basename(map_file)} + {os.path.basename(scen_file)} (N={num_agents})")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        runtime = time.time() - start_time

        solved = "solved: True" in result.stdout
        
        if solved:
            print(f"  Result: ✅ ({runtime:.2f}s)")
            failure_reason = None
        else:
            failure_reason = analyze_failure_reason(result.stdout, max_timestep)
            failure_emoji = {
                'insufficient_time': '⏰',
                'livelock': '🔄',
                'deadlock': '🚫',
                'unknown': '❌'
            }
            
            print(f"  Result: {failure_emoji.get(failure_reason, '❌')} - {failure_reason.replace('_', ' ').title()} ({runtime:.2f}s)")

        if result.stderr:
            print(f"  Error: {result.stderr}")
        
        # Save detailed output for failed cases
        if not solved:
            debug_file = output_file.replace('.txt', '_debug.txt')
            with open(debug_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Max timestep: {max_timestep}\n")
                f.write(f"Failure reason: {failure_reason}\n")
                f.write(f"Runtime: {runtime:.2f}s\n")
                f.write(f"\nSTDOUT:\n{result.stdout}\n")
                f.write(f"\nSTDERR:\n{result.stderr}\n")
            
        return solved, runtime, failure_reason, max_timestep
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, 0, 'error', max_timestep

def main():
    # Create output directory
    os.makedirs("batch_results", exist_ok=True)
    
    # Auto-generate experiments by scanning directories
    experiments = []

    # Add your specific experiment
    experiments.append(("assets/pushmap.map", "assets/pushmap-random-1.scen", 5))
    
    # Find all maps
    for map_file in glob.glob("assets/*.map"):
        map_name = os.path.basename(map_file).replace('.map', '')
        print(f"Found map: {map_name}")
        
        # Find corresponding scenarios in assets/scen/
        scen_pattern = f"assets/scen/{map_name}-random-*.scen"
        scen_files = glob.glob(scen_pattern)
        
        print(f"  Found {len(scen_files)} scenarios for {map_name}")
        
        for scen_file in scen_files:
            experiments.append((map_file, scen_file, 50))  

    print(f"\nTotal experiments to run: {len(experiments)}")
    
    # If no experiments found, show debug info
    if len(experiments) == 0:
        print("No experiments found! Debug info:")
        print(f"Maps found: {glob.glob('assets/*.map')}")
        print(f"Scenarios found: {glob.glob('assets/scen/*.scen')}")
        return
    
    # Run all experiments
    results = []
    total_time = 0
    failure_stats = {}
    
    print(f"Starting batch run with {len(experiments)} experiments...")
    print("="*80)
    
    for i, (map_file, scen_file, num_agents) in enumerate(experiments, 1):
        # Create output filename
        map_name = os.path.basename(map_file).replace('.map', '')
        scen_name = os.path.basename(scen_file).replace('.scen', '')
        output_file = f"batch_results/{map_name}_{scen_name}_N{num_agents}.txt"
        
        print(f"[{i}/{len(experiments)}]", end=" ")
        solved, runtime, failure_reason, max_timestep = run_experiment(
            map_file, scen_file, num_agents, output_file
        )
        
        results.append({
            'map': map_name,
            'scenario': scen_name,
            'agents': num_agents,
            'solved': solved,
            'runtime': runtime,
            'failure_reason': failure_reason,
            'max_timestep': max_timestep
        })
        total_time += runtime
        
        # Track failure statistics
        if not solved and failure_reason:
            failure_stats[failure_reason] = failure_stats.get(failure_reason, 0) + 1
    
    # Print summary
    print("="*80)
    print("BATCH RUN SUMMARY:")
    print(f"Total experiments: {len(results)}")
    solved_count = sum(1 for r in results if r['solved'])
    print(f"Solved: {solved_count} ({solved_count/len(results)*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per experiment: {total_time/len(results):.2f}s")
    
    if failure_stats:
        print(f"\nFailure breakdown:")
        for reason, count in failure_stats.items():
            print(f"  {reason.replace('_', ' ').title()}: {count}")
    
    # Save detailed summary to file
    with open("batch_results/summary.txt", "w") as f:
        f.write("Batch Run Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Solved: {solved_count} ({solved_count/len(results)*100:.1f}%)\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Average time: {total_time/len(results):.2f}s\n\n")
        
        if failure_stats:
            f.write("Failure breakdown:\n")
            for reason, count in failure_stats.items():
                f.write(f"  {reason.replace('_', ' ').title()}: {count}\n")
            f.write("\n")
        
        f.write("Detailed results:\n")
        f.write("-" * 50 + "\n")
        for r in results:
            status = "SOLVED" if r['solved'] else f"FAILED ({r['failure_reason']})"
            f.write(f"{r['map']} + {r['scenario']} (N={r['agents']}, max_t={r['max_timestep']}): "
                   f"{status} ({r['runtime']:.2f}s)\n")
    
    print("Results saved to batch_results/")
    print("Check *_debug.txt files for detailed failure analysis")

if __name__ == "__main__":
    main(),