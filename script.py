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
from collections import defaultdict

PIBT_VERSIONS = ["pibt_new", "oriori", "oripibt"]



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
            map_lines = [line for line in lines if line.strip() and not line.startswith(('type', 'height', 'width', 'map'))]
            height = len(map_lines)
            width = len(map_lines[0].strip()) if map_lines else 0
        
        return height, width
    except Exception as e:
        print(f"Error reading map {map_file}: {e}")
        return None, None

def analyze_failure_reason(stdout_content, max_timestep):
    # """
    # Analyze the failure reason based on output.
    # Returns: 'insufficient_time', 'livelock', 'other', or 'unknown'
    # """
    # Look for specific patterns in the output
    # if "max timestep reached" in stdout_content.lower():
    #     return 'insufficient_time'
    # elif "livelock" in stdout_content.lower():
    #     return 'livelock'
    # elif "deadlock" in stdout_content.lower():
    #     return 'deadlock'
    
    # # Check if we can infer from plan length (if available in output)
    # # This would require modifications to your driver code to output more info
    # lines = stdout_content.split('\n')
    # for line in lines:
    #     if 'timestep' in line.lower() and 'reached' in line.lower():
    #         return 'insufficient_time'
    #     elif 'cycle' in line.lower() or 'loop' in line.lower():
    #         return 'livelock'
    
    return 'unknown'

def run_experiment(map_file, scen_file, num_agents, output_file, max_timestep=None, pibt_version="pibt_new"):
    """Run a single experiment"""
    
    # Calculate max timestep if not provided
    if max_timestep is None:
        height, width = get_map_dimensions(map_file)
        if height and width:
            max_timestep = height * width * num_agents 
            print(f"  Map dimensions: {height}x{width}, max timestep: {max_timestep}")
        else:
            max_timestep = 1000  # fallback
            print(f"  Could not determine map size, using  fallback: {max_timestep}")
    
    cmd = [
        "python3", "app.py",
        "-m", map_file,
        "-i", scen_file,
        "-N", str(num_agents),
        "-o", output_file,
        "--max-timestep", str(max_timestep),
        "--pibt-version", pibt_version
    ]
    
    print(f"Running: {os.path.basename(map_file)} + {os.path.basename(scen_file)} (N={num_agents})")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        runtime = time.time() - start_time

        solved = "solved: True" in result.stdout
        
        if result.stderr:
            print(f"    ⚠️  Warning: {result.stderr[:100]}")
        
        # Save detailed output for failed cases
        if not solved:
            debug_file = output_file.replace('.txt', '_debug.txt')
            with open(debug_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"PIBT Version: {pibt_version}\n")
                f.write(f"Max timestep: {max_timestep}\n")
                f.write(f"Runtime: {runtime:.2f}s\n")
                f.write(f"\nSTDOUT:\n{result.stdout}\n")
                f.write(f"\nSTDERR:\n{result.stderr}\n")
            
        return solved, runtime, max_timestep
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return False, 0, max_timestep

def main():
    # Create output directories
    os.makedirs("batch_results", exist_ok=True)
    for version in PIBT_VERSIONS:
        os.makedirs(f"batch_results/{version}", exist_ok=True)
    
    # Auto-generate experiments by scanning directories -> FILE
    experiments = []

    # Add specific experiments
    experiments.append(("assets/small.map", "assets/small-random-1.scen", 2))
    experiments.append(("assets/pushmap.map", "assets/pushmap-random-1.scen", 5))
    
    # Find all maps
    for map_file in glob.glob("assets/*.map"):
        map_name = os.path.basename(map_file).replace('.map', '')
        
        # Find corresponding scenarios
        scen_pattern = f"assets/scen/{map_name}-random-*.scen"
        scen_files = glob.glob(scen_pattern)
        
        for scen_file in scen_files:
            experiments.append((map_file, scen_file, 50))  

    print(f"\n{'='*80}")
    print(f"PIBT VERSION COMPARISON BATCH RUNNER")
    print(f"{'='*80}")
    print(f"Testing {len(PIBT_VERSIONS)} versions: {', '.join(PIBT_VERSIONS)}")
    print(f"Total experiments per version: {len(experiments)}")
    print(f"Total runs: {len(experiments) * len(PIBT_VERSIONS)}")
    print(f"{'='*80}\n")
    
    # If no experiments found, show debug info
    if len(experiments) == 0:
        print("⚠️  No experiments found! Debug info:")
        print(f"Maps found: {glob.glob('assets/*.map')}")
        print(f"Scenarios found: {glob.glob('assets/scen/*.scen')}")
        return
    
    # Store results for comparison
    all_results = defaultdict(list)  # version -> list of results
    
    # Run all experiments for each version
    for version in PIBT_VERSIONS:
        print(f"\n{'='*80}")
        print(f"TESTING VERSION: {version.upper()}")
        print(f"{'='*80}\n")
        
        version_results = []
        version_time = 0
        
        for i, (map_file, scen_file, num_agents) in enumerate(experiments, 1):
            # Create output filename
            map_name = os.path.basename(map_file).replace('.map', '')
            scen_name = os.path.basename(scen_file).replace('.scen', '')
            output_file = f"batch_results/{version}/{map_name}_{scen_name}_N{num_agents}.txt"
            
            print(f"[{i}/{len(experiments)}] {map_name} + {scen_name} (N={num_agents})")
            
            solved, runtime, max_timestep = run_experiment(
                map_file, scen_file, num_agents, output_file, pibt_version=version
            )
            
            status_emoji = "✅" if solved else "❌"
            print(f"  {status_emoji} {version}: {'SOLVED' if solved else 'FAILED'} ({runtime:.2f}s)")
            
            version_results.append({
                'map': map_name,
                'scenario': scen_name,
                'agents': num_agents,
                'solved': solved,
                'runtime': runtime,
                'max_timestep': max_timestep
            })
            version_time += runtime
        
        all_results[version] = version_results
        
        # Print version summary
        solved_count = sum(1 for r in version_results if r['solved'])
        print(f"\n{version.upper()} Summary:")
        print(f"  Solved: {solved_count}/{len(version_results)} ({solved_count/len(version_results)*100:.1f}%)")
        print(f"  Total time: {version_time:.2f}s")
        print(f"  Avg time: {version_time/len(version_results):.2f}s")
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"{'Version':<15} {'Solved':<10} {'Success Rate':<15} {'Total Time':<12} {'Avg Time'}")
    print("-" * 80)
    
    for version in PIBT_VERSIONS:
        results = all_results[version]
        solved_count = sum(1 for r in results if r['solved'])
        total_time = sum(r['runtime'] for r in results)
        avg_time = total_time / len(results)
        success_rate = solved_count / len(results) * 100
        
        print(f"{version:<15} {solved_count}/{len(results):<7} {success_rate:>6.1f}%{'':<8} {total_time:>8.2f}s    {avg_time:.2f}s")
    
    # Detailed comparison for each experiment
    print(f"\n{'='*80}")
    print("Detailed Per-Experiment Comparison:")
    print(f"{'='*80}\n")
    
    comparison_data = []
    
    for i, (map_file, scen_file, num_agents) in enumerate(experiments):
        map_name = os.path.basename(map_file).replace('.map', '')
        scen_name = os.path.basename(scen_file).replace('.scen', '')
        
        print(f"{map_name} + {scen_name} (N={num_agents}):")
        
        exp_comparison = {
            'map': map_name,
            'scenario': scen_name,
            'agents': num_agents
        }
        
        for version in PIBT_VERSIONS:
            result = all_results[version][i]
            status = "✅ SOLVED" if result['solved'] else "❌ FAILED"
            print(f"  {version:<12}: {status:<12} ({result['runtime']:.2f}s)")
            
            exp_comparison[f'{version}_solved'] = result['solved']
            exp_comparison[f'{version}_time'] = result['runtime']
        
        # Highlight differences
        solved_versions = [v for v in PIBT_VERSIONS if all_results[v][i]['solved']]
        failed_versions = [v for v in PIBT_VERSIONS if not all_results[v][i]['solved']]
        
        if solved_versions and failed_versions:
            print(f"  ⚠️  Different results: {', '.join(solved_versions)} solved, {', '.join(failed_versions)} failed")
        elif solved_versions:
            times = [all_results[v][i]['runtime'] for v in PIBT_VERSIONS]
            fastest = PIBT_VERSIONS[times.index(min(times))]
            slowest = PIBT_VERSIONS[times.index(max(times))]
            if max(times) > min(times) * 1.2:  # 20% difference
                print(f"  🏃 {fastest} was fastest ({min(times):.2f}s), {slowest} was slowest ({max(times):.2f}s)")
        
        comparison_data.append(exp_comparison)
        print()
    
    # Save detailed comparison to file
    with open("batch_results/comparison_summary.txt", "w") as f:
        f.write("PIBT VERSION COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Versions tested: {', '.join(PIBT_VERSIONS)}\n")
        f.write(f"Total experiments: {len(experiments)}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Version':<15} {'Solved':<15} {'Success Rate':<15} {'Total Time':<15} {'Avg Time'}\n")
        f.write("-" * 80 + "\n")
        
        for version in PIBT_VERSIONS:
            results = all_results[version]
            solved_count = sum(1 for r in results if r['solved'])
            total_time = sum(r['runtime'] for r in results)
            avg_time = total_time / len(results)
            success_rate = solved_count / len(results) * 100
            
            f.write(f"{version:<15} {solved_count}/{len(results):<12} "
                   f"{success_rate:.1f}%{'':<11} {total_time:.2f}s{'':<10} {avg_time:.2f}s\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Detailed Per-Experiment Results:\n")
        f.write("="*80 + "\n\n")
        
        for exp in comparison_data:
            f.write(f"{exp['map']} + {exp['scenario']} (N={exp['agents']}):\n")
            for version in PIBT_VERSIONS:
                status = "SOLVED" if exp[f'{version}_solved'] else "FAILED"
                f.write(f"  {version:<12}: {status:<8} ({exp[f'{version}_time']:.2f}s)\n")
            f.write("\n")
    
    print(f"\n{'='*80}")
    print("✨ Results saved to:")
    print(f"  - batch_results/comparison_summary.txt (main comparison)")
    print(f"  - batch_results/<version>/ (individual results)")
    print(f"  - batch_results/<version>/*_debug.txt (failure details)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()