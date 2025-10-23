#!/usr/bin/env python3
"""
Enhanced parallel batch runner with automatic max timestep calculation,
failure reason detection, and metrics extraction (makespan, sum of cost).
"""

import math
import subprocess
import os
import time
import glob
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

PIBT_VERSIONS = ["pibt_new", "oriori", "oripibt"]

# Number of parallel processes
NUM_PARALLEL = int(sys.argv[1]) if len(sys.argv) > 1 else 4

def get_map_dimensions(map_file):
    """Parse map file to get dimensions (height, width)"""
    try:
        with open(map_file, 'r') as f:
            lines = f.readlines()
        
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

def parse_output_metrics(output_file):
    """
    Parse the output file to extract makespan and sum of cost.    
    """
    try:
        with open(output_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and ':' in line]
        
        if not lines:
            return None, None
        
        # Makespan is the number of timesteps (number of lines)
        makespan = len(lines)
        
        # Parse agent trajectories
        agent_paths = defaultdict(list)
        
        for line in lines:
            if ':' not in line:
                continue
            
            # Extract agent positions from line
            # Format: 0:(2,1),(2,2),(1,2)
            parts = line.split(':')
            if len(parts) < 2:
                continue
            
            timestep = int(parts[0])
            positions_str = parts[1]
            
            # extract (x,y) positions
            positions = re.findall(r'\((\d+),(\d+)\)', positions_str)
            
            for agent_idx, (x, y) in enumerate(positions):
                agent_paths[agent_idx].append((int(x), int(y)))
        
        # SOC
        sum_of_cost = 0
        for agent_idx, path in agent_paths.items():
            # Cost is number of moves = path length - 1, or just path length if counting all steps
            # Using path length (including start position)
            agent_cost = 0
            for i in range(1, len(path)):
                # Count as move if position changed
                if path[i] != path[i-1]:
                    agent_cost += 1
                else:
                    # Count waiting as a move too (standard in MAPF)
                    agent_cost += 1
            sum_of_cost += agent_cost
        
        return makespan, sum_of_cost
    
    except Exception as e:
        print(f"    ⚠️  Error parsing metrics from {output_file}: {e}")
        return None, None

def run_single_experiment(args):
    """
    Return (experiment_info, result_dict)
    """
    map_file, scen_file, num_agents, output_file, max_timestep, pibt_version, exp_id, total_exps = args
    
    cmd = [
        "python3", "app.py",
        "-m", map_file,
        "-i", scen_file,
        "-N", str(num_agents),
        "-o", output_file,
        "--max-timestep", str(max_timestep),
        "--pibt-version", pibt_version
    ]
    
    map_name = os.path.basename(map_file).replace('.map', '')
    scen_name = os.path.basename(scen_file).replace('.scen', '')
    
    print(f"[{exp_id}/{total_exps}] Starting: {map_name} + {scen_name} (N={num_agents}, {pibt_version})")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        runtime = time.time() - start_time
        
        solved = "solved: True" in result.stdout
        
        # Parse metrics from output file
        makespan = None
        sum_of_cost = None
        if solved and os.path.exists(output_file):
            makespan, sum_of_cost = parse_output_metrics(output_file)
        
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
        
        status = "✅" if solved else "❌"
        metrics_str = ""
        if makespan is not None:
            metrics_str = f", makespan={makespan}, soc={sum_of_cost}"
        print(f"[{exp_id}/{total_exps}] {status} {pibt_version}: {map_name}/{scen_name} ({runtime:.2f}s{metrics_str})")
        
        return (map_file, scen_file, num_agents, pibt_version), {
            'solved': solved,
            'runtime': runtime,
            'makespan': makespan,
            'sum_of_cost': sum_of_cost,
            'max_timestep': max_timestep,
            'map': map_name,
            'scenario': scen_name,
            'agents': num_agents
        }
        
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print(f"[{exp_id}/{total_exps}] ⏱️  TIMEOUT: {map_name}/{scen_name} ({pibt_version})")
        return (map_file, scen_file, num_agents, pibt_version), {
            'solved': False,
            'runtime': runtime,
            'makespan': None,
            'sum_of_cost': None,
            'max_timestep': max_timestep,
            'map': map_name,
            'scenario': scen_name,
            'agents': num_agents,
            'timeout': True
        }
    except Exception as e:
        print(f"[{exp_id}/{total_exps}] ❌ ERROR: {map_name}/{scen_name} ({pibt_version}): {e}")
        return (map_file, scen_file, num_agents, pibt_version), {
            'solved': False,
            'runtime': 0,
            'makespan': None,
            'sum_of_cost': None,
            'max_timestep': max_timestep,
            'map': map_name,
            'scenario': scen_name,
            'agents': num_agents,
            'error': str(e)
        }

def main():
    # output dir
    os.makedirs("batch_results", exist_ok=True)
    for version in PIBT_VERSIONS:
        os.makedirs(f"batch_results/{version}", exist_ok=True)
    
    experiments = []
    
    # Add specific experiments
    experiments.append(("assets/small.map", "assets/small-random-1.scen", 2))
    experiments.append(("assets/pushmap.map", "assets/pushmap-random-1.scen", 5))
    
    # Find all maps
    for map_file in glob.glob("assets/*.map"):
        map_name = os.path.basename(map_file).replace('.map', '')
        scen_pattern = f"assets/scen/{map_name}-random-*.scen"
        scen_files = glob.glob(scen_pattern)
        
        for scen_file in scen_files:
            experiments.append((map_file, scen_file, 50))
    
    print(f"\n{'='*80}")
    print(f"PARALLEL PIBT VERSION COMPARISON BATCH RUNNER")
    print(f"{'='*80}")
    print(f"Testing {len(PIBT_VERSIONS)} versions: {', '.join(PIBT_VERSIONS)}")
    print(f"Total experiments per version: {len(experiments)}")
    print(f"Total runs: {len(experiments) * len(PIBT_VERSIONS)}")
    print(f"Parallel workers: {NUM_PARALLEL}")
    print(f"{'='*80}\n")
    
    if len(experiments) == 0:
        print("⚠️  No experiments found! Debug info:")
        print(f"Maps found: {glob.glob('assets/*.map')}")
        print(f"Scenarios found: {glob.glob('assets/scen/*.scen')}")
        return
    
    # Prepare all tasks
    all_tasks = []
    task_id = 1
    total_tasks = len(experiments) * len(PIBT_VERSIONS)
    
    for version in PIBT_VERSIONS:
        for map_file, scen_file, num_agents in experiments:
            # Calculate max timestep
            height, width = get_map_dimensions(map_file)
            if height and width:
                max_timestep = height * width * num_agents
            else:
                max_timestep = 1000
            
            # Create output filename
            map_name = os.path.basename(map_file).replace('.map', '')
            scen_name = os.path.basename(scen_file).replace('.scen', '')
            output_file = f"batch_results/{version}/{map_name}_{scen_name}_N{num_agents}.txt"
            
            all_tasks.append((
                map_file, scen_file, num_agents, output_file, 
                max_timestep, version, task_id, total_tasks
            ))
            task_id += 1
    
    # Run experiments in parallel
    print(f"🚀 Starting {total_tasks} experiments with {NUM_PARALLEL} parallel workers...\n")
    start_time = time.time()
    
    results_dict = {}
    
    with ProcessPoolExecutor(max_workers=NUM_PARALLEL) as executor:
        futures = {executor.submit(run_single_experiment, task): task for task in all_tasks}
        
        for future in as_completed(futures):
            try:
                key, result = future.result()
                results_dict[key] = result
            except Exception as e:
                print(f"❌ Task failed with exception: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ All experiments completed in {total_time:.2f}s")
    print(f"{'='*80}\n")
    
    # Organize results by version
    all_results = defaultdict(list)
    for version in PIBT_VERSIONS:
        for map_file, scen_file, num_agents in experiments:
            key = (map_file, scen_file, num_agents, version)
            if key in results_dict:
                all_results[version].append(results_dict[key])
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"{'Version':<15} {'Solved':<10} {'Rate':<8} {'Total Time':<12} {'Avg Time':<12} {'Avg Makespan':<15} {'Avg SOC'}")
    print("-" * 100)
    
    for version in PIBT_VERSIONS:
        results = all_results[version]
        solved_count = sum(1 for r in results if r['solved'])
        total_time_v = sum(r['runtime'] for r in results)
        avg_time = total_time_v / len(results) if results else 0
        success_rate = solved_count / len(results) * 100 if results else 0
        
        # Calculate average makespan and sum of cost for solved instances
        makespans = [r['makespan'] for r in results if r['solved'] and r['makespan'] is not None]
        socs = [r['sum_of_cost'] for r in results if r['solved'] and r['sum_of_cost'] is not None]
        
        avg_makespan = sum(makespans) / len(makespans) if makespans else 0
        avg_soc = sum(socs) / len(socs) if socs else 0
        
        print(f"{version:<15} {solved_count}/{len(results):<7} {success_rate:>5.1f}%  "
              f"{total_time_v:>8.2f}s    {avg_time:>8.2f}s    "
              f"{avg_makespan:>10.1f}      {avg_soc:>10.1f}")
    
    # Detailed comparison
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
            key = (map_file, scen_file, num_agents, version)
            result = results_dict.get(key, {})
            
            status = "✅ SOLVED" if result.get('solved') else "❌ FAILED"
            metrics = f"{result.get('runtime', 0):.2f}s"
            if result.get('makespan'):
                metrics += f", makespan={result['makespan']}, soc={result['sum_of_cost']}"
            
            print(f"  {version:<12}: {status:<12} ({metrics})")
            
            exp_comparison[f'{version}_solved'] = result.get('solved', False)
            exp_comparison[f'{version}_time'] = result.get('runtime', 0)
            exp_comparison[f'{version}_makespan'] = result.get('makespan')
            exp_comparison[f'{version}_soc'] = result.get('sum_of_cost')
        
        # Highlight differences
        solved_versions = [v for v in PIBT_VERSIONS if results_dict.get((map_file, scen_file, num_agents, v), {}).get('solved')]
        failed_versions = [v for v in PIBT_VERSIONS if not results_dict.get((map_file, scen_file, num_agents, v), {}).get('solved')]
        
        if solved_versions and failed_versions:
            print(f"  ⚠️  Different results: {', '.join(solved_versions)} solved, {', '.join(failed_versions)} failed")
        elif len(solved_versions) == len(PIBT_VERSIONS):
            # Compare quality metrics
            makespans = {v: results_dict.get((map_file, scen_file, num_agents, v), {}).get('makespan') for v in PIBT_VERSIONS}
            socs = {v: results_dict.get((map_file, scen_file, num_agents, v), {}).get('sum_of_cost') for v in PIBT_VERSIONS}
            
            if all(makespans.values()):
                best_makespan = min(makespans.values())
                best_version = [v for v, m in makespans.items() if m == best_makespan][0]
                if max(makespans.values()) > best_makespan:
                    print(f"  🏆 {best_version} has best makespan ({best_makespan})")
        
        comparison_data.append(exp_comparison)
        print()
    
    # Save detailed comparison to file
    with open("batch_results/comparison_summary.txt", "w") as f:
        f.write("PIBT VERSION COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Versions tested: {', '.join(PIBT_VERSIONS)}\n")
        f.write(f"Total experiments: {len(experiments)}\n")
        f.write(f"Parallel workers: {NUM_PARALLEL}\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        
        f.write("Overall Statistics:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Version':<15} {'Solved':<15} {'Rate':<10} {'Total Time':<15} {'Avg Time':<15} {'Avg Makespan':<15} {'Avg SOC'}\n")
        f.write("-" * 100 + "\n")
        
        for version in PIBT_VERSIONS:
            results = all_results[version]
            solved_count = sum(1 for r in results if r['solved'])
            total_time_v = sum(r['runtime'] for r in results)
            avg_time = total_time_v / len(results) if results else 0
            success_rate = solved_count / len(results) * 100 if results else 0
            
            makespans = [r['makespan'] for r in results if r['solved'] and r['makespan'] is not None]
            socs = [r['sum_of_cost'] for r in results if r['solved'] and r['sum_of_cost'] is not None]
            
            avg_makespan = sum(makespans) / len(makespans) if makespans else 0
            avg_soc = sum(socs) / len(socs) if socs else 0
            
            f.write(f"{version:<15} {solved_count}/{len(results):<12} {success_rate:.1f}%{'':<6} "
                   f"{total_time_v:.2f}s{'':<10} {avg_time:.2f}s{'':<10} "
                   f"{avg_makespan:.1f}{'':<10} {avg_soc:.1f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Detailed Per-Experiment Results:\n")
        f.write("="*80 + "\n\n")
        
        for exp in comparison_data:
            f.write(f"{exp['map']} + {exp['scenario']} (N={exp['agents']}):\n")
            for version in PIBT_VERSIONS:
                status = "SOLVED" if exp[f'{version}_solved'] else "FAILED"
                metrics = f"{exp[f'{version}_time']:.2f}s"
                if exp[f'{version}_makespan']:
                    metrics += f", makespan={exp[f'{version}_makespan']}, soc={exp[f'{version}_soc']}"
                f.write(f"  {version:<12}: {status:<8} ({metrics})\n")
            f.write("\n")
    
    print(f"\n{'='*80}")
    print("✨ Results saved to:")
    print(f"  - batch_results/comparison_summary.txt (main comparison)")
    print(f"  - batch_results/<version>/ (individual results)")
    print(f"  - batch_results/<version>/*_debug.txt (failure details)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()