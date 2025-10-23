#!/usr/bin/env python3
"""
Simple parallel batch runner using subprocess.Popen with manual polling.
No fancy classes, just simple process pool management.
"""

import subprocess
import os
import time
import glob
import sys
from collections import defaultdict
import re

# PIBT_VERSIONS = ["pibt_new", "oriori", "oripibt"]
PIBT_VERSIONS = ["pibt_new"]  

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
    """Parse the output file to extract makespan and sum of cost."""
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
            
            parts = line.split(':')
            if len(parts) < 2:
                continue
            
            timestep = int(parts[0])
            positions_str = parts[1]
            
            # Extract all (x,y) positions
            positions = re.findall(r'\((\d+),(\d+)\)', positions_str)
            
            for agent_idx, (x, y) in enumerate(positions):
                agent_paths[agent_idx].append((int(x), int(y)))
        
        # Calculate sum of cost (total moves for all agents)
        sum_of_cost = 0
        for agent_idx, path in agent_paths.items():
            agent_cost = 0
            for i in range(1, len(path)):
                # Count every timestep as a cost (including waiting)
                agent_cost += 1
            sum_of_cost += agent_cost
        
        return makespan, sum_of_cost
    
    except Exception as e:
        print(f"    ⚠️  Error parsing metrics from {output_file}: {e}")
        return None, None

def main():
    # Create output directories
    os.makedirs("batch_results", exist_ok=True)
    for version in PIBT_VERSIONS:
        os.makedirs(f"batch_results/{version}", exist_ok=True)
    
    experiments = []
    map_count=0
    
    if os.path.exists("assets/small.map"):
        experiments.append(("assets/small.map", "assets/small-random-1.scen", 2))
    if os.path.exists("assets/pushmap.map"):
        experiments.append(("assets/pushmap.map", "assets/pushmap-random-1.scen", 5))
    map_count+=2
    
    # Find all maps
    for map_file in glob.glob("assets/*.map"):
        map_name = os.path.basename(map_file).replace('.map', '')
        scen_pattern = f"assets/scen/{map_name}-random-*.scen"
        scen_files = glob.glob(scen_pattern)

        scen_pattern_even = f"assets/scen/{map_name}-even-*.scen"
        scen_files += glob.glob(scen_pattern_even)

        print(f"Found {len(scen_files)} scenarios for map {map_name}")
        
        for scen_file in scen_files:
            experiments.append((map_file, scen_file, 50))
        

    
    
    print(f"\n{'='*80}")
    print(f"SIMPLE PARALLEL BATCH RUNNER")
    print(f"{'='*80}")
    print(f"Testing {len(PIBT_VERSIONS)} versions: {', '.join(PIBT_VERSIONS)}")
    print(f"Total experiments per version: {len(experiments)}, from {map_count} maps")
    print(f"Total runs: {len(experiments) * len(PIBT_VERSIONS)}")
    print(f"Parallel workers: {NUM_PARALLEL}")
    print(f"{'='*80}\n")
    
    if len(experiments) == 0:
        print("⚠️  No experiments found!")
        return
    
    # Prepare all tasks
    all_tasks = []
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
            
            all_tasks.append({
                'map_file': map_file,
                'scen_file': scen_file,
                'num_agents': num_agents,
                'output_file': output_file,
                'max_timestep': max_timestep,
                'version': version,
                'map_name': map_name,
                'scen_name': scen_name
            })
    
    print(f"🚀 Starting {len(all_tasks)} experiments with {NUM_PARALLEL} parallel workers...\n")
    start_time = time.time()
    
    # Process pool management (simple approach)
    processPool = []
    results_dict = {}
    completed = 0
    total_tasks = len(all_tasks)
    task_idx = 0
    
    while task_idx < total_tasks or len(processPool) > 0:
        # Check for finished processes
        for i in range(len(processPool) - 1, -1, -1):
            proc_info = processPool[i]
            result = proc_info['process'].poll()
            
            if result is not None:
                # Process finished
                task = proc_info['task']
                runtime = time.time() - proc_info['start_time']
                
                # Read stdout/stderr
                stdout = proc_info['process'].stdout.read() if proc_info['process'].stdout else ""
                stderr = proc_info['process'].stderr.read() if proc_info['process'].stderr else ""
                
                solved = "solved: True" in stdout
                
                # Parse metrics
                makespan = None
                sum_of_cost = None
                if solved and os.path.exists(task['output_file']):
                    makespan, sum_of_cost = parse_output_metrics(task['output_file'])
                
                # Save debug info for failed cases
                if not solved:
                    debug_file = task['output_file'].replace('.txt', '_debug.txt')
                    with open(debug_file, 'w') as f:
                        f.write(f"Command: python3 app.py -m {task['map_file']} -i {task['scen_file']} -N {task['num_agents']}\n")
                        f.write(f"PIBT Version: {task['version']}\n")
                        f.write(f"Max timestep: {task['max_timestep']}\n")
                        f.write(f"Runtime: {runtime:.2f}s\n")
                        f.write(f"Return code: {result}\n")
                        f.write(f"\nSTDOUT:\n{stdout}\n")
                        f.write(f"\nSTDERR:\n{stderr}\n")
                
                # Store result
                key = (task['map_file'], task['scen_file'], task['num_agents'], task['version'])
                results_dict[key] = {
                    'solved': solved,
                    'runtime': runtime,
                    'makespan': makespan,
                    'sum_of_cost': sum_of_cost,
                    'max_timestep': task['max_timestep'],
                    'map': task['map_name'],
                    'scenario': task['scen_name'],
                    'agents': task['num_agents']
                }
                
                # Print result
                completed += 1
                status = "✅" if solved else "❌"
                percent = (completed / total_tasks) * 100
                metrics_str = f", makespan={makespan}, soc={sum_of_cost}" if makespan else ""
                print(f"[{completed}/{total_tasks}] ({percent:.1f}%) {status} {task['version']}: "
                      f"{task['map_name']}/{task['scen_name']} ({runtime:.2f}s{metrics_str})")
                
                # Remove from pool
                processPool.pop(i)
        
        # Start new processes if pool not full and tasks remain
        while len(processPool) < NUM_PARALLEL and task_idx < total_tasks:
            task = all_tasks[task_idx]
            
            cmd = [
                "python3", "app.py",
                "-m", task['map_file'],
                "-i", task['scen_file'],
                "-N", str(task['num_agents']),
                "-o", task['output_file'],
                "--max-timestep", str(task['max_timestep']),
                "--pibt-version", task['version']
            ]
            
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processPool.append({
                    'process': proc,
                    'task': task,
                    'start_time': time.time()
                })
                task_idx += 1
            except Exception as e:
                print(f"❌ Failed to start: {task['map_name']}/{task['scen_name']}: {e}")
                task_idx += 1
        
        # Small sleep to avoid busy waiting
        time.sleep(0.1)
    
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
    
    # Detailed comparison table
    print(f"\n{'='*80}")
    print("Detailed Per-Experiment Comparison:")
    print(f"{'='*80}\n")
    print(f"{'Experiment':<40} {'Version':<12} {'Status':<12} {'Time':<10} {'Makespan':<10} {'SOC'}")
    print("-" * 100)
    
    comparison_data = []
    
    for i, (map_file, scen_file, num_agents) in enumerate(experiments):
        map_name = os.path.basename(map_file).replace('.map', '')
        scen_name = os.path.basename(scen_file).replace('.scen', '')
        exp_name = f"{map_name}/{scen_name} (N={num_agents})"
        
        exp_comparison = {
            'map': map_name,
            'scenario': scen_name,
            'agents': num_agents
        }
        
        # Show all versions for this experiment together
        for v_idx, version in enumerate(PIBT_VERSIONS):
            key = (map_file, scen_file, num_agents, version)
            result = results_dict.get(key, {})
            
            status = "✅ SOLVED" if result.get('solved') else "❌ FAILED"
            runtime = result.get('runtime', 0)
            makespan = result.get('makespan', '-')
            soc = result.get('sum_of_cost', '-')
            
            # Only show experiment name on first row
            exp_display = exp_name if v_idx == 0 else ""
            
            print(f"{exp_display:<40} {version:<12} {status:<12} {runtime:>8.2f}s  {str(makespan):>8}  {str(soc):>8}")
            
            exp_comparison[f'{version}_solved'] = result.get('solved', False)
            exp_comparison[f'{version}_time'] = result.get('runtime', 0)
            exp_comparison[f'{version}_makespan'] = result.get('makespan')
            exp_comparison[f'{version}_soc'] = result.get('sum_of_cost')
        
        print()
        
        # Highlight differences
        solved_versions = [v for v in PIBT_VERSIONS if results_dict.get((map_file, scen_file, num_agents, v), {}).get('solved')]
        failed_versions = [v for v in PIBT_VERSIONS if not results_dict.get((map_file, scen_file, num_agents, v), {}).get('solved')]
        
        if solved_versions and failed_versions:
            print(f"  ⚠️  Different results: {', '.join(solved_versions)} solved, {', '.join(failed_versions)} failed\n")
        elif len(solved_versions) == len(PIBT_VERSIONS) and len(PIBT_VERSIONS) > 1:
            # Compare quality metrics
            makespans = {v: results_dict.get((map_file, scen_file, num_agents, v), {}).get('makespan') for v in PIBT_VERSIONS}
            
            if all(makespans.values()):
                best_makespan = min(makespans.values())
                best_version = [v for v, m in makespans.items() if m == best_makespan][0]
                if max(makespans.values()) > best_makespan:
                    print(f"  🏆 {best_version} has best makespan ({best_makespan})\n")
        
        comparison_data.append(exp_comparison)
    
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