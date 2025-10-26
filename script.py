#!/usr/bin/env python3
"""
Improved parallel batch runner with ADAPTIVE timeout based on problem complexity.
Timeout scales with: agent count, map size, and observed difficulty.
"""

import subprocess
import os
import time
import glob
import sys
from collections import defaultdict
import re
import json

# Configuration
PIBT_VERSIONS = ["pibt_new"]  
NUM_PARALLEL = int(sys.argv[1]) if len(sys.argv) > 1 else 40

# Adaptive timeout configuration
BASE_TIMEOUT = 60  # 1 minute base
TIMEOUT_PER_AGENT = 4  # 4 seconds per agent
TIMEOUT_PER_1K_CELLS = 10  # 10 seconds per 1000 map cells
DENSITY_FACTOR = 1.5  # Multiply timeout by this for high-density instances
MAX_TIMEOUT = 1800  # 30 minutes absolute maximum
MIN_TIMEOUT = 60  # 1 minute minimum

# Agent range configuration - just specify the range you want for all maps
AGENT_RANGE = range(50, 101, 5)  # 50, 55, 60, ..., 100 for all maps

def calculate_adaptive_timeout(num_agents, map_width, map_height):
    """
    Calculate timeout based on problem complexity.
    
    Formula: BASE + (agents × TIMEOUT_PER_AGENT) + (map_area/1000 × TIMEOUT_PER_1K_CELLS)
    
    With density adjustment: if agents/area > 0.1, multiply by DENSITY_FACTOR
    """
    map_area = map_width * map_height
    density = num_agents / map_area
    
    # Base calculation
    timeout = BASE_TIMEOUT
    timeout += num_agents * TIMEOUT_PER_AGENT
    timeout += (map_area / 1000) * TIMEOUT_PER_1K_CELLS
    
    # High-density instances are harder
    if density > 0.1:
        timeout *= DENSITY_FACTOR
    
    # Apply bounds
    timeout = max(MIN_TIMEOUT, min(MAX_TIMEOUT, timeout))
    
    return int(timeout)

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
        
        makespan = len(lines)
        agent_paths = defaultdict(list)
        
        for line in lines:
            if ':' not in line:
                continue
            
            parts = line.split(':')
            if len(parts) < 2:
                continue
            
            timestep = int(parts[0])
            positions_str = parts[1]
            positions = re.findall(r'\((\d+),(\d+)\)', positions_str)
            
            for agent_idx, (x, y) in enumerate(positions):
                agent_paths[agent_idx].append((int(x), int(y)))
        
        sum_of_cost = 0
        for agent_idx, path in agent_paths.items():
            agent_cost = len(path) - 1  # Number of moves
            sum_of_cost += agent_cost
        
        return makespan, sum_of_cost
    
    except Exception as e:
        return None, None

def print_progress_bar(completed, total, width=50):
    """Print a progress bar"""
    percent = completed / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r[{bar}] {completed}/{total} ({percent*100:.1f}%)", end='', flush=True)

def write_progress_log(log_file, map_name, num_agents, scenario, status, runtime, makespan, soc, timeout_limit, is_timeout=False):
    """Append progress to log file organized by agent count"""
    with open(log_file, 'a') as f:
        status_icon = "✅" if status == "SOLVED" else ("⏱️" if is_timeout else "❌")
        metrics = f"runtime={runtime:.2f}s (limit={timeout_limit}s)"
        if makespan is not None:
            metrics += f", makespan={makespan}, soc={soc}"
        f.write(f"  {status_icon} {scenario:<40} {status:<12} {metrics}\n")
        f.flush()

def initialize_progress_logs(output_dir, experiments):
    """Create initial progress log files organized by agent count"""
    # Group experiments by map and agent count
    by_map_agents = defaultdict(lambda: defaultdict(list))
    
    for map_file, scen_file, num_agents in experiments:
        map_name = os.path.basename(map_file).replace('.map', '')
        scen_name = os.path.basename(scen_file).replace('.scen', '')
        by_map_agents[map_name][num_agents].append(scen_name)
    
    # Create log files
    for map_name, agents_dict in by_map_agents.items():
        for num_agents, scenarios in agents_dict.items():
            log_file = f"{output_dir}/progress_{map_name}_N{num_agents}.log"
            with open(log_file, 'w') as f:
                f.write(f"{'='*80}\n")
                f.write(f"Map: {map_name} | Agents: {num_agents} | Scenarios: {len(scenarios)}\n")
                f.write(f"{'='*80}\n\n")
    
    return by_map_agents

def main():
    # Create output directories
    os.makedirs("batch_results", exist_ok=True)
    for version in PIBT_VERSIONS:
        os.makedirs(f"batch_results/{version}", exist_ok=True)
    
    # Create progress logs directory
    progress_dir = "batch_results/progress_logs"
    os.makedirs(progress_dir, exist_ok=True)
    
    experiments = []
    map_count = 0
    
    # Special small test cases
    if os.path.exists("assets/small.map"):
        experiments.append(("assets/small.map", "assets/small-random-1.scen", 2))
        map_count += 1
    if os.path.exists("assets/pushmap.map"):
        experiments.append(("assets/pushmap.map", "assets/pushmap-random-1.scen", 5))
        map_count += 1
    
    # Find all maps and generate experiments with agent ranges
    for map_file in glob.glob("assets/*.map"):
        map_name = os.path.basename(map_file).replace('.map', '')
        scen_pattern = f"assets/scen/{map_name}-random-*.scen"
        scen_files = glob.glob(scen_pattern)

        scen_pattern_even = f"assets/scen/{map_name}-even-*.scen"
        scen_files += glob.glob(scen_pattern_even)
        
        if not scen_files:
            continue
        
        # Get map dimensions
        height, width = get_map_dimensions(map_file)
        
        # Calculate sample timeouts for display
        sample_timeout_50 = calculate_adaptive_timeout(50, width, height) if height and width else 0
        sample_timeout_100 = calculate_adaptive_timeout(100, width, height) if height and width else 0
        
        print(f"Map {map_name} ({width}×{height}, area={width*height}): {len(scen_files)} scenarios, "
              f"agents: {list(AGENT_RANGE)}, "
              f"timeout@50: {sample_timeout_50}s, @100: {sample_timeout_100}s")
        
        for scen_file in scen_files:
            for num_agents in AGENT_RANGE:
                experiments.append((map_file, scen_file, num_agents))
        
        map_count += 1
    
    print(f"\n{'='*80}")
    print(f"🚀 PARALLEL BATCH RUNNER WITH ADAPTIVE TIMEOUT")
    print(f"{'='*80}")
    print(f"Testing {len(PIBT_VERSIONS)} versions: {', '.join(PIBT_VERSIONS)}")
    print(f"Maps found: {map_count}")
    print(f"Total experiments per version: {len(experiments)}")
    print(f"Total runs: {len(experiments) * len(PIBT_VERSIONS)}")
    print(f"Parallel workers: {NUM_PARALLEL}")
    print(f"\nTimeout formula:")
    print(f"  Base: {BASE_TIMEOUT}s + {TIMEOUT_PER_AGENT}s/agent + {TIMEOUT_PER_1K_CELLS}s/1k cells")
    print(f"  Range: {MIN_TIMEOUT}s - {MAX_TIMEOUT}s")
    print(f"  Density penalty (>10%): ×{DENSITY_FACTOR}")
    print(f"\nProgress logs: {progress_dir}/progress_<map>_N<agents>.log")
    print(f"{'='*80}\n")
    
    if len(experiments) == 0:
        print("⚠️  No experiments found!")
        return
    
    # Initialize progress logs
    by_map_agents = initialize_progress_logs(progress_dir, experiments)
    
    # Prepare all tasks
    all_tasks = []
    for version in PIBT_VERSIONS:
        for map_file, scen_file, num_agents in experiments:
            height, width = get_map_dimensions(map_file)
            if height and width:
                max_timestep = height * width * num_agents
                timeout = calculate_adaptive_timeout(num_agents, width, height)
            else:
                max_timestep = 1000
                timeout = BASE_TIMEOUT
            
            map_name = os.path.basename(map_file).replace('.map', '')
            scen_name = os.path.basename(scen_file).replace('.scen', '')
            output_file = f"batch_results/{version}/{map_name}_{scen_name}_N{num_agents}.txt"
            log_file = f"{progress_dir}/progress_{map_name}_N{num_agents}.log"
            
            all_tasks.append({
                'map_file': map_file,
                'scen_file': scen_file,
                'num_agents': num_agents,
                'output_file': output_file,
                'max_timestep': max_timestep,
                'timeout': timeout,
                'version': version,
                'map_name': map_name,
                'scen_name': scen_name,
                'log_file': log_file,
                'map_width': width,
                'map_height': height
            })
    
    print(f"⏳ Starting {len(all_tasks)} experiments with adaptive timeouts...")
    print()
    start_time = time.time()
    
    # Process pool management
    processPool = []
    results_dict = {}
    completed = 0
    timeout_count = 0
    total_tasks = len(all_tasks)
    task_idx = 0
    
    last_update_time = time.time()
    last_completed = 0
    
    # Track timeout statistics
    timeout_by_agents = defaultdict(int)
    total_by_agents = defaultdict(int)
    
    while task_idx < total_tasks or len(processPool) > 0:
        # Check for finished processes and timeouts
        for i in range(len(processPool) - 1, -1, -1):
            proc_info = processPool[i]
            result = proc_info['process'].poll()
            elapsed = time.time() - proc_info['start_time']
            
            # Check for timeout (adaptive per task)
            task_timeout = proc_info['task']['timeout']
            is_timeout = False
            if elapsed > task_timeout and result is None:
                proc_info['process'].kill()
                proc_info['process'].wait()
                result = -1  # Timeout signal
                is_timeout = True
                timeout_count += 1
                timeout_by_agents[proc_info['task']['num_agents']] += 1
            
            if result is not None:
                task = proc_info['task']
                runtime = elapsed
                
                total_by_agents[task['num_agents']] += 1
                
                stdout = proc_info['process'].stdout.read() if proc_info['process'].stdout else ""
                stderr = proc_info['process'].stderr.read() if proc_info['process'].stderr else ""
                
                solved = "solved: True" in stdout and not is_timeout
                
                makespan = None
                sum_of_cost = None
                if solved and os.path.exists(task['output_file']):
                    makespan, sum_of_cost = parse_output_metrics(task['output_file'])
                
                # Save debug info for failed/timeout cases
                if not solved:
                    debug_file = task['output_file'].replace('.txt', '_debug.txt')
                    with open(debug_file, 'w') as f:
                        f.write(f"Command: python3 app.py -m {task['map_file']} -i {task['scen_file']} -N {task['num_agents']}\n")
                        f.write(f"PIBT Version: {task['version']}\n")
                        f.write(f"Map size: {task['map_width']}×{task['map_height']}\n")
                        f.write(f"Max timestep: {task['max_timestep']}\n")
                        f.write(f"Timeout limit: {task_timeout}s\n")
                        f.write(f"Runtime: {runtime:.2f}s\n")
                        f.write(f"Timeout: {is_timeout}\n")
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
                    'timeout_limit': task_timeout,
                    'map': task['map_name'],
                    'scenario': task['scen_name'],
                    'agents': task['num_agents'],
                    'timeout': is_timeout
                }
                
                # Write to progress log
                status = "SOLVED" if solved else ("TIMEOUT" if is_timeout else "FAILED")
                write_progress_log(
                    task['log_file'],
                    task['map_name'],
                    task['num_agents'],
                    task['scen_name'],
                    status,
                    runtime,
                    makespan,
                    sum_of_cost,
                    task_timeout,
                    is_timeout
                )
                
                completed += 1
                processPool.pop(i)
        
        # Print progress every 2 seconds or when completed changes
        current_time = time.time()
        if completed != last_completed or (current_time - last_update_time > 2):
            print_progress_bar(completed, total_tasks)
            
            # Show some stats
            if completed > 0:
                elapsed = current_time - start_time
                rate = completed / elapsed
                remaining = total_tasks - completed
                eta = remaining / rate if rate > 0 else 0
                
                solved_count = sum(1 for r in results_dict.values() if r['solved'])
                success_rate = (solved_count / completed * 100) if completed > 0 else 0
                
                timeout_display = f" | {timeout_count} timeouts" if timeout_count > 0 else ""
                
                print(f" | {solved_count}/{completed} solved ({success_rate:.1f}%){timeout_display} | "
                      f"ETA: {eta/60:.1f}m", end='')
            
            last_update_time = current_time
            last_completed = completed
        
        # Start new processes
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
                print(f"\n❌ Failed to start: {task['map_name']}/{task['scen_name']}: {e}")
                task_idx += 1
        
        time.sleep(0.1)
    
    # Final progress bar
    print_progress_bar(completed, total_tasks)
    print()  # New line after progress bar
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Average time per experiment: {total_time/total_tasks:.2f}s")
    if timeout_count > 0:
        print(f"⚠️  Timeouts: {timeout_count}/{total_tasks} ({timeout_count/total_tasks*100:.1f}%)")
    print(f"{'='*80}\n")
    
    # Organize results
    all_results = defaultdict(list)
    for version in PIBT_VERSIONS:
        for map_file, scen_file, num_agents in experiments:
            key = (map_file, scen_file, num_agents, version)
            if key in results_dict:
                all_results[version].append(results_dict[key])
    
    # Generate summary
    print(f"\n{'='*80}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Version':<15} {'Solved':<12} {'Timeout':<10} {'Rate':<8} {'Avg Time':<12} {'Avg Makespan':<15} {'Avg SOC'}")
    print("-" * 100)
    
    for version in PIBT_VERSIONS:
        results = all_results[version]
        solved_count = sum(1 for r in results if r['solved'])
        timeout_count_v = sum(1 for r in results if r.get('timeout', False))
        avg_time = sum(r['runtime'] for r in results) / len(results) if results else 0
        success_rate = solved_count / len(results) * 100 if results else 0
        
        makespans = [r['makespan'] for r in results if r['solved'] and r['makespan'] is not None]
        socs = [r['sum_of_cost'] for r in results if r['solved'] and r['sum_of_cost'] is not None]
        
        avg_makespan = sum(makespans) / len(makespans) if makespans else 0
        avg_soc = sum(socs) / len(socs) if socs else 0
        
        print(f"{version:<15} {solved_count}/{len(results):<9} {timeout_count_v:<10} {success_rate:>5.1f}%  "
              f"{avg_time:>8.2f}s    {avg_makespan:>10.1f}      {avg_soc:>10.1f}")
    
    # Agent scaling analysis with timeout info
    print(f"\n{'='*80}")
    print("📈 SCALING BY AGENT COUNT (with adaptive timeouts)")
    print(f"{'='*80}\n")
    
    agent_counts = sorted(set(exp[2] for exp in experiments))
    print(f"{'Agents':<10} {'Tested':<10} {'Solved':<10} {'Timeout':<10} {'Rate':<8} {'Avg Time':<12} {'Avg Timeout':<12} {'Avg Makespan'}")
    print("-" * 100)
    
    for num_agents in agent_counts:
        relevant_results = [r for r in all_results[PIBT_VERSIONS[0]] if r['agents'] == num_agents]
        if not relevant_results:
            continue
        
        solved = sum(1 for r in relevant_results if r['solved'])
        timeout_c = sum(1 for r in relevant_results if r.get('timeout', False))
        rate = solved / len(relevant_results) * 100 if relevant_results else 0
        avg_time = sum(r['runtime'] for r in relevant_results) / len(relevant_results)
        avg_timeout_limit = sum(r['timeout_limit'] for r in relevant_results) / len(relevant_results)
        
        makespans = [r['makespan'] for r in relevant_results if r['solved'] and r['makespan']]
        avg_makespan = sum(makespans) / len(makespans) if makespans else 0
        
        print(f"{num_agents:<10} {len(relevant_results):<10} {solved:<10} {timeout_c:<10} {rate:>5.1f}%  "
              f"{avg_time:>8.2f}s    {avg_timeout_limit:>8.1f}s    {avg_makespan:>10.1f}")
    
    # Timeout analysis
    if timeout_count > 0:
        print(f"\n{'='*80}")
        print("⏱️  TIMEOUT ANALYSIS")
        print(f"{'='*80}\n")
        print("Agent counts with most timeouts:")
        sorted_timeouts = sorted(timeout_by_agents.items(), key=lambda x: x[1], reverse=True)[:10]
        for agents, count in sorted_timeouts:
            total = total_by_agents[agents]
            pct = count / total * 100 if total > 0 else 0
            print(f"  {agents} agents: {count}/{total} ({pct:.1f}%) timed out")
    
    # Save detailed results
    with open("batch_results/comparison_summary.txt", "w") as f:
        f.write("PIBT BATCH EXPERIMENT RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"Versions: {', '.join(PIBT_VERSIONS)}\n")
        f.write(f"Total experiments: {len(experiments)}\n")
        f.write(f"Adaptive timeout formula:\n")
        f.write(f"  {BASE_TIMEOUT}s + {TIMEOUT_PER_AGENT}s×agents + {TIMEOUT_PER_1K_CELLS}s×(area/1000)\n")
        f.write(f"  Density penalty (>10%): ×{DENSITY_FACTOR}\n")
        f.write(f"  Range: {MIN_TIMEOUT}s - {MAX_TIMEOUT}s\n")
        f.write(f"Timeouts occurred: {timeout_count}\n\n")
        
        # Copy console output
        for version in PIBT_VERSIONS:
            results = all_results[version]
            solved_count = sum(1 for r in results if r['solved'])
            f.write(f"{version}: {solved_count}/{len(results)} solved ({solved_count/len(results)*100:.1f}%)\n")
    
    print(f"\n{'='*80}")
    print("✨ DONE! Results saved to:")
    print(f"  📄 batch_results/comparison_summary.txt (main summary)")
    print(f"  📁 batch_results/progress_logs/ (detailed progress by map & agent count)")
    print(f"  📁 batch_results/{PIBT_VERSIONS[0]}/ (individual results)")
    print(f"{'='*80}\n")
    
    # Suggest timeout adjustments if needed
    if timeout_count > total_tasks * 0.2:  # More than 20% timeouts
        print("💡 SUGGESTION: High timeout rate detected!")
        print("   Consider increasing timeout parameters:")
        print(f"   - BASE_TIMEOUT (currently {BASE_TIMEOUT}s)")
        print(f"   - TIMEOUT_PER_AGENT (currently {TIMEOUT_PER_AGENT}s)")
        print(f"   - MAX_TIMEOUT (currently {MAX_TIMEOUT}s)")
        print()
    
    # Play a beep sound (works on most systems)
    print("\a")  # Terminal bell

if __name__ == "__main__":
    main()