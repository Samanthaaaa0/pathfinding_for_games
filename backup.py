from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np

from .dist_table import DistTable
from .utils import Config, Configs, Coord, Grid, get_neighbors, is_valid_coord

@dataclass
class SwapGroup:
    agents: Set[int]
    vertex: Coord
    reserved_area: Set[Coord]
    start_time: int
    duration: int

class PIBT:
    def __init__(self, grid: Grid, starts: Config, goals: Config, seed: int = 0):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.num_agents = len(starts)
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

        self.NIL = self.num_agents  # meaning \bot
        self.NIL_COORD: Coord = (-1, -1)  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        self.new_configs = []  

        self.restore = False
        self.restore_paths = {}  # agent_id -> stack of positions to restore
        self.swapping_agents = set()

        self.movement_stack = {}  # agent_id -> stack of (old_pos, new_pos, is_exchange_move)
        self.exchange_positions = {}  # agent_id -> final position after all exchanges
        self.in_swap_operation = False

        # Parallel execution state - just simple arrays
        self.current_timestep = 0
        self.agent_states = ["pibt"] * self.num_agents  # "pibt", "swap", "clearing", "waiting"
        self.agent_plans = {}  # agent_id -> List[Coord]
        self.plan_steps = [0] * self.num_agents  # current step in plan for each agent
        self.wait_until = [0] * self.num_agents  # timestep when agent can move again
        self.active_swaps = []  # List[SwapGroup]

    def funcPIBT(self, i_from: Config, i_moveto: Config, i: int = 0, root_agent: int = None) -> bool:
        """
        Recursive function to implement the PIBT algorithm.
        root_agent tracks the original high-priority agent that started the chain.
        """
        if i >= self.num_agents:
            return True
        
        if self.restore:
            return self.handle_restore_agent(i, i_from, i_moveto)

        # Set root_agent to current agent if this is the top-level call
        if root_agent is None:
            root_agent = i

        # get candidate configurations
        candidate = [i_from[i]] + list(get_neighbors(self.grid, i_from[i]))
        self.rng.shuffle(candidate)  # tie-breaking, randomize
        candidate = sorted(candidate, key=lambda u: self.dist_tables[i].get(u))

        # Check if swap is required and possible
        k = None
        if len(candidate) > 1:  # Only check if there are actual neighbors
            k = self.swap_required_and_possible(i, candidate[0], i_from)
        
        # Line 4: If swap is needed, reverse candidate order
        if k is not None:
            candidate.reverse()
            print(f"[PIBT_SWAP] Agent {i} reversed candidates due to swap with {k}")

        # FIXED: Track potential swap candidates while trying all other options first
        potential_swap_candidates = set()

        for v in candidate:
            if v == i_from[i] and potential_swap_candidates:
                break

            # Check for vertex conflict - exclude nodes that are already requested by others
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # Avoid swap conflict - EXCLUDE previous position it inherited from
            if j != self.NIL and i_moveto[j] == i_from[i]:
                continue

            # reserve next location
            i_moveto[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the avoid edge conflict condition)
            if (
                j != self.NIL
                and (i_moveto[j] == self.NIL_COORD)
                and (not self.funcPIBT(i_from, i_moveto, j, root_agent))  # Pass root_agent down
            ):
                
                i_moveto[i] = self.NIL_COORD
                self.occupied_nxt[v] = self.NIL

                # save as potential swap candidate
                if i == root_agent:
                    potential_swap_candidates.add((j, v))
                    print("Potential swap:", potential_swap_candidates)

                continue 

            if v == candidate[0] and k is not None and i_moveto[k] == self.NIL_COORD:
                # Pull agent j to i's current location
                i_moveto[k] = i_from[i]
                self.occupied_nxt[i_from[i]] = k
                print(f"[PIBT_SWAP] Executed swap: Agent {i} -> {v}, Agent {k} -> {i_from[i]}")

            # Success! Found a valid move
            return True
        
        if i == root_agent and potential_swap_candidates:
            print(f"[PIBT] Root agent {i} exhausted all candidates, trying swaps...")

            i_moveto[i] = v
            self.occupied_nxt[v] = i
            
            # Try swaps in order of preference (closest to goal first)
            list(potential_swap_candidates).sort(key=lambda x: self.dist_tables[i].get(x[1]))
            
            for j, v in potential_swap_candidates:
                print(f"[PIBT] Root agent {i} trying swap with {j} at vertex {v}")
                if self.try_swap(i, j, i_from, i_moveto):
                    print(f"[PIBT] Swap success: A{i} <-> A{j}")
                    return True
                else:
                    print(f"[PIBT] Swap failed: A{i} <-> A{j}")
        
        # failed to secure node
        i_moveto[i] = i_from[i]
        self.occupied_nxt[i_from[i]] = i
        print(f"[PIBT] Agent {i} failed to move, staying at {i_from[i]}")
        return False

    def step(self, i_from: Config, priorities: list[float]) -> Config:

        # setup
        N = len(i_from)
        i_moveto: Config = []

        for i, v in enumerate(i_from):
            i_moveto.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if i_moveto[i] == self.NIL_COORD:
                self.funcPIBT(i_from, i_moveto, i)

        # cleanup
        for i in range(N):
            if i_moveto[i] == self.NIL_COORD:
                i_moveto[i] = i_from[i]
                if self.occupied_nxt[i_from[i]] == self.NIL:
                    self.occupied_nxt[i_from[i]] = i

        self.occupied_now.fill(self.NIL)
        self.occupied_nxt.fill(self.NIL)

        if self.new_configs:
            # Return only the first step of the multi-step sequence
            # Store the rest for future steps
            next_config = self.new_configs.pop(0)
            print(f"[STEP] Executing swap step, {len(self.new_configs)} steps remaining")
            return next_config

        if self.restore:
            restore_complete = all(
                not self.restore_paths.get(i, []) for i in range(self.num_agents)
            )
            if restore_complete:
                print("[RESTORE] Restore phase complete")
                self.restore = False
                self.restore_paths.clear()
                self.swapping_agents.clear()

        return i_moveto

    def run(self, max_timestep: int = 50) -> Configs:
        # identify priorities
        priorities: list[float] = []
        for i in range(self.num_agents):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        # main loop, generate sequence of configurations
        configs = [self.starts.copy()]
        # print("Step 0:", configs[0])

        while len(configs) <= max_timestep:
            # obtain new configuration
            current_config = configs[-1].copy()
            Q = self.step(current_config, priorities)
            
            if self.new_configs != []:
                configs += self.new_configs.copy()
                self.new_configs = []
            else:
                configs.append(Q)

            Q = configs[-1]
            # print(f"Step {len(configs) - 1}: {Q}")

            # update priorities & goal check
            # todo: comment it so wont occur "priority problem"
            flg_fin = True
            for i in range(self.num_agents):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                # else:
                #     priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break  # goal

        configs = self.remove_redundant_moves(configs)

        # for i in range(len(configs)):
        #     print(f"Step {i}: {configs[i]}")

        return configs
    
    def clear_vertex(self, Pre, v: Coord, current_config: Config, U: set[Coord], i_moveto: Config, reserved_positions) -> bool:
        # Find which agent is at vertex v
        agent_at_v = self.get_agent_at_position(current_config, v)
        
        if agent_at_v == self.NIL:
            return True  # Vertex is already clear
        
        # print(f"[CLEAR_VERTEX] Trying to clear vertex {v}, occupied by agent {agent_at_v}")
        
        # Check neighbors of v for empty spots
        for neighbor in get_neighbors(self.grid, v):
            if neighbor in U:
                continue
                
            # Check if neighbor is empty in current config
            agent_at_neighbor = self.get_agent_at_position(current_config, neighbor)
            # if agent_at_neighbor == self.NIL and neighbor not in reserved_positions:
            if agent_at_neighbor == self.NIL \
                and neighbor not in i_moveto \
                and neighbor not in reserved_positions:

                self.movement_stack[agent_at_v].append({current_config[agent_at_v], neighbor, False})

                # Reserve the neighbor to prevent collisions
                reserved_positions.add(neighbor)
                # Found empty neighbor, move agent there
                i_moveto[agent_at_v] = neighbor
                Pre[agent_at_v].append(neighbor)
                current_config[agent_at_v] = neighbor 

                # print(f"[CLEAR_VERTEX] Moved agent {agent_at_v} from {v} to {neighbor}")
                return True
        
        # print(f"[CLEAR_VERTEX] Failed to clear vertex {v}")
        return False

    def grid_degree(self, v: Coord) -> int:
        return sum(1 for nbr in get_neighbors(self.grid, v) if is_valid_coord(self.grid, nbr))

    def exchange(self, Pi, r0, s0, v, i_from, i_moveto):
        """
        Modified exchange method that allows other agents to move in parallel
        """
        current_config = i_from
        # print(f"[EXCHANGE] Current config: {current_config}")
        
        # identify agent positions - r on v, s not on v
        if current_config[r0] == v:
            r, s = r0, s0
        else:
            r, s = s0, r0

        # find 2 unoccupied neighbors of v
        neighbors = get_neighbors(self.grid, v)
        occupied = set(Pi[-1])
        unoccupied = [n for n in neighbors if n not in occupied]
        print(f"[EXCHANGE] Unoccupied neighbors of {v}: {unoccupied}")
        if len(unoccupied) < 2:
            return False

        v1, v2 = unoccupied[:2]
        vs = current_config[s]  # Position of s before exchange

        exchange_configs = []

        print(f"[EXCHANGE] v: {v}, v1: {v1}, v2: {v2}, vs: {vs}")

        # Create Pre structure for the exchange sequence
        Pre_exchange = {}
        for k in range(self.num_agents):
            Pre_exchange[k] = []

        # Define the exchange sequence for agents r and s
        exchange_steps = [
            (r, v1, s, v),      # step1: r->v1, s->v
            (s, v2),            # step2: s->v2  
            (r, v),             # step3: r->v
            (r, vs, s, v)       # step4: r->vs, s->v
        ]

        # Build Pre_exchange paths
        for step in exchange_steps:
            if len(step) == 4:  # Both agents move
                agent1, pos1, agent2, pos2 = step
                Pre_exchange[agent1].append(pos1)
                Pre_exchange[agent2].append(pos2)
            else:  # Single agent moves
                agent, pos = step
                Pre_exchange[agent].append(pos)

        # Generate configurations using the same approach as clear/move methods
        temp_config = current_config.copy()
        self.generate_config(Pi, Pre_exchange, temp_config, i_moveto)

        # Update final positions
        final_config = Pi[-1]
        i_moveto[r] = final_config[r]
        i_moveto[s] = final_config[s]

        return True

    def find_high_degree_vertices(self, curr, i_from, max_candidates=3, max_radius=10) -> list[Coord]:

        start = curr
        queue = [(start, 0)]  # (pos, distance)
        visited = {start}
        parent = {start: None}
        candidates = []

        while queue and len(candidates) < max_candidates:
            pos, dist = queue.pop(0)

            if self.grid_degree(pos) >= 3:
                path = []
                node = pos
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                candidates.append((pos, path))

            if dist < max_radius:
                for nbr in get_neighbors(self.grid, pos):
                    if nbr not in visited:
                        visited.add(nbr)
                        parent[nbr] = pos
                        queue.append((nbr, dist + 1))

        return candidates

    def clear(self, Pre, r0, s0, v, i_from, i_moveto, path):
        reserved_positions = set()
        # print(f"..[CLEAR] clearing for {r0} and {s0}")
        current_config = i_from.copy()

        # Check if the high-degree vertex v is occupied
        agent_at_v = self.get_agent_at_position(current_config, v)

        if agent_at_v != self.NIL and agent_at_v not in {r0, s0}:
            # print(f"[CLEAR] High-degree vertex {v} is occupied by agent {agent_at_v}")
            
            # First, try to move the agent at v to a safe location
            v_neighbors = get_neighbors(self.grid, v)
            for safe_spot in v_neighbors:
                if (safe_spot not in current_config and 
                    safe_spot not in reserved_positions):
                    
                    print(f"[CLEAR] Moving agent {agent_at_v} from {v} to {safe_spot}")
                    Pre[agent_at_v].append(safe_spot)
                    current_config[agent_at_v] = safe_spot
                    reserved_positions.add(safe_spot)
                    self.movement_stack[agent_at_v].append((i_from[agent_at_v], safe_spot, False))
                    break
            else:
                # If no immediate neighbor is free, try to clear one
                for safe_spot in v_neighbors:
                    U = {v}  # Avoid moving back to v
                    if self.clear_vertex(Pre, safe_spot, current_config, U, i_moveto, reserved_positions):
                        print(f"[CLEAR] Cleared {safe_spot} and moving agent {agent_at_v} there")

                        self.movement_stack[agent_at_v].append((i_from[agent_at_v], safe_spot, False))
                        Pre[agent_at_v].append(safe_spot)
                        current_config[agent_at_v] = safe_spot
                        reserved_positions.add(safe_spot)
                        break
                else:
                    # print(f"[CLEAR_STACK] Failed to relocate agent {agent_at_v} from vertex {v}")
                    return False

        # assuming r is the agent at v (path[-1]), and s is the other agent beside v (path[-2])
        r, s = r0, s0
        v0 = path[-2] if len(path) > 1 else current_config[s]

        neighbors = get_neighbors(self.grid, v)
        occupied_positions = set(current_config)
        E = [n for n in neighbors if n not in occupied_positions and n != v0]

        # print(f"[CLEAR] Clearing neighbors of {v}, initially empty: {E}")

        if len(E) >= 3:
            return True

        # -------------------------------
        # Stage 1
        # -------------------------------
        for n in neighbors:
            if n in E or n == v0:
                continue
            U = {v, v0}  # avoid pushing into the r's goal vertex and s' goal vertex
            if self.clear_vertex(Pre, n, current_config, U, i_moveto, reserved_positions):
                # print(f"[CLEAR] Successfully cleared neighbor {n}")

                E.append(n)
                if len(E) >= 3:
                    return True
                

        if len(E) == 0:
            # print("[CLEAR] No empty neighbors available")
            return False
        
        # print("[CLEAR] Stage 1 FAILED ...............................")

        # -------------------------------
        # Stage 2
        # Use a neighbor n that can reach the empty spot ε without passing through v
        # -------------------------------

        ε = E[0]
        # print(f"[CLEAR] Empty neighbor ε: {ε}, neighbors: {neighbors}")

        for n in neighbors:
            if n in {v0, ε}:
                continue

            Pi0 = Pre.copy()  # Start with the current Pre

            if self.clear_vertex(Pi0, n, current_config, {v, v0}, i_moveto, reserved_positions) and \
            self.clear_vertex(Pi0, ε, current_config, {v, v0, n}, i_moveto, reserved_positions):

                for i in Pi0.keys():
                    for k in Pi0[i]:
                        if k not in Pre[i]:
                            Pre[i].append(k)
                # print(f"[CLEAR] Successfully cleared vertex {n} and {ε}")
                return True
            
        # print("[CLEAR] Stage 2 FAILED ...............................")

        # -------------------------------
        # Stage 3
        # -------------------------------
        '''
        - move r to ε  
        - move s to v  
        - clear vertex(n, {v, ε})  
        - clear vertex(v₀, {v, ε, n})
        '''

        for n in neighbors:
            if n in {v0, ε}:
                continue
            A0 = i_from.copy()
            # print("A0:", A0, "r:", r, "s:", s, "v:", v, "ε:", ε, "n:", n)

            # Move r to ε if free
            if ε in reserved_positions or ε in current_config:
                continue
            reserved_positions.add(ε)
            A0[r] = ε

            # Move s to v if free
            if v in reserved_positions or v in current_config:
                continue
            reserved_positions.add(v)
            A0[s] = v
                
            Pi0 = {i: [] for i in range(self.num_agents)}
            Pi0[r].append(ε)
            Pi0[s].append(v)


            # Try to clear n, avoiding v and ε
            if not self.clear_vertex(Pi0, n, A0, {v, ε}, i_moveto, reserved_positions):
                continue

            # i_moveto.update(A0)
            for i in Pi0.keys():
                for k in Pi0[i]:
                    if k not in Pre[i]:
                        Pre[i].append(k)
                        self.movement_stack[i].append((i_from[i], k, False))
            return True
        
        # print("[CLEAR] Stage 3 FAILED ...............................")

        # -------------------------------
        # Stage 4 - Make Space Behind the Empty Spot
        # -------------------------------
        '''
        - clear vertex(ε, {v, v₀, Pi(s)}) → try clearing ε - tot its already clear?
        - move t (agent on n) through v to ε  
        - clear vertex(ε, {v, v₀, n})
        '''
        
        # Choose a neighbor n != v0 or ε
        n = next((x for x in neighbors if x not in {v0, ε}), None)
        print("n:",n,"e:",ε,"v0:",v0,"v:",v)
        if n is None:
            return False
        
        occupied_positions = set(current_config)

        if n in occupied_positions:
            t = self.get_agent_at_position(current_config, n)
            print("t:",t)

        # Move t -> v -> ε
        A1 = current_config.copy()
        reserved_positions.add(v)
        A1[t] = v

        A2 = A1.copy()
        reserved_positions.add(ε)
        A2[t] = ε

        Pre[t].append(v)
        Pre[t].append(ε)
        self.movement_stack[t].append((i_from[t], v, False))
        self.movement_stack[t].append((v, ε, False))

        # print(f"[CLEAR] Moved agent {t} from {n} to {v} and then to {ε}")
        
        return self.clear_vertex(Pre, ε, A2, {v, v0, n}, i_moveto, reserved_positions)

    def get_agent_at_position(self, config: Config, pos: Coord) -> int:
        """Helper function to find which agent is at a given position"""
        # print("config:", config, "pos:", pos)
        try:
            return config.index(pos)
        except ValueError:
            return self.NIL
        
    def handle_restore_agent(self, i: int, i_from: Config, i_moveto: Config) -> bool:
        """
        Handle agent movement during restore mode.
        """
        # If agent has restore path, try to follow it
        if i in self.restore_paths and self.restore_paths[i]:
            target = self.restore_paths[i][0]  # peek at next position
            
            # Check if target is available
            if self.occupied_nxt[target] == self.NIL:
                j = self.occupied_now[target]
                
                # Check for swap conflict
                if j == self.NIL or i_moveto[j] != i_from[i]:
                    # Move to target and pop from restore path
                    i_moveto[i] = target
                    self.occupied_nxt[target] = i
                    self.restore_paths[i].pop(0)
                    print(f"[RESTORE] Agent {i} moved to {target}, remaining path: {self.restore_paths[i]}")
                    return True 
        
        # If can't follow restore path or no path, use normal PIBT logic
        # but prioritize moving toward goal
        self.restore_paths[i] = []  # Clear restore path for this agent
        print(f"[RESTORE] Agent {i} cannot follow restore path, using PIBT logic")

        # # Stay in place if no good move
        i_moveto[i] = i_from[i]
        self.occupied_nxt[i_from[i]] = i
        return False

    def generate_config(self, Pi, Pre, i_from, i_moveto):
        while any(Pre[o] for o in range(self.num_agents)):
            moved = False
            new_path = Pi[-1].copy() if Pi else i_from.copy()

            for k in range(self.num_agents):

                if not Pre[k]:  # No path left
                    continue

                target = Pre[k][0]

                if target is None:
                    continue

                i_moveto[k] = target
                self.occupied_nxt[target] = k
                Pre[k].pop(0)
                new_path[k] = target
                moved = True
                
            Pi.append(new_path.copy())

            if not moved:  # To prevent infinite loop if no agent can move
                break

        return Pi

    def clear_path_to_vertex(self, Pre, path, current_config, i_moveto, agent_id) -> bool:
        """
        Clear agents from the path positions (except start and end).
        This should be called before using the existing clear() method.
        """
        if len(path) <= 2:
            return True  # No intermediate positions to clear
        
        print(f"[CLEAR_PATH] Clearing path: {path}")
        
        # Clear intermediate positions in the path
        for pos in path[1:-1]:  # Skip start and end positions
            agent_at_pos = self.get_agent_at_position(current_config, pos)
            
            if agent_at_pos != self.NIL:
                agent_priority = self.dist_tables[agent_id].get(current_config[agent_id])
                blocking_priority = self.dist_tables[agent_at_pos].get(current_config[agent_at_pos])

                if blocking_priority > agent_priority:
                    return False

                # Try to find ANY available neighbor, not just those outside the path
                safe_pos = None
                neighbors = get_neighbors(self.grid, pos)
                
                # First try neighbors not in current config and not in path
                for neighbor in neighbors:
                    if (neighbor not in current_config and 
                        neighbor not in path):
                        safe_pos = neighbor
                        break
                
                # If that fails, try any empty neighbor (even if in path)
                if not safe_pos:
                    for neighbor in neighbors:
                        if neighbor not in current_config:
                            safe_pos = neighbor
                            break
                
                # If still no space, use clear_vertex to make space
                if not safe_pos:
                    for neighbor in neighbors:
                        if self.clear_vertex(Pre, neighbor, current_config, {pos}, i_moveto, set()):
                            safe_pos = neighbor
                            break
                
                if safe_pos:
                    self.movement_stack[agent_at_pos].append((current_config[agent_at_pos], neighbor, False))
                    Pre[agent_at_pos].append(safe_pos)
                    current_config[agent_at_pos] = safe_pos
                    print(f"[CLEAR_PATH] Moved agent {agent_at_pos} from {pos} to {safe_pos}")
                else:
                    print(f"[CLEAR_PATH] Could not find safe position for agent {agent_at_pos} at {pos}")
                    return False
        
        return True

    def optimize_restore_paths(self, original_config, normal_paths: list[Config]) -> None:
        """
        Simulate normal PIBT from original positions and cancel redundant moves.
        """
        print("[OPTIMIZE] Simulating normal PIBT to optimize restore paths")
        
        # Create a temporary PIBT instance to simulate normal movement
        current_pos = original_config.copy()
        max_steps = max(len(path) for path in self.restore_paths.values()) if self.restore_paths else 0

        # normal_paths = self.pibt_simulation.copy()  # Store the sequence of normal PIBT configurations

        for agent in range(self.num_agents):
            if agent not in self.restore_paths or not self.restore_paths[agent]:
                continue

            # If agent has a restore path, simulate normal PIBT to see if it conflicts
            restore_target = self.restore_paths[agent][0]  # Next restore position
            normal_target = normal_paths[0][agent]  # Where normal PIBT wants to go
        
            # If normal PIBT wants to go to the same place as restore, cancel both
            if normal_target == restore_target:
                # todo: remove from restore path (the whole path at once after checking all)
                print(f"[OPTIMIZE] Canceled redundant move for agent {agent} to {restore_target}")
            # If they want to go to different places, keep restore path
            elif normal_target != current_pos[agent]:
                # Normal PIBT wants to move somewhere else, prioritize restore
                pass       
        
            # Update current positions based on remaining restore paths or normal movement
            for agent in range(self.num_agents):
                if agent in self.restore_paths and self.restore_paths[agent]:
                    current_pos[agent] = self.restore_paths[agent][0]
                    self.restore_paths[agent].pop(0)
                else:
                    current_pos[agent] = normal_paths[agent]

    def simulate_normal_pibt_sequence(
        self,
        start_config: Config,
        max_steps: int = 10
    ) -> tuple[list[Config], list[tuple[int, int]]]:
        """
        Simulate normal PIBT for multiple steps without swap operations.
        If assume_swapped=(i,j), the start_config is modified as if i and j already swapped.
        Returns (sequence of configurations, detected swap chain).
        """
        print(f"[SIMULATE] Simulating normal PIBT for {max_steps} steps...")

        # Disable swaps temporarily
        original_try_swap = self.try_swap
        self.try_swap = lambda *args: False  

        # Save & reset restore state
        old_restore = self.restore
        old_restore_paths = self.restore_paths.copy()
        old_swapping_agents = self.swapping_agents.copy()
        self.restore = False
        self.restore_paths = {}
        self.swapping_agents = set()

        simulation_sequence = [start_config.copy()]
        current_config = start_config.copy()

        for step in range(max_steps):
            priorities = []
            for agent_id in range(self.num_agents):
                dist = self.dist_tables[agent_id].get(current_config[agent_id])
                priorities.append(dist / self.grid.size + step)

            next_config = self.step(current_config, priorities)
            simulation_sequence.append(next_config.copy())
            current_config = next_config

            if all(current_config[aid] == self.goals[aid] for aid in range(self.num_agents)):
                print(f"[SIMULATE] All agents reached goals at step {step+1}")
                break

        # Restore methods & state
        self.try_swap = original_try_swap
        self.restore = old_restore
        self.restore_paths = old_restore_paths
        self.swapping_agents = old_swapping_agents

        print(f"[SIMULATE] Generated {len(simulation_sequence)} configurations")

        return simulation_sequence

    def move_agents_to_high_vertex(self, Pre: dict, agents: list[int], v: Coord, temp_from: Config, i_moveto: Config, path: list):
        """
        Move agents to high-degree vertex - FIXED for 2-agent case
        """
        # print(f"[MOVE GROUP] Moving agents {agents} to high-degree vertex {v}")
        print(f"[MOVE GROUP] Path: {path}")
        
        if not path or len(agents) < 2:
            return False
        
        # Find the agent already on the path
        path_agent = None
        start_idx = 0
        
        for agent in agents:
            if temp_from[agent] in path:
                path_agent = agent
                start_idx = path.index(temp_from[agent])
                break
        
        if path_agent is None:
            print("[MOVE GROUP] No agent on path, moving first agent to path start")
            path_agent = agents[0]
            self.movement_stack[path_agent].append((temp_from[path_agent], path[0], False))
            Pre[path_agent].append(path[0])
            temp_from[path_agent] = path[0]
            start_idx = 0
        
        # print(f"[MOVE GROUP] Lead agent {path_agent} starting at path index {start_idx}")
        
        # FIXED: Ensure the lead agent reaches the target vertex v
        target_idx = path.index(v) if v in path else len(path) - 1
        remaining_path = path[start_idx + 1:target_idx + 1]  # Include target vertex
        
        # Get other agents that need to follow
        followers = [a for a in agents if a != path_agent]
        
        for step, next_pos in enumerate(remaining_path):
            # print(f"[MOVE GROUP] Step {step + 1}:")
            
            # Lead agent moves forward
            old_pos = temp_from[path_agent]
            self.movement_stack[path_agent].append((old_pos, next_pos, False))
            Pre[path_agent].append(next_pos)
            temp_from[path_agent] = next_pos
            # print(f"  Lead agent {path_agent}: {old_pos} -> {next_pos}")
            
            # FIXED: Move followers in sequence, ensuring they follow the train
            prev_pos = old_pos  # Position that follower should move to
            
            for i, follower in enumerate(followers):
                if i == 0:  # First follower takes lead's old position
                    follower_old = temp_from[follower]
                    self.movement_stack[follower].append((follower_old, prev_pos, False))
                    Pre[follower].append(prev_pos)
                    temp_from[follower] = prev_pos
                    # print(f"  Follower {follower}: {follower_old} -> {prev_pos}")
                    prev_pos = follower_old  # For next follower
                else:  # Additional followers take previous follower's old position
                    follower_old = temp_from[follower] 
                    self.movement_stack[follower].append((follower_old, prev_pos, False))
                    Pre[follower].append(prev_pos)
                    temp_from[follower] = prev_pos
                    # print(f"  Follower {i+1} {follower}: {follower_old} -> {prev_pos}")
                    prev_pos = follower_old
            
            # print(f"  Positions now: {[temp_from[a] for a in agents]}")
        
        # CRITICAL FIX: For 2-agent exchanges, ensure both agents are positioned for exchange
        if len(agents) == 2:
            lead_agent, other_agent = agents[0], agents[1]
            
            # Ensure one agent is ON the vertex and other is adjacent
            if temp_from[lead_agent] != v and temp_from[other_agent] != v:
                # Neither is on vertex, move one there
                if path_agent == lead_agent:
                    # Move lead agent to vertex
                    old_pos = temp_from[lead_agent]
                    self.movement_stack[lead_agent].append((old_pos, v, False))
                    Pre[lead_agent].append(v)
                    temp_from[lead_agent] = v
                    # print(f"  FINAL: Lead agent {lead_agent}: {old_pos} -> {v}")
                else:
                    # Move other agent to vertex  
                    old_pos = temp_from[other_agent]
                    self.movement_stack[other_agent].append((old_pos, v, False))
                    Pre[other_agent].append(v)
                    temp_from[other_agent] = v
                    # print(f"  FINAL: Other agent {other_agent}: {old_pos} -> {v}")
        
        print(f"[MOVE GROUP] Final positions: {[temp_from[a] for a in agents]}")
        
        # Verify at least one agent is on the exchange vertex
        agents_on_vertex = [a for a in agents if temp_from[a] == v]
        if not agents_on_vertex:
            # print(f"[MOVE GROUP] WARNING: No agent positioned on exchange vertex {v}")
            return False
            
        return True

    def is_vertex_safe_during_period(self, vertex, agent, start_step, end_step, normal_simulation):
        """
        Check if vertex is safe (no other agent passes through) during the given period.
        """
        for step in range(start_step, min(end_step, len(normal_simulation))):
            for other_agent in range(self.num_agents):
                if other_agent == agent:
                    continue
                    
                # Check if other agent passes through this vertex
                if (step < len(normal_simulation) and 
                    normal_simulation[step][other_agent] == vertex):
                    return False
                    
                # Check restore paths of other agents
                if (other_agent in self.restore_paths and 
                    step < len(self.restore_paths[other_agent]) and
                    self.restore_paths[other_agent][step] == vertex):
                    return False
                    
        return True

    def simulate_agent_only_pibt(self, agent_id: int, start_config: Config, detected_swaps: List, max_steps: int = 5) -> tuple[list[Config], list[tuple[int, int]]]:
        """
        Simulate PIBT for a specific agent only to detect additional swaps needed.
        More efficient than full simulation.
        """
        print(f"[SIMULATE_AGENT] Simulating PIBT for agent {agent_id} only")
        
        # Disable swaps temporarily
        original_try_swap = self.try_swap
        self.try_swap = lambda *args: False

        # create swap config
        simulation_sequence = [start_config.copy()]
        current_config = start_config.copy()
        # detected_swaps = []
        
        for _ in range(max_steps):
            
            # Get desired move for this agent
            candidates = [current_config[agent_id]] + get_neighbors(self.grid, current_config[agent_id])
            self.rng.shuffle(candidates)
            candidates = sorted(candidates, key=lambda u: self.dist_tables[agent_id].get(u))
            
            desired_pos = candidates[0]  # Best position for this agent
            
            # Check if another agent blocks this position
            blocking_agent = self.get_agent_at_position(current_config, desired_pos)
                
            if blocking_agent != self.NIL and blocking_agent != agent_id:

                # Check priority - don't swap with higher priority agents
                agent_priority = self.dist_tables[agent_id].get(current_config[agent_id])
                blocking_priority = self.dist_tables[blocking_agent].get(current_config[blocking_agent])
                
                if blocking_priority < agent_priority:  # Higher priority (lower distance)
                    print(f"[SIMULATE_AGENT] Agent {blocking_agent} has higher priority, no swap")
                    break

                detected_swaps.append((agent_id, blocking_agent))
                print(f"[SIMULATE_AGENT] Detected swap needed: {agent_id} <-> {blocking_agent}")

                '''swap detected! find the next swap if theres one'''
                # Create swap config
                swapped_config = current_config.copy()
                swapped_config[agent_id], swapped_config[blocking_agent] = swapped_config[blocking_agent], swapped_config[agent_id]

                # Find next swap
                simulation_sequence, detected_swaps = self.simulate_agent_only_pibt(agent_id, swapped_config, detected_swaps, max_steps=5)

                break

            simulation_sequence.append(current_config.copy())

            
            # Stop if agent reached goal
            if current_config[agent_id] == self.goals[agent_id]:
                break
        
        # Restore original method
        self.try_swap = original_try_swap
        # print("detected_swpa:", detected_swaps, "\n\n")
        return simulation_sequence, detected_swaps

    def try_swap(self, i: int, j: int, i_from: Config, i_moveto: Config) -> bool:
        """
        Improved swap handling with centralized exchange coordination.
        """
        print(f"[Swap] {i} <-> {j} starts...\n")

        # Detect complete swap chain
        print(f"----------------------\n[SIMULATE] Agent-specific lookahead for agent {i}")
        swapped_config = i_from.copy()
        swapped_config[i], swapped_config[j] = swapped_config[j], swapped_config[i]
        
        _, followup_swaps = self.simulate_agent_only_pibt(i, swapped_config, [], max_steps=3)
        
        # Build complete swap chain
        swap_chain = [(i, j)]
        for pair in followup_swaps:
            if pair not in swap_chain and (pair[1], pair[0]) not in swap_chain:
                swap_chain.append(pair)
        
        print(f"[SWAP_PLAN] Final swap chain: {swap_chain}\n----------------------------------")
        
        # Find suitable high-degree vertex
        candidates = self.find_high_degree_vertices(i_from[i], i_from)
        
        for v, path in candidates:
            if not v:
                continue
                
            print(f"[SWAP] Trying vertex {v} with path {path}")
            
            # Try to perform all exchanges at this vertex
            if self.perform_coordinated_exchanges(swap_chain, v, path, i_from, i_moveto):
                return True
        
        print("SWAP FAILED - No suitable vertex found")
        return False

    def perform_coordinated_exchanges(self, swap_chain: list, v: Coord, path: list, i_from: Config, i_moveto: Config) -> bool:
        """
        Perform all exchanges at the same high-degree vertex with proper coordination.
        """
        print("----------------------------------------------")
        print(f"[COORDINATED] Performing {len(swap_chain)} exchanges at vertex {v}")

        # Initialize stack-based tracking
        self.in_swap_operation = True
        self.movement_stack = {i: [] for i in range(self.num_agents)}
        
        # Initialize state tracking
        Pi = [i_from.copy()]
        Pre = {k: [] for k in range(self.num_agents)}
        current_state = i_from.copy()
        
        # Phase 1: Initial setup - move all involved agents to the vertex area
        involved_agents = sorted({a for pair in swap_chain for a in pair})
        print(f"[COORDINATED] Involved agents: {involved_agents}")
        
        if not self.setup_exchange_area(Pre, involved_agents, v, path, current_state, i_moveto):
            print("[COORDINATED] Failed to setup exchange area")
            return False

        # Generate configurations for setup phase
        self.generate_config(Pi, Pre, current_state, i_moveto)
        current_state = Pi[-1].copy()
        print(f"[COORDINATED] After setup, {len(Pi)} configurations generated")

        if self.execute_get_out_strategy(Pi, involved_agents, current_state, i_moveto):
            return True

        
        '''
        # Phase 2: Perform all exchanges sequentially at the same vertex
        for swap_idx, (a, b) in enumerate(swap_chain):
            print(f"[COORDINATED] Exchange {swap_idx+1}/{len(swap_chain)}: {a} <-> {b}")

            # involved_agents = {agent for pair in swap_chain for agent in pair}
            involved_agents = {a, b}
            if swap_idx+1 < len(swap_chain):
                for pair in swap_chain[swap_idx+1:]:
                    involved_agents.add(pair[0])
                    involved_agents.add(pair[1])
            involved_agents = list(involved_agents)

            # FIXED: Pass swap_chain to keep all agents connected
            if not self.position_agents_for_exchange(Pre, a, b, v, current_state, i_moveto, path, involved_agents):
                print(f"[COORDINATED] Failed to position agents {a}, {b} for exchange")
                return False
            
            # Generate positioning configurations
            self.generate_config(Pi, Pre, current_state, i_moveto)
            current_state = Pi[-1].copy()
            
            # Perform the actual exchange
            if not self.exchange(Pi, a, b, v, current_state, i_moveto):
                print(f"[COORDINATED] Exchange {a} <-> {b} failed")
                return False
            
            current_state = Pi[-1].copy()
            print(f"[COORDINATED] Exchange {swap_idx+1} complete, total configs: {len(Pi)}")

            # for x in Pi[p-1:]:
            #     print(x)
            # p = len(Pi)
        '''
        
        # Phase 3: record final exchange position
        self.record_final_exchange_positions(swap_chain, current_state)

        # Update global state
        self.new_configs = Pi[1:]
        self.in_swap_operation = False
        
        return True

    # good!
    def setup_exchange_area(self, Pre: dict, involved_agents: list, v: Coord, path: list, current_state: Config, i_moveto: Config) -> bool:
        """
        Setup the exchange area by clearing paths and positioning agents.
        This is called once at the beginning.
        """
        print(f"[SETUP_AREA] Setting up exchange area at {v}")
        
        # Clear the path if needed
        if path and len(path) > 1:
            if not self.clear_path_to_vertex(Pre, path, current_state, i_moveto, involved_agents[0]):
                print("[SETUP_AREA] Failed to clear path")
                return False
        
        # Clear around the high-degree vertex (generous clearing)
        if not self.clear(Pre, involved_agents[0], involved_agents[1], v, current_state, i_moveto, path):
            print("[SETUP_AREA] Failed to clear around vertex")
            return False
        
        # Move involved agents to the vicinity of the vertex
        if not self.move_agents_to_high_vertex(Pre, involved_agents, v, current_state, i_moveto, path):
            print("[SETUP_AREA] Failed to move agents to exchange area")
            return False
        
        return True

    # good
    def position_agents_for_exchange(self, Pre: dict, a: int, b: int, v: Coord, current_state: Config, i_moveto, original_path: list, involved_agents) -> bool:
        """
        Ensure agents a and b are properly positioned for exchange at vertex v.
        """

        if current_state[a] == v or current_state[b] == v:
            return True
        
        print(f"[POSITION_EXCHANGE] Positioning agents {a}, {b} for exchange at {v}")

        if not self.clear(Pre, a, b, v, current_state, i_moveto, original_path):
            return False
        if not self.move_agents_to_high_vertex(Pre, involved_agents, v, current_state, i_moveto, original_path):
            return False
    
        return True

    # not using for now
    def setup_stack_based_restore_mode(self, Pi: list, swap_chain: list, original_positions: Config, normal_simulation=None):
        """
        Simple stack-based restore - Pi is list of configs, each config is list of positions.
        """
        print(f"[STACK_RESTORE] Setting up restore for {len(swap_chain)} swaps")
        print(f"[STACK_RESTORE] Pi has {len(Pi)} configurations")
        
        self.restore = True  
        self.restore_paths = {}
        
        # Print what we tracked in the stack for debugging
        print(f"[STACK_RESTORE] Movement stack contents:")
        for agent, moves in self.movement_stack.items():
            if moves:
                print(f"  Agent {agent}: {moves}")
        
        # Build restore paths from stack (reverse positioning moves only)
        for agent in range(self.num_agents):
            if agent in self.movement_stack and self.movement_stack[agent]:
                restore_path = []
                
                # Go through stack in REVERSE order to undo positioning moves
                for old_pos, new_pos, is_exchange_move in reversed(self.movement_stack[agent]):
                    if not is_exchange_move:  # Only undo positioning moves
                        restore_path.append(old_pos)
                
                if restore_path:
                    self.restore_paths[agent] = restore_path
                    print(f"[STACK_RESTORE] Agent {agent} will restore through: {restore_path}")
        
        # For swapped agents, make sure they end up at their swapped positions
        swapped_agents = {a for pair in swap_chain for a in pair}
        final_config = Pi[-1] if Pi else original_positions  # Pi[-1] is the last config
        
        for agent in swapped_agents:
            if agent in self.restore_paths:
                # Agent should end at their swapped position
                swapped_pos = final_config[agent]  # final_config[agent] gets agent's final position
                if self.restore_paths[agent]:
                    self.restore_paths[agent][-1] = swapped_pos
                    print(f"[STACK_RESTORE] Agent {agent} final target set to: {swapped_pos}")
        
        print(f"[STACK_RESTORE] Restore setup complete")
    
    def record_final_exchange_positions(self, swap_chain: list, final_state: Config):
        """
        Record where agents ended up after all exchanges.
        """
        self.exchange_positions = {}
        for a, b in swap_chain:
            self.exchange_positions[a] = final_state[a]
            self.exchange_positions[b] = final_state[b]
        
        print(f"[RECORD_FINAL] Final exchange positions: {self.exchange_positions}")

    '''post-processing -- smooth'''
    # remove consecutive duplicate configurations
    def remove_redundant_moves(self, configs: list) -> list:
        """
        Simple redundancy removal - just remove consecutive duplicate configurations.
        """
        if len(configs) <= 1:
            return configs
        
        print(f"[SMOOTH] Removing duplicates from {len(configs)} configurations")
        
        # Remove consecutive duplicates
        cleaned_configs = [configs[0]]  # Always keep first config
        
        for i in range(1, len(configs)):
            if configs[i] != cleaned_configs[-1]:  # Only add if different from last
                cleaned_configs.append(configs[i])
            # else:
            #     print(f"[SMOOTH] Removed duplicate at step {i}: {configs[i]}")
        
        print(f"[SMOOTH] Final configuration count: {len(cleaned_configs)}")
        return cleaned_configs

    '''PATTERN DETECTOR IMPLEMENTATION'''
    def swap_required_and_possible(self, i: int, target_vertex: Coord, i_from: Config) -> Optional[int]:
        """
        Pattern detector for swap requirement and possibility.
        Returns agent ID j if swap with agent i is required and possible, None otherwise.
        """
        # Check if there's an agent j at the target vertex
        j = None
        for agent_id, pos in enumerate(i_from):
            if pos == target_vertex:
                j = agent_id
                break
        
        if j is None or j == i:
            return None
            
        # Only consider swap if current vertex has degree <= 2
        if self.get_vertex_degree(i_from[i]) > 2:
            return None
            
        # print(f"[SWAP_DETECTOR] Checking swap requirement for agents {i} and {j}")
        
        # First emulation: Check if swap is required
        swap_required = self.emulate_swap_necessity(i, j, i_from)
        if not swap_required:
            # print(f"[SWAP_DETECTOR] Swap not required for {i} and {j}")
            return None
            
        # Second emulation: Check if swap is possible  
        swap_possible = self.emulate_swap_possibility(i, j, i_from)
        if not swap_possible:
            # print(f"[SWAP_DETECTOR] Swap not possible for {i} and {j}")
            return None
            
        print(f"[SWAP_DETECTOR] Swap required and possible: {i} <-> {j}")
        return j

    def emulate_swap_necessity(self, i: int, j: int, i_from: Config) -> bool:
        """
        First emulation: Check if swap is necessary.
        Move i to j's location while moving j away, ignoring other agents.
        """
        current_i = i_from[i]
        current_j = i_from[j]
        goal_i = self.goals[i]
        goal_j = self.goals[j]
        
        simulation_steps = 0
        max_steps = 10  # Prevent infinite loops
        
        while simulation_steps < max_steps:
            # Move i toward j's current position
            current_i = current_j
            
            # Move j to another vertex (not i's location)
            j_neighbors = [n for n in get_neighbors(self.grid, current_j) 
                          if n != current_i and is_valid_coord(self.grid, n)]
            
            if not j_neighbors:
                break
                
            # Move j toward its goal among available neighbors
            current_j = min(j_neighbors, key=lambda v: self.dist_tables[j].get(v))
            
            # Check stopping conditions
            # (i) Swap not required: j's location has degree > 2
            if self.get_vertex_degree(current_j) > 2:
                return False
                
            # (ii) Swap required: j's location has degree 1, or i reaches goal while j's nearest neighbor toward goal is i's goal
            if self.get_vertex_degree(current_j) == 1:
                return True
                
            if current_i == goal_i:
                # Check if j's nearest neighbor toward its goal is i's goal
                j_neighbors = [n for n in get_neighbors(self.grid, current_j) if is_valid_coord(self.grid, n)]
                if j_neighbors:
                    nearest_to_goal = min(j_neighbors, key=lambda v: self.dist_tables[j].get(v))
                    if nearest_to_goal == goal_i:
                        return True
                        
            simulation_steps += 1
            
        return False

    def emulate_swap_possibility(self, i: int, j: int, i_from: Config) -> bool:
        """
        Second emulation: Check if swap is possible.
        Move j to i's location while moving i away.
        """
        current_i = i_from[i]
        current_j = i_from[j]
        
        simulation_steps = 0
        max_steps = 10
        
        while simulation_steps < max_steps:
            # Move j toward i's current position
            current_j = current_i
            
            # Move i to another vertex
            i_neighbors = [n for n in get_neighbors(self.grid, current_i) 
                          if n != current_j and is_valid_coord(self.grid, n)]
            
            if not i_neighbors:
                break
                
            # Move i toward its goal among available neighbors
            current_i = min(i_neighbors, key=lambda v: self.dist_tables[i].get(v))
            
            # Check stopping conditions
            # (i) Swap possible: i's location has degree > 2
            if self.get_vertex_degree(current_i) > 2:
                return True
                
            # (ii) Swap impossible: i is on vertex with degree 1
            if self.get_vertex_degree(current_i) == 1:
                return False
                
            simulation_steps += 1
            
        return False

    def get_vertex_degree(self, v: Coord) -> int:
        """Get the degree of a vertex (number of valid neighbors)"""
        return len([n for n in get_neighbors(self.grid, v) if is_valid_coord(self.grid, n)])

    '''Corrirdor SWAP???????????'''
    def execute_get_out_strategy(self, Pi, swap_chain: list, i_from: Config, i_moveto: Config) -> bool:
        """
        Execute get-out-of-the-way strategy:
        1. First agent finds a high-degree vertex and moves there temporarily
        2. Let the chain of agents flow through
        3. First agent moves to the final target
        """
        if len(swap_chain) < 2:
            return False
            
        first_agent = swap_chain[0]
        final_target = i_from[swap_chain[-1]]  # Where first agent ultimately wants to go
        
        print(f"[GET_OUT] Agent {first_agent} getting out of way, target: {final_target}")
        
        # Find a nearby high-degree vertex for temporary parking
        parking_spot = self.find_waiting_spot(i_from[first_agent], i_from)
        
        if parking_spot is None:
            print("[GET_OUT] No suitable parking spot found")
            return False
        
        # Build the get-out sequence
        
        # Step 1: Move first agent to parking spot
        step1_config = i_from.copy()
        step1_config[first_agent] = parking_spot
        Pi.append(step1_config)
        
        # Step 2-N: Let chain agents flow through (each takes previous agent's position)
        current_config = step1_config.copy()
        for i in range(1, len(swap_chain)):
            agent = swap_chain[i]
            prev_agent = swap_chain[i-1]
            
            if i == 1:
                # First follower takes first agent's original position
                current_config[agent] = i_from[first_agent]
            else:
                # Each subsequent agent takes previous agent's original position
                prev_original_pos = i_from[swap_chain[i-1]]
                current_config[agent] = prev_original_pos
            
            Pi.append(current_config.copy())
        
        # Final step: First agent moves to final target
        current_config[first_agent] = final_target
        Pi.append(current_config.copy())
        
        # Update the global state
        # self.new_configs = configs[1:]  # Skip initial config
        
        # Set immediate moves for this step
        i_moveto[first_agent] = parking_spot
        
        print(f"[GET_OUT] Strategy executed, {len(Pi)} total configurations")
        return True

    def find_waiting_spot(self, transit_hub: Coord, i_from: Config) -> Optional[Coord]:
        """
        Find a neighbor of the transit hub where A1 can wait
        """
        for neighbor in get_neighbors(self.grid, transit_hub):
            if (is_valid_coord(self.grid, neighbor) and 
                neighbor not in i_from):  # Empty spot
                return neighbor
        return None




