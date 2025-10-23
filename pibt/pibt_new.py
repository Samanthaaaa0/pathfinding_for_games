from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np
import math
from .dist_table import DistTable
from .utils import Config, Configs, Coord, Grid, get_neighbors, is_valid_coord

class PIBT:
    def __init__(self, grid: Grid, starts: Config, goals: Config, seed: int = 0):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.num_agents = len(starts)
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        self.priorities = []

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

        self.exchange_positions = {}  # agent_id -> final position after all exchanges
        self.in_swap_operation = False

        # restore restore
        self.move_stack = [[] for _ in range(self.num_agents)]  # list of (path to v)
        self.swap_stack = [[] for _ in range(self.num_agents)]  # list of (path from to other place (for swap)

        # Parallel execution state - just simple arrays
        self.current_timestep = 0
        self.agent_states = ["pibt"] * self.num_agents  # "pibt", "swap", "clearing", "waiting"
        self.agent_plans = {}  # agent_id -> List[Coord]
        self.plan_steps = [0] * self.num_agents  # current step in plan for each agent
        self.wait_until = [0] * self.num_agents  # timestep when agent can move again
        self.active_swaps = []  # List[SwapGroup]

        self.max_root_wait = 5

        # - livelock detection -
        self.push_count = {}  # key = (pushed_agent, pushing_agent), value = count
        self.livelock_threshold = 20
        self.livelock_detected = False
        self.involved_agents = set()

        self.push_count_reset_interval = self.livelock_threshold * 3
        self.current_step_count = 0

    def funcPIBT(self, priorities, i_from: Config, i_moveto: Config, i: int = 0,root_agent: int = None) -> bool:
        """
        Recursive function to implement the PIBT algorithm.
        root_agent tracks the original high-priority agent that started the chain.
        """
        if i >= self.num_agents:
            return True
        
        if i != root_agent:
            if self.register_push(i, root_agent):  # j was pushed by i
                print(f"[LIVELOCK] Livelock detected during PIBT execution")
        
        if self.restore:
            return self.handle_restore_agent(i, i_from, i_moveto)

        '''HANDLE LIVELOCK'''
        if self.livelock_detected and i in self.involved_agents:
            print(f"[LIVELOCK] Agent {i} involved in livelock, triggering resolution")
            return self.handle_livelock_agent(i, i_from, i_moveto)

        # get candidate configurations
        candidate = [i_from[i]] + get_neighbors(self.grid, i_from[i])
        self.rng.shuffle(candidate)  # tie-breaking, randomize
        candidate = sorted(candidate, key=lambda u: self.dist_tables[i].get(u))

        potential_swap_candidates = set()
        # all_blocking_lower_priority = True
        # has_blocking = False

        for v in candidate:

            if i == root_agent and v == i_from[i] and i != self.goals[i]:
                # ensure wait count tracking
                if not hasattr(self, "root_wait_count"):
                    self.root_wait_count = {}
                if i not in self.root_wait_count:
                    self.root_wait_count[i] = 0

                if potential_swap_candidates:
                    # If already waited enough → don't allow staying in place
                    if self.root_wait_count[i] >= self.max_root_wait:
                        break
                    else:
                        # Allow bounded waiting
                        self.root_wait_count[i] += 1

                else:
                # If no swap candidates exist, allow staying as fallback
                # (root_wait_count still increments so it won’t stay forever)
                    if self.root_wait_count[i] >= self.max_root_wait:
                        break
                    else:
                        # Allow bounded waiting
                        self.root_wait_count[i] += 1

            j = self.occupied_now[v]

            # Check for vertex conflict - exclude nodes that are already requested by others
            if self.occupied_nxt[v] != self.NIL:
                continue

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
                and (not self.funcPIBT(priorities, i_from, i_moveto, j, root_agent))  # Pass root_agent down
            ):

                # has_blocking = True

                # save as potential swap candidate
                if i == root_agent:
                    potential_swap_candidates.add((j, v))
                continue 

                # if i == root_agent:
                #     # print(f"[DEBUG] Root agent {i} checking deadlock")
                #     print(f"  - potential_swap_candidates: {potential_swap_candidates}")
                #     # print(f"  - all_blocking_lower_priority: {all_blocking_lower_priority}")
                #     potential_swap_candidates.add((j, v))
                #     if priorities[j] > priorities[i]:  # j has HIGHER priority (processed earlier)
                #         print(f"[PIBT] Root agent {i} blocked by higher-priority agent {j} at vertex {v}")
                #         all_blocking_lower_priority = False
                #     else:
                #         # j has equal or lower priority - potential deadlock candidate
                #         # potential_swap_candidates.add((j, v))
                #         continue    # GO SWAP LA

                # Success! Found a valid move
            return True

        # if i == root_agent and has_blocking and potential_swap_candidates and all_blocking_lower_priority:
        if i == root_agent and potential_swap_candidates:
            print(f"[DEADLOCK] Agent {i} deadlocked - all blocking agents have equal/lower priority")
            # Try swaps in order of preference (closest to goal first)
            sorted_candidates = sorted(
                potential_swap_candidates,
                key=lambda x: self.dist_tables[i].get(x[1])
            )

            for j, v in sorted_candidates:
                print(f"[PIBT] Root agent {i} trying swap with {j} at vertex {v}")
                if self.try_swap(i, j, i_from, i_moveto):
                    print(f"[PIBT] Swap success: A{i} <-> A{j}")
                    return True
                else:
                    print(f"[PIBT] Swap failed: A{i} <-> A{j}")

        # failed to secure node
        i_moveto[i] = i_from[i]
        self.occupied_nxt[i_from[i]] = i

        return False

    def step(self, i_from: Config, priorities: list[float]) -> Config:

        '''LIVELOCK'''
        self.current_step_count += 1
        if self.current_step_count % self.push_count_reset_interval == 0:
            self.reset_push_counts()
            self.livelock_detected = False
            self.involved_agents.clear()

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
                self.funcPIBT(priorities, i_from, i_moveto, i, i)

        # cleanup
        for i in range(N):
            if i_moveto[i] == self.NIL_COORD:
                i_moveto[i] = i_from[i]
                if self.occupied_nxt[i_from[i]] == self.NIL:
                    self.occupied_nxt[i_from[i]] = i

        self.occupied_now.fill(self.NIL)
        self.occupied_nxt.fill(self.NIL)

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

        self.priorities = priorities

        # print("-- "*20)
        # print("Priorities:", priorities)
        # print("-- "*20, end="\n")

        # main loop, generate sequence of configurations
        configs = [self.starts.copy()]
        print("Step 0:", configs[0])

        while len(configs) <= max_timestep:
            # obtain new configuration
            current_config = configs[-1].copy()
            Q = self.step(current_config, self.priorities)
            
            if self.new_configs != []:
                configs += self.new_configs.copy()
                self.new_configs = []
            else:
                configs.append(Q)

            Q = configs[-1]
            print(f"Step {len(configs) - 1}: {Q}")

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

        if configs[-1] != self.goals:
            configs.append(self.goals)

        return configs
    
    def clear_vertex(self, Pre, v: Coord, current_config: Config, U: set[Coord], i_moveto: Config, reserved_positions) -> bool:
        # Find which agent is at vertex v
        agent_at_v = self.get_agent_at_position(current_config, v)

        if agent_at_v == self.NIL:
            return True  # Vertex is already clear

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

                # self.movement_stack[agent_at_v].append((current_config[agent_at_v], neighbor, False))
                self.swap_stack[agent_at_v].append(neighbor)

                # Reserve the neighbor to prevent collisions
                reserved_positions.add(neighbor)
                # Found empty neighbor, move agent there
                i_moveto[agent_at_v] = neighbor
                Pre[agent_at_v].append(neighbor)
                current_config[agent_at_v] = neighbor 

                print(f"[CLEAR_VERTEX] Moved agent {agent_at_v} from {v} to {neighbor}")
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
        print(f"..[CLEAR] clearing for {r0} and {s0}")
        current_config = i_from.copy()

        # Check if the high-degree vertex v is occupied
        agent_at_v = self.get_agent_at_position(current_config, v)

        if agent_at_v != self.NIL and agent_at_v not in {r0, s0}:
            print(f"[CLEAR] High-degree vertex {v} is occupied by agent {agent_at_v}")

            # First, try to move the agent at v to a safe location
            v_neighbors = get_neighbors(self.grid, v)
            for safe_spot in v_neighbors:
                if (safe_spot not in current_config and 
                    safe_spot not in reserved_positions):

                    print(f"[CLEAR] Moving agent {agent_at_v} from {v} to {safe_spot}")
                    Pre[agent_at_v].append(safe_spot)
                    current_config[agent_at_v] = safe_spot
                    reserved_positions.add(safe_spot)

                    self.swap_stack[agent_at_v].append(safe_spot)
                    break

            else:
                # If no immediate neighbor is free, try to clear one
                for safe_spot in v_neighbors:
                    U = {v}  # Avoid moving back to v
                    if self.clear_vertex(Pre, safe_spot, current_config, U, i_moveto, reserved_positions):
                        print(f"[CLEAR] Cleared {safe_spot} and moving agent {agent_at_v} there")

                        self.swap_stack[agent_at_v].append(safe_spot)
                        Pre[agent_at_v].append(safe_spot)
                        current_config[agent_at_v] = safe_spot
                        reserved_positions.add(safe_spot)
                        break

                else:
                    print(f"[CLEAR_STACK] Failed to relocate agent {agent_at_v} from vertex {v}")
                    return False

        # assuming r is the agent at v (path[-1]), and s is the other agent beside v (path[-2])
        r, s = r0, s0
        v0 = path[-2] if len(path) > 1 else current_config[s]

        neighbors = get_neighbors(self.grid, v)
        occupied_positions = set(current_config)
        E = [n for n in neighbors if n not in occupied_positions and n != v0]

        print(f"[CLEAR] Clearing neighbors of {v}, initially empty: {E}")

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
                print(f"[CLEAR] Successfully cleared neighbor {n}")

                E.append(n)
                if len(E) >= 3:
                    return True
                

        if len(E) == 0:
            print("[CLEAR] No empty neighbors available")
            return False
        
        print("[CLEAR] Stage 1 FAILED ...............................")

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
                print(f"[CLEAR] Successfully cleared vertex {n} and {ε}")
                return True
            
        print("[CLEAR] Stage 2 FAILED ...............................")

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
            print("A0:", A0, "r:", r, "s:", s, "v:", v, "ε:", ε, "n:", n)

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
                        # self.movement_stack[i].append((i_from[i], k, False))
                        self.swap_stack[i].append(k)
            return True
        
        print("[CLEAR] Stage 3 FAILED ...............................")

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
        if n is None:
            return False
        
        occupied_positions = set(current_config)

        if n in occupied_positions:
            t = self.get_agent_at_position(current_config, n)

        # Move t -> v -> ε
        A1 = current_config.copy()
        reserved_positions.add(v)
        A1[t] = v

        A2 = A1.copy()
        reserved_positions.add(ε)
        A2[t] = ε

        Pre[t].append(v)
        Pre[t].append(ε)
        # self.movement_stack[t].append((i_from[t], v, False))
        # self.movement_stack[t].append((v, ε, False))

        self.swap_stack[t].append(v)
        self.swap_stack[t].append(ε)

        print(f"[CLEAR] Moved agent {t} from {n} to {v} and then to {ε}")
        
        return self.clear_vertex(Pre, ε, A2, {v, v0, n}, i_moveto, reserved_positions)

    def get_agent_at_position(self, config: Config, pos: Coord) -> int:
        """Helper function to find which agent is at a given position"""
        try:
            return config.index(pos)
        except ValueError:
            return self.NIL

    def handle_restore_agent(self, i: int, i_from: Config, i_moveto: Config) -> bool:
        """
        Handle agent movement during restore mode.
        """
        if i in self.restore_paths and self.restore_paths[i]:
            target = self.restore_paths[i][0]
            
            # If already at target, just consume it and stay
            if i_from[i] == target:
                i_moveto[i] = i_from[i]
                self.occupied_nxt[i_from[i]] = i
                self.restore_paths[i].pop(0)
                print(f"[RESTORE] Agent {i} already at {target}, continuing")
                return True
            
            # Try to move to target
            if self.occupied_nxt[target] == self.NIL:
                j = self.occupied_now[target]
                
                if j == self.NIL or i_moveto[j] != i_from[i]:
                    i_moveto[i] = target
                    self.occupied_nxt[target] = i
                    self.restore_paths[i].pop(0)
                    print(f"[RESTORE] Agent {i} moved to {target}, remaining: {self.restore_paths[i]}")
                    return True 
        
        # Fallback to PIBT
        self.restore_paths[i] = []
        print(f"[RESTORE] Agent {i} cannot follow restore path, using PIBT logic")
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
                    # if self.movement_stack[agent_at_pos] and self.movement_stack[agent_at_pos][-1][1] != current_config[agent_at_pos]:
                    if self.swap_stack[agent_at_pos] and self.swap_stack[agent_at_pos][-1] != current_config[agent_at_pos]:
                        print("WRONG STACK! - clear_path_to_vertex")
                        print(f"last pos: {self.swap_stack[agent_at_pos][-1]}, current pos: {current_config[agent_at_pos]}")
                    else:
                        # self.movement_stack[agent_at_pos].append((current_config[agent_at_pos], neighbor, False))
                        self.swap_stack[agent_at_pos].append(neighbor)

                    Pre[agent_at_pos].append(safe_pos)
                    current_config[agent_at_pos] = safe_pos
                    print(f"[CLEAR_PATH] Moved agent {agent_at_pos} from {pos} to {safe_pos}")
                else:
                    print(f"[CLEAR_PATH] Could not find safe position for agent {agent_at_pos} at {pos}")
                    return False
        
        return True

    def move_agents_to_high_vertex(self, Pre: dict, agents: list[int], v: Coord, temp_from: Config, i_moveto: Config, path: list):
        """
        Move agents to high-degree vertex - FIXED for 2-agent case
        """
        print(f"[MOVE GROUP] Moving agents {agents} to high-degree vertex {v}")
        # print(f"[MOVE GROUP] Path: {path}")

        # print("teeth:", self.movement_stack[0])

        for agent in agents:
            self.move_stack[agent].append(temp_from[agent])
        
        if not path:
            print("no way bro")
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
            # print("[MOVE GROUP] No agent on path, moving first agent to path start")
            path_agent = agents[0]
            # if self.movement_stack[path_agent] and self.movement_stack[path_agent][-1][1] != temp_from[path_agent]:
            #     print("WRONG STACK! - clear_path_to_vertex")
            #     print(f"last pos: {self.movement_stack[path_agent][-1][1]}, current pos: {temp_from[path_agent]}")
            # else:
            # self.movement_stack[path_agent].append((temp_from[path_agent], path[0], False))
            self.move_stack[path_agent].append(path[0])

            Pre[path_agent].append(path[0])
            temp_from[path_agent] = path[0]
            start_idx = 0
        
        # print(f"[MOVE GROUP] Lead agent {path_agent} starting at path index {start_idx}")
        
        # ensure the lead agent reaches the target vertex v
        target_idx = path.index(v) if v in path else len(path) - 1
        remaining_path = path[start_idx + 1:target_idx + 1]  # Include target vertex
        
        # Get other agents that need to follow
        followers = [a for a in agents if a != path_agent]
        
        for _, next_pos in enumerate(remaining_path):
            # print(f"[MOVE GROUP] Step {step + 1}:")
            
            # Lead agent moves forward
            old_pos = temp_from[path_agent]
            # self.movement_stack[path_agent].append((old_pos, next_pos, False))
            self.move_stack[path_agent].append(next_pos)
            Pre[path_agent].append(next_pos)
            temp_from[path_agent] = next_pos
            print(f"  Lead agent {path_agent}: {old_pos} -> {next_pos}")
            
            # Move followers in sequence, ensuring they follow the train
            prev_pos = old_pos  # Position that follower should move to
            
            for i, follower in enumerate(followers):
                if i == 0:  # First follower takes lead's old position
                    follower_old = temp_from[follower]
                    # self.movement_stack[follower].append((follower_old, prev_pos, False))
                    self.move_stack[follower].append(prev_pos)

                    Pre[follower].append(prev_pos)
                    temp_from[follower] = prev_pos
                    print(f"  Follower {follower}: {follower_old} -> {prev_pos}")
                    prev_pos = follower_old  # For next follower
                else:  # Additional followers take previous follower's old position
                    follower_old = temp_from[follower] 
                    # self.movement_stack[follower].append((follower_old, prev_pos, False))
                    self.move_stack[follower].append(prev_pos)

                    Pre[follower].append(prev_pos)
                    temp_from[follower] = prev_pos
                    print(f"  Follower {i+1} {follower}: {follower_old} -> {prev_pos}")
                    prev_pos = follower_old
            
            # print(f"  Positions now: {[temp_from[a] for a in agents]}")
        
        # For 2-agent exchanges, ensure both agents are positioned for exchange
        if len(agents) == 2:
            lead_agent, other_agent = agents[0], agents[1]
            
            # Ensure one agent is ON the vertex and other is adjacent
            if temp_from[lead_agent] != v and temp_from[other_agent] != v:
                # Neither is on vertex, move one there
                if path_agent == lead_agent:
                    # Move lead agent to vertex
                    old_pos = temp_from[lead_agent]
                    # self.movement_stack[lead_agent].append((old_pos, v, False))
                    self.move_stack[lead_agent].append(v)

                    Pre[lead_agent].append(v)
                    temp_from[lead_agent] = v
                    print(f"  FINAL: Lead agent {lead_agent}: {old_pos} -> {v}")
                else:
                    # Move other agent to vertex  
                    old_pos = temp_from[other_agent]
                    # self.movement_stack[other_agent].append((old_pos, v, False))
                    self.move_stack[other_agent].append(v)

                    Pre[other_agent].append(v)
                    temp_from[other_agent] = v
                    print(f"  FINAL: Other agent {other_agent}: {old_pos} -> {v}")
        
        print(f"[MOVE GROUP] Final positions: {[temp_from[a] for a in agents]}")
        
        # Verify at least one agent is on the exchange vertex
        agents_on_vertex = [a for a in agents if temp_from[a] == v]
        if not agents_on_vertex:
            print(f"[MOVE GROUP] WARNING: No agent positioned on exchange vertex {v}")
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
                # agent_priority = self.dist_tables[agent_id].get(current_config[agent_id])
                # blocking_priority = self.dist_tables[blocking_agent].get(current_config[blocking_agent])

                agent_priority = self.priorities[agent_id]
                blocking_priority = self.priorities[blocking_agent]
                
                if blocking_priority > agent_priority and desired_pos != self.goals[blocking_agent]:  # Higher priority (lower distance)
                    print(f"[SIMULATE_AGENT] Agent {blocking_agent} has higher priority and not on its goal {self.goals[blocking_agent]}, no swap")
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

        sc = [j]
        
        # Build complete swap chain
        swap_chain = [(i, j)]
        for pair in followup_swaps:
            if pair not in swap_chain and (pair[1], pair[0]) not in swap_chain:
                swap_chain.append(pair)
                sc.append(pair[1])
        
        # print(f"[SWAP_PLAN] Final swap chain: {swap_chain}\n----------------------------------")
        
        # Find suitable high-degree vertex
        candidates = self.find_high_degree_vertices(i_from[i], i_from)
        
        for v, path in candidates:
            if not v:
                continue
                
            print(f"[SWAP] Trying vertex {v} with path {path}")
            
            # Try to perform all exchanges at this vertex
            # if self.perform_coordinated_exchanges(swap_chain, v, path, i_from, i_moveto):
            #     return True

            print()
            print("*-"*20)
            print("Swapping agents:", sc)
            print("*-"*20)
            print()
            if self.corridor_swap(i, sc, v, path, i_from, i_moveto):
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
        # self.movement_stack = {i: [] for i in range(self.num_agents)}
        
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

            # Pass swap_chain to keep all agents connected
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
        
        # Phase 3: record final exchange position
        self.record_final_exchange_positions(swap_chain, current_state)

        # Update global state
        self.new_configs = Pi[1:]
        self.in_swap_operation = False
        
        return True

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
        
        # print(f"[SMOOTH] Removing duplicates from {len(configs)} configurations")
        
        # Remove consecutive duplicates
        cleaned_configs = [configs[0]]  # Always keep first config
        
        for i in range(1, len(configs)):
            if configs[i] != cleaned_configs[-1]:  # Only add if different from last
                cleaned_configs.append(configs[i])
            # else:
        #         print(f"[SMOOTH] Removed duplicate at step {i}: {configs[i]}")
        
        print(f"[SMOOTH] Final configuration count: {len(cleaned_configs)}")
        return cleaned_configs
    
    '''
    Corridor Swap
    
    move to high degree vertex,
    swapping agent -> high deg -> its neightbour, wait
    remaining agents -> hgih deg -> to other neigbours
    swapping agent -> towards goal
    '''
    def corridor_swap(self, i, swap_chain: list, v: Coord, path: list, i_from: Config, i_moveto: Config) -> bool:
        """
        Corridor swap with simple restore.
        """
        print("-"*50)

        self.in_swap_operation = True

        Pi = [i_from.copy()]
        Pre = {k: [] for k in range(self.num_agents)}
        current_state = i_from.copy()
        
        involved_agents = [i] + swap_chain
        print(f"[CORRIDOR] Involved agents: {involved_agents}")
        
        # Setup
        if not self.setup_exchange_area(Pre, involved_agents, v, path, current_state, i_moveto):
            return False
        
        self.generate_config(Pi, Pre, current_state, i_moveto)
        current_state = Pi[-1].copy()
        Pre = {k: [] for k in range(self.num_agents)}  # Clear
        
        # Find wait spot
        wait_spot = None
        for nbr in get_neighbors(self.grid, v):
            if nbr not in current_state:
                wait_spot = nbr
                break
        
        if not wait_spot:
            return False
        
        # Move i aside
        Pre[i].append(wait_spot)

        self.generate_config(Pi, Pre, current_state, i_moveto)
        current_state = Pi[-1].copy()
        Pre = {k: [] for k in range(self.num_agents)}
        
        print(f"[CORRIDOR] A{i} -> {wait_spot}")
        
        # Each swapping agent passes through
        for agent in swap_chain:

            prev_state = current_state.copy()

            print(f"\n\t[CORRIDOR SWAP] A{agent}'s turn...\n")

            if not self.move_agents_to_high_vertex(Pre, [agent], v, current_state, i_moveto, path):
                print("[CORRIDOR SWAP] MOVE FAILED")
                return False

            prev_pos = Pre[agent][-2] if len(Pre[agent])>1 else prev_state[agent]

            print(f"[CORRIDOR SWAP] A{agent}'s previous position: {prev_pos}")

            self.update_curr_state(Pre, current_state)

            # move to neighbour
            if not self.clear_vertex(Pre, v, current_state, {wait_spot, prev_pos}, i_moveto, set()):

                self.update_curr_state(Pre, current_state)

                neighbors = [n for n in get_neighbors(self.grid, v) if n not in {wait_spot, prev_pos}]
                cleared = False

                for neighbor in neighbors:
                    # First, try to clear the neighbor's neighbors (cascade clearing)
                    neighbor_neighbors = get_neighbors(self.grid, neighbor)
                    for nn in neighbor_neighbors:
                        if nn != v:  # Don't push back to v
                            self.clear_vertex(Pre, nn, current_state, {v, neighbor}, i_moveto, set())
                            self.update_curr_state(Pre, current_state)

                    # clear the neighbor itself
                    if self.clear_vertex(Pre, neighbor, current_state, {v}, i_moveto, set()):
                        self.update_curr_state(Pre, current_state)

                        if self.clear_vertex(Pre, v, current_state, {wait_spot, prev_pos}, i_moveto, set()):
                            cleared = True
                            break
                
                if not cleared:
                    print(f"[CORRIDOR SWAP] Failed to clear vertex for A{agent}")
                    self.in_swap_operation = False
                    return False
                
            self.update_curr_state(Pre, current_state)

            # update wait?
            for aa in swap_chain:
                if aa != agent and Pre[aa] and current_state[aa] == prev_state[aa]:
                    self.swap_stack[aa].append(current_state[aa])

        self.generate_config(Pi, Pre, current_state, i_moveto)
        current_state = Pi[-1].copy()
        
        # Agent i returns to v
        Pre[i].append(v)
        self.generate_config(Pi, Pre, current_state, i_moveto)
        current_state = Pi[-1].copy()
        
        print(f"\n[CORRIDOR] A{i} back to {v}")

        # RESTORE TIME
        if not self.restore_func(i, involved_agents, prev_state, Pi, i_moveto):
            print("[CORRIDOR] RESTORE FAILED")
            return False

        self.new_configs = Pi[1:]
        self.in_swap_operation = False
            
        return True

    def find_path_to_vertex(self, start: Coord, target: Coord) -> list:
        """Simple BFS to find path from start to target."""
        if start == target:
            return [start]
        
        # if start just one step away from target, return direct path
        if target in get_neighbors(self.grid, start):
            return [start, target]
        
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            pos, path = queue.pop(0)
            
            for nbr in get_neighbors(self.grid, pos):
                if nbr == target:
                    return path + [target]
                
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))
        
        return []  # No path found

    def trim_redundant_restoration(self, rotated_stack: list, involved_agents: list, current_state: Config) -> list:
        """
        Simulate PIBT after restoration and remove redundant moves by comparing path lists.
        """
        # print("\n[TRIM] Analyzing restoration paths for redundancy...")
        
        # Calculate where agents will be AFTER full restoration
        post_restore_config = current_state.copy()
        for agent in involved_agents:
            if rotated_stack[agent]:
                # Get final position after all restoration moves
                post_restore_config[agent] = rotated_stack[agent][-1]
        
        # print(f"[TRIM] Post-restoration positions: {[post_restore_config[a] for a in involved_agents]}")
        
        # Simulate PIBT for a few steps
        orig_swap = self.in_swap_operation
        self.in_swap_operation = True
        self.new_configs = []  # Clear to prevent interference
        
        sim_configs = [post_restore_config.copy()]
        current = post_restore_config.copy()
        sim_priorities = self.priorities.copy()
        
        for _ in range(8):  # Simulate more steps
            next_config = self.step(current, sim_priorities)
            
            # Only add if different from last config
            if next_config != sim_configs[-1]:
                sim_configs.append(next_config)
                current = next_config
            
            for i in range(self.num_agents):
                if current[i] != self.goals[i]:
                    sim_priorities[i] += 1
        
        self.in_swap_operation = orig_swap
        self.new_configs = []  # Clear again
        
        # Convert to path lists and trim
        trimmed_stack = [[] for _ in range(self.num_agents)]
        
        for agent in involved_agents:
            if not rotated_stack[agent]:
                continue
            
            # Convert restore moves to path list (remove consecutive duplicates)
            restore_path = [current_state[agent]]
            for move in rotated_stack[agent]:  # Reverse to get forward order
                if move != restore_path[-1]:  # Skip duplicates
                    restore_path.append(move)
            # restore_path.append(rotated_stack[agent][-1]) # Final position after restoration
            
            # Convert PIBT sim to path list (already filtered above)
            pibt_path = [config[agent] for config in sim_configs]
            
            # print(f"\n[TRIM] A{agent} restore path: {restore_path}")
            # print(f"[TRIM] A{agent} PIBT path:    {pibt_path}")
            
            # Reverse PIBT and find overlap with restore path
            pibt_reversed = pibt_path[::-1]
            
            # Find common suffix
            overlap = 0
            for i in range(1, min(len(restore_path), len(pibt_reversed)) + 1):
                if restore_path[-i:] == pibt_reversed[:i]:
                    overlap = i-1
            
            if overlap > 0:
                # Trim the overlapping moves from rotated_stack
                trimmed_stack[agent] = rotated_stack[agent][:overlap]
                # print(f"[TRIM] A{agent}: Trimmed {overlap} redundant moves (overlap found)")
            else:
                trimmed_stack[agent] = rotated_stack[agent]
            
        print("\n[TRIM] Trimmed restoration paths:")
        for a in involved_agents:
            print(f"A{a}: {trimmed_stack[a]}")
        print()
        
        return trimmed_stack

    def restore_func(self, i: int, involved_agents: list, current_state: Config, Pi: list, i_moveto: Config) -> bool:
        print("\n[RESTORE] Moving to final swapped positions...")
        Pre = {k: [] for k in range(self.num_agents)}

        # Use list instead of dict
        rotated_stack = [[] for _ in range(self.num_agents)]

        for idx, agent in enumerate(involved_agents):
            if idx == 0:  # agent i
                rotated_stack[agent] = self.move_stack[involved_agents[-1]].copy()
            else:
                rotated_stack[agent] = self.move_stack[involved_agents[idx-1]].copy()

        # print()
        # for agent in involved_agents:
        #     print(f"A{agent}: {rotated_stack[agent]}")
        # print()

        # update actual stack
        print("[RESTORE] Merging rotated stacks with swap stacks...")

        for a in involved_agents:
            # print("SWAP STACK:", self.swap_stack[a])
            rotated_stack[a] = rotated_stack[a] + self.swap_stack[a]

        # print()
        # for agent in involved_agents:
        #     print(f"A{agent}: {rotated_stack[agent]}")
        # print()

        rotated_stack = {a: list(reversed(rotated_stack[a])) for a in involved_agents}
        rotated_stack = self.trim_redundant_restoration(rotated_stack, involved_agents, current_state)

        # Restore order: i first, then others reversed
        restore_order = [i] + [a for a in involved_agents[::-1] if a != i]
        print("restore order:", restore_order)

        for step in range(max(len(rotated_stack[a]) for a in restore_order)):
            # Pre = {k: [] for k in range(self.num_agents)}
            
            for agent in restore_order:
                if step < len(rotated_stack[agent]):
                    target = rotated_stack[agent][step]

                    # if target not in current_state or target == current_state[agent]:
                    Pre[agent].append(target)
                    current_state[agent] = target

                    # fr smallmap:
                    # if target not in current_state or target == current_state[agent]:
                    #     Pre[agent].append(target)
                    #     current_state[agent] = target

                     # TODO: 小蓝不走 气鼠了
                    # this condition is wrong and i wnat to change but idk how
                    # elif target in current_state and target != current_state[agent]: # conflict, wait a timestep and move to the target
                    # else:
                    #     # blocking = [a for a, pos in current_state if pos == target][0]
                    #     Pre[agent].append(current_state[agent])
                    #     # Pre[agent].append(target)
                    #     current_state[agent] = current_state[agent]

        self.generate_config(Pi, Pre, current_state, i_moveto)
        current_state = Pi[-1].copy()

        print(f"[RESTORE] Final positions: {current_state}")

        return Pi


    def update_curr_state(self, Pre, current_state):
        for a in range(self.num_agents):
            if a in Pre.keys() and Pre[a]:
                current_state[a] = Pre[a][-1]

    '''
    DETECT LIVELOCK:
    - Maintain an n * n matrix pushCount[i][j], where:
        - i = agent being pushed
        - j = agent doing the pushing (higher priority in PIBT)
    - Every time PIBT forces agent i to move because of agent j, increment pushCount[i][j] += 1.
    - If the count exceeds a threshold θ, declare a livelock.
    - You can also identify the set of agents involved by looking at the non-zero rows/cols around the cycle.
    '''
    def register_push(self, pushed_agent: int, pushing_agent: int) -> bool:
        """
        Register that pushing_agent caused pushed_agent to move.
        Returns True if livelock is detected.
        """
        key = (pushed_agent, pushing_agent)
        self.push_count[key] = self.push_count.get(key, 0) + 1

        if self.push_count[key] > self.livelock_threshold:
            # print('urgh')
            return self.detect_livelock()
        return False

    def detect_livelock(self) -> bool:
        """
        Detect livelock by analyzing push patterns.
        Returns True if livelock is detected and sets involved agents.
        """
        self.involved_agents = set()
        
        # Find all agents involved in excessive pushing
        for (pushed, pushing), count in self.push_count.items():
            if count > self.livelock_threshold:
                self.involved_agents.add(pushed)
                self.involved_agents.add(pushing)

        if self.involved_agents:
            self.livelock_detected = True
            print(f"[LIVELOCK] Detected livelock involving agents: {self.involved_agents}")

            return True

        return False

    def reset_push_counts(self):
        """Reset push counts to prevent false positives from old data."""
        self.push_count.clear()
        print("[LIVELOCK] Push counts reset")

    def handle_livelock_agent(self, i, i_from, i_moveto):
        sc = []
        for a in self.involved_agents:
            if a != i: sc.append(a)

        candidates = self.find_high_degree_vertices(i_from[i], i_from)
        
        for v, path in candidates:
            if not v:
                continue

            print(f"[SWAP] Trying vertex {v} with path {path}")
            
            if self.corridor_swap(i, sc, v, path, i_from, i_moveto):
                self.livelock_detected = False
                self.push_count.clear()
                return True
            
        print("BRO TOO SAD JUST DIE")
        return False
