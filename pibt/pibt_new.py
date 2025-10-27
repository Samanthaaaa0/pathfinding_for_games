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
        self.livelock_threshold = max(8, min(50, int(10 + 12 * math.log10(self.num_agents + 1))))
        self.livelock_detected = False
        self.involved_agents = set()

        self.before_frozen_priorities = {}
        self.frozen_priorities = {}  # agent_id -> frozen_priority_value
        self.push_count_reset_interval = self.livelock_threshold * 3
        self.current_step_count = 0

    def funcPIBT(self, priorities, i_from: Config, i_moveto: Config, i: int = 0,root_agent: int = None) -> tuple:
        """
        Recursive function to implement the PIBT algorithm.
        root_agent tracks the original high-priority agent that started the chain.
        return type: (bool, int) => (succ, blocking agent)
        """

        if i != root_agent:
            if self.register_push(i, root_agent):  # j was pushed by i
                print(f"[LIVELOCK] Livelock detected during PIBT execution")

        # get candidate configurations
        candidate = [i_from[i]] + get_neighbors(self.grid, i_from[i])
        self.rng.shuffle(candidate)  # tie-breaking, randomize
        candidate = sorted(candidate, key=lambda u: self.dist_tables[i].get(u))
        topi=self.priorities.index((max(self.priorities)))
        if i==topi:
            print(i,candidate,i_from[i])
            candidate=candidate[:candidate.index(i_from[i])]
            print(candidate)

        potential_swap_candidates = set()
        # all_blocking_lower_priority = True
        # has_blocking=False
        max_blocking_a=-1
        for v in candidate:            

            j = self.occupied_now[v]

            # Check for vertex conflict - exclude nodes that are already requested by others
            if self.occupied_nxt[v] != self.NIL:
                max_blocking_a=max(max_blocking_a,self.occupied_nxt[v])
                continue

            # Avoid swap conflict - EXCLUDE previous position it inherited from
            if j != self.NIL and i_moveto[j] == i_from[i]:
                continue

            # reserve next location
            i_moveto[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the avoid edge conflict condition)
            if (j != self.NIL and (i_moveto[j] == self.NIL_COORD)):
                succ,block_id=self.funcPIBT(priorities, i_from, i_moveto, j, root_agent)
                
                if not succ:
                    max_blocking_a=max(max_blocking_a,block_id)
                    #has_blocking = True

                    i_moveto[i] = self.NIL_COORD
                    self.occupied_nxt[v] = self.NIL
                    continue   

                    # if i == root_agent:
                    #     print(f"[DEBUG] Root agent {i} checking deadlock")
                        
                    #     # print(f"  - all_blocking_lower_priority: {all_blocking_lower_priority}")
                    #     # potential_swap_candidates.add((j, v))
                        
                    #     if priorities[j] > priorities[i]:  # j has HIGHER priority
                    #         print(f"[PIBT] Root agent {i} blocked by higher-priority agent {j} at vertex {v}")
                    #         all_blocking_lower_priority = False

                    #     else:
                    #         # j has equal or lower priority - potential deadlock candidate
                    #         potential_swap_candidates.add((j, v))
                    #         print(f"  - - - - - - - - - - - - - - - -  - - - - - - - -  - - - - - - - -   S: {potential_swap_candidates}")
                    #         # has_blocking=True

            #if (
            #    j != self.NIL
            #    and (i_moveto[j] == self.NIL_COORD)
            #    and (not self.funcPIBT(priorities, i_from, i_moveto, j, root_agent))  # Pass root_agent down
            #):

            # if i = root + v = in_place + has swap candidate -> perform swap
            #if i == root_agent and v == i_from[i] and potential_swap_candidates and has_blocking:
            #    print("?")
            #    break

            # Success! Found a valid move
            # print(f"A{i} moves to {v}")
            return (True, -1)

        # resolution handling
        #if i == root_agent and potential_swap_candidates and all_blocking_lower_priority:
        # if i==3:
        #     print(f"{i} got here")
        #     print(f"blocking_a: {max_blocking_a}")
        #     print(f"proi: blocking: {self.priorities[max_blocking_a]}, i: {self.priorities[i]}")
        if i == root_agent and self.priorities[max_blocking_a]<=self.priorities[i]:
            # print(f"i:{i} | j:{j}")
        # if i == root_agent and potential_swap_candidates:
            if self.livelock_detected:
                print(f"[LIVELOCK->DEADLOCK] Agent {i} converted to deadlock, triggering swap")
            else:
                print(f"[DEADLOCK] Agent {i} deadlocked")

            # Try swaps in order of preference (closest to goal first)
            # sorted_candidates = sorted(
            #     potential_swap_candidates,
            #     key=lambda x: self.dist_tables[i].get(x[1])
            # )

            v=candidate[0]
            j=self.occupied_now[v]
            print(f"try swap: j:{j},v:{v}")
            if self.try_swap(i, j, i_from, i_moveto):
                print(f"[PIBT] Swap success: A{i} <-> A{j}")
                return (True,-1)
            else:
                print(f"[PIBT] Swap failed: A{i} <-> A{j}")

            '''
            for j, v in candidate:
                print(f"[PIBT] Root agent {i} trying swap with {j} at vertex {v}")
                if self.try_swap(i, j, i_from, i_moveto):
                    print(f"[PIBT] Swap success: A{i} <-> A{j}")
                    return (True,-1)
                else:
                    print(f"[PIBT] Swap failed: A{i} <-> A{j}")
            '''
        

        # failed to secure node
        i_moveto[i] = i_from[i]
        self.occupied_nxt[i_from[i]] = i

        return (False, max(max_blocking_a,i))

    def step(self, i_from: Config, priorities: list[float]) -> Config:
        # check livelock
        if self.detect_livelock():
            # print('oh shi')
            # exit()
            self.handle_livelock(i_from)

        if self.frozen_priorities:
            print(f"[LIVELOCK] Frozen priorities active: {self.frozen_priorities}")
            self.check_livelock_resolved(i_from)

        temp_priorities = priorities.copy()
        # update priorities with frozen priorities
        for agent_id, frozen_priority in self.frozen_priorities.items():
            temp_priorities[agent_id] = frozen_priority
            print(f"[LIVELOCK] Agent {agent_id} has frozen priority {frozen_priority}")

        # livelock detection bookkeeping
        self.current_step_count += 1
        if self.current_step_count % self.push_count_reset_interval == 0:
            self.reset_push_counts()
            self.livelock_detected = False
            self.involved_agents.clear()
            self.frozen_priorities.clear()
            self.temp_priorities = []

        # setup
        N = len(i_from)
        i_moveto: Config = []

        if temp_priorities != priorities:
            print('- '*30)
            print(f"P changed:", temp_priorities)
            print('- '*30)

        for i, v in enumerate(i_from):
            i_moveto.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: temp_priorities[i], reverse=True)
        # print("A:",A)
        for i in A:
            if i_moveto[i] == self.NIL_COORD:
                self.funcPIBT(temp_priorities, i_from, i_moveto, i, i)
                # print(f"[STEP] Agent {i} moved to: {i_moveto[i]}")

        # cleanup
        for i in range(N):
            if i_moveto[i] == self.NIL_COORD:
                i_moveto[i] = i_from[i]
                if self.occupied_nxt[i_from[i]] == self.NIL:
                    self.occupied_nxt[i_from[i]] = i

        self.occupied_now.fill(self.NIL)
        self.occupied_nxt.fill(self.NIL)

        return i_moveto

    def run(self, max_timestep: int = 50) -> Configs:
        # identify priorities
        priorities: list[float] = []
        for i in range(self.num_agents):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        self.priorities = priorities

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
            flg_fin = True
            for i in range(self.num_agents):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    self.priorities[i] += 1
                else:
                    self.priorities[i] -= np.floor(self.priorities[i])
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
                    print(f"[CLEAR_PATH] Cannot clear position {pos} occupied by higher-priority agent {agent_at_pos}")
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

        for agent in agents:
            # self.move_stack[agent].append(temp_from[agent])
            if not self.move_stack[agent] or self.move_stack[agent][-1] != temp_from[agent]:
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
            path_agent = agents[0]
            self.move_stack[path_agent].append(path[0])

            Pre[path_agent].append(path[0])
            temp_from[path_agent] = path[0]
            start_idx = 0

        # ensure the lead agent reaches the target vertex v
        target_idx = path.index(v) if v in path else len(path) - 1
        remaining_path = path[start_idx + 1:target_idx + 1]  # Include target vertex

        # Get other agents that need to follow
        followers = [a for a in agents if a != path_agent]

        for _, next_pos in enumerate(remaining_path):

            # Lead agent moves forward
            old_pos = temp_from[path_agent]
            self.move_stack[path_agent].append(next_pos)
            Pre[path_agent].append(next_pos)
            temp_from[path_agent] = next_pos
            print(f"  Lead agent {path_agent}: {old_pos} -> {next_pos}")

            # Move followers in sequence, ensuring they follow the train
            prev_pos = old_pos  # Position that follower should move to

            for i, follower in enumerate(followers):
                if i == 0:  # First follower takes lead's old position
                    follower_old = temp_from[follower]
                    self.move_stack[follower].append(prev_pos)

                    Pre[follower].append(prev_pos)
                    temp_from[follower] = prev_pos
                    print(f"  Follower {follower}: {follower_old} -> {prev_pos}")
                    prev_pos = follower_old  # For next follower
                else:  # Additional followers take previous follower's old position
                    follower_old = temp_from[follower] 
                    self.move_stack[follower].append(prev_pos)

                    Pre[follower].append(prev_pos)
                    temp_from[follower] = prev_pos
                    print(f"  Follower {i+1} {follower}: {follower_old} -> {prev_pos}")
                    prev_pos = follower_old

        # For 2-agent exchanges, ensure both agents are positioned for exchange
        if len(agents) == 2:
            lead_agent, other_agent = agents[0], agents[1]
            
            # Ensure one agent is ON the vertex and other is adjacent
            if temp_from[lead_agent] != v and temp_from[other_agent] != v:
                # Neither is on vertex, move one there
                if path_agent == lead_agent:
                    # Move lead agent to vertex
                    old_pos = temp_from[lead_agent]
                    self.move_stack[lead_agent].append(v)

                    Pre[lead_agent].append(v)
                    temp_from[lead_agent] = v
                    print(f"  FINAL: Lead agent {lead_agent}: {old_pos} -> {v}")
                else:
                    # Move other agent to vertex  
                    old_pos = temp_from[other_agent]
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
                simulation_sequence, detected_swaps = self.simulate_agent_only_pibt(agent_id, swapped_config, detected_swaps, max_steps=max_steps-1)

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

        # Detect complete swap chain by simulating PIBT 
        print(f"----------------------\n[SIMULATE] Agent-specific lookahead for agent {i}")
        swapped_config = i_from.copy()
        swapped_config[i], swapped_config[j] = swapped_config[j], swapped_config[i]
        
        _, followup_swaps = self.simulate_agent_only_pibt(i, swapped_config, [], max_steps=5)

        sc = [j]
        
        # Build complete swap chain
        swap_chain = [(i, j)]
        for pair in followup_swaps:
            if pair not in swap_chain and (pair[1], pair[0]) not in swap_chain:
                swap_chain.append(pair)
                sc.append(pair[1])

        # Find suitable high-degree vertex
        candidates = self.find_high_degree_vertices(i_from[i], i_from)
        
        for v, path in candidates:
            if v is None:
                continue
                
            print(f"[SWAP] Trying vertex {v} with path {path}")

            print()
            print("*-"*20)
            print("Swapping agents:", sc)
            print("*-"*20)
            print()

            if self.corridor_swap(i, sc, v, path, i_from, i_moveto):
                return True
        
        print("SWAP FAILED - No suitable vertex found")
        return False

    def setup_exchange_area(self, Pre: dict, involved_agents: list, v: Coord, path: list, current_state: Config, i_moveto: Config) -> bool:
        """
        Setup the exchange area by clearing paths and positioning agents.
        This is called once at the beginning.
        """
        print(f"[SETUP_AREA] Setting up exchange area at {v}")

        closest_agent = min(involved_agents, key=lambda a: self.manhattan_distance(current_state[a], v))
        
        # Clear the path if needed
        if path and len(path) > 1:
            if not self.clear_path_to_vertex(Pre, path, current_state, i_moveto, involved_agents[0]):
                print("[SETUP_AREA] Failed to clear path")
                return False
        
        # Clear around the high-degree vertex
        if not self.clear(Pre, involved_agents[0], involved_agents[1], v, current_state, i_moveto, path):
            print("[SETUP_AREA] Failed to clear around vertex")
            return False

        # Move involved agents to the vicinity of the vertex
        if not self.move_agents_to_high_vertex(Pre, involved_agents, v, current_state, i_moveto, path):
            print("[SETUP_AREA] Failed to move agents to exchange area")
            return False
                
        return True
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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

        wait_spot = self.find_wait_spot(v, current_state, i)
        print(f"Wait Spot {i}: {wait_spot}")
        
        if not wait_spot:
            return False
        
        # Move i aside
        Pre[i].append(wait_spot)

        self.generate_config(Pi, Pre, current_state, i_moveto)
        current_state = Pi[-1].copy()
        Pre = {k: [] for k in range(self.num_agents)}
        
        print('- '*30)
        print(f"[CORRIDOR] A{i} -> {wait_spot}")
        print(f"Config Now: {current_state}")
        print('- '*30)
        
        # Each swapping agent passes through
        for agent in swap_chain:

            prev_state = current_state.copy()

            print(f"\n\t[CORRIDOR SWAP] A{agent}'s turn...\n")

            if not self.move_agents_to_high_vertex(Pre, swap_chain[swap_chain.index(agent):], v, current_state, i_moveto, path):
                print("[CORRIDOR SWAP] MOVE FAILED")
                return False
            
            # print()
            # print('+ - '*20)
            # print('Move stack:')
            # for o in involved_agents:
            #     print(f'{o} - {self.move_stack[o]}')
            # print('+ - '*20, end='\n\n')

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

        self.livelock_detected = False

        # RESTORE TIME
        if not self.restore_func(i, involved_agents, prev_state, Pi, i_moveto):
            print("[CORRIDOR] RESTORE FAILED")
            return False

        self.new_configs = Pi[1:]
        self.in_swap_operation = False
            
        return True
    
    def find_wait_spot(self, v: Coord, current_state: Config, i: int) -> Coord:
        """
        Find the wait spot with the least degree (fewest connections).
        Dead ends and corners are ideal as they don't block flow.
        """
        neighbors = get_neighbors(self.grid, v)
        available_spots = [nbr for nbr in neighbors if nbr not in current_state]
        
        if not available_spots:
            return None
        
        # Find the spot with the least degree
        best_spot = None
        min_degree = float('inf')
        
        for spot in available_spots:
            # Count the degree (number of valid neighbors) of this spot
            spot_neighbors = get_neighbors(self.grid, spot)
            degree = len(spot_neighbors)
            
            # Prefer spots with lower degree
            if degree < min_degree:
                min_degree = degree
                best_spot = spot
            elif degree == min_degree:
                # Tie-breaker: prefer spots farther from the goal if known
                if self.goals and i < len(self.goals):
                    curr_dist = self.dist_tables[i].get(best_spot)
                    new_dist = self.dist_tables[i].get(spot)
                    if new_dist > curr_dist:  # Farther from goal is better for waiting
                        best_spot = spot
        
        print(f"[CORRIDOR] Selected wait spot: {best_spot} (degree: {min_degree})")
        return best_spot

    def trim_redundant_restoration(self, rotated_stack: list, involved_agents: list, current_state: Config) -> list:
        """
        Simulate lightweight PIBT-like movement without actually calling step() to avoid recursion.
        """
        # Calculate where agents will be AFTER full restoration
        post_restore_config = current_state.copy()
        for agent in involved_agents:
            if rotated_stack[agent]:
                post_restore_config[agent] = rotated_stack[agent][-1]
        
        # Lightweight simulation - just greedy moves, no actual PIBT calls
        sim_configs = [post_restore_config.copy()]
        current = post_restore_config.copy()
        
        # Simulate simple greedy movement for a few steps
        for _ in range(8):
            next_config = current.copy()
            moved = False
            
            # Sort agents by their distance to goal (simulate priority)
            sorted_agents = sorted(involved_agents, 
                                key=lambda a: self.dist_tables[a].get(current[a]))
            
            for agent in sorted_agents:
                # Skip if already at goal
                if current[agent] == self.goals[agent]:
                    continue
                
                # Get neighbors and sort by distance to goal
                current_pos = current[agent]
                candidates = [current_pos] + get_neighbors(self.grid, current_pos)
                
                # Shuffle for randomness then sort by distance
                self.rng.shuffle(candidates)
                candidates = sorted(candidates, 
                                key=lambda pos: self.dist_tables[agent].get(pos))
                
                # Try to move to best available position
                for desired_pos in candidates:
                    # Check if position is free
                    position_free = True
                    for other_agent in range(self.num_agents):
                        if other_agent != agent and next_config[other_agent] == desired_pos:
                            position_free = False
                            break
                    
                    if position_free:
                        if desired_pos != current[agent]:  # Only if actually moving
                            next_config[agent] = desired_pos
                            moved = True
                        break  # Take the best available move
            
            # Only add config if something changed
            if moved:
                sim_configs.append(next_config.copy())
                current = next_config.copy()
            else:
                break  # No progress possible
        
        # Convert to path lists and trim
        trimmed_stack = [[] for _ in range(self.num_agents)]
        
        for agent in involved_agents:
            if not rotated_stack[agent]:
                continue
            
            # Build restore path
            restore_path = [current_state[agent]]
            for move in rotated_stack[agent]:
                if move != restore_path[-1]:
                    restore_path.append(move)
            
            # Build simulated path
            pibt_path = [config[agent] for config in sim_configs]
            
            # Find how many moves from end of restore match beginning of simulated path
            redundant_moves = 0
            max_check = min(len(restore_path) - 1, len(pibt_path))
            
            for k in range(1, max_check + 1):
                # Check if last k positions of restore match first k of simulation
                if restore_path[-k:] == pibt_path[:k]:
                    redundant_moves = k
            
            # Trim redundant moves from the end
            if redundant_moves > 0:
                moves_to_keep = len(rotated_stack[agent]) - redundant_moves
                if moves_to_keep > 0:
                    trimmed_stack[agent] = rotated_stack[agent][:moves_to_keep]
                else:
                    trimmed_stack[agent] = []  # All moves redundant
            else:
                trimmed_stack[agent] = rotated_stack[agent]
        
        print("\n[TRIM] Trimmed restoration paths:")
        for a in involved_agents:
            if trimmed_stack[a]:
                print(f"A{a}: {trimmed_stack[a]}")
        print()
        
        return trimmed_stack
    
    def restore_func(self, i: int, involved_agents: list, current_state: Config, Pi: list, i_moveto: Config) -> bool:
        print("\n[RESTORE] Moving to final swapped positions...")
        Pre = {k: [] for k in range(self.num_agents)}

        # for o in involved_agents:
        #     print()
        #     print(self.move_stack[o])
        #     self.move_stack[o] = self.trim_consecutive_duplicates(self.move_stack[o])
        #     print(self.move_stack[o])
        #     print()

        print()
        print('+ - '*20)
        print('Move stack:')
        for o in involved_agents:
            print(f'{o} - {self.move_stack[o]}')
        print('+ - '*20, end='\n\n')

        print('+ - '*20)
        print('Swap stack:')
        for o in involved_agents:
            print(f'{o} - {self.swap_stack[o]}')
        print('+ - '*20, end='\n\n')

        # Use list instead of dict
        rotated_stack = [[] for _ in range(self.num_agents)]

        for idx, agent in enumerate(involved_agents):
            if idx == 0:  # agent i
                rotated_stack[agent] = self.move_stack[involved_agents[-1]].copy()
            else:
                rotated_stack[agent] = self.move_stack[involved_agents[idx-1]].copy()

        # update actual stack
        print("[RESTORE] Merging rotated stacks with swap stacks...")

        for a in involved_agents:
            rotated_stack[a] = rotated_stack[a] + self.swap_stack[a]

        print()
        print('='*80)
        print('BEFORE ROTATED STACK:')
        for o in involved_agents:
            print(f'{o} - {rotated_stack[o]}')
        print('='*80,end='\n\n')

        rotated_stack = {a: list(reversed(rotated_stack[a])) for a in involved_agents}
        rotated_stack = self.trim_redundant_restoration(rotated_stack, involved_agents, current_state)

        # Restore order: i first, then others reversed
        restore_order = [i] + [a for a in involved_agents[::-1] if a != i]
        print("restore order:", restore_order)

        for step in range(max(len(rotated_stack[a]) for a in restore_order)):
            
            for agent in restore_order:
                if step < len(rotated_stack[agent]):
                    target = rotated_stack[agent][step]

                    # if target not in current_state or target == current_state[agent]:
                    Pre[agent].append( target)
                    current_state[agent] = target

        self.generate_config(Pi, Pre, current_state, i_moveto)
        current_state = Pi[-1].copy()

        print(f"[RESTORE] Final positions: {current_state}")

        # reset!
        self.move_stack = [[] for _ in range(self.num_agents)]
        self.swap_stack = [[] for _ in range(self.num_agents)]

        return Pi

    def update_curr_state(self, Pre, current_state):
        for a in range(self.num_agents):
            if a in Pre.keys() and Pre[a]:
                current_state[a] = Pre[a][-1]

    '''
    DETECT LIVELOCK:
    - Maintain an ushCount[i][j], where:
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

    def detect_livelock(self) -> bool:
        """
        Detect livelock by analyzing push patterns.
        Returns True if livelock is detected and sets involved agents.
        """
        # if any agent in push count exceeded threshold, gather all of them
        if any(count > self.livelock_threshold for count in self.push_count.values()):

            self.involved_agents = set()
            
            # Find all agents involved in excessive pushing
            for (pushed, pushing), count in self.push_count.items():
                if count > self.livelock_threshold:
                    self.involved_agents.add(pushed)
                    self.involved_agents.add(pushing)
            
                    # also consider agents that were pushed by these agents (only if they also exceed threshold)
                    for (p2, pusher2), count2 in self.push_count.items():
                        if pusher2 == pushed and count2 > self.livelock_threshold:
                            self.involved_agents.add(p2)

            if self.involved_agents:
                self.livelock_detected = True
                print(f"[LIVELOCK] Detected livelock involving agents: {self.involved_agents}")

                return True

        return False

    def reset_push_counts(self):
        """Reset push counts to prevent false positives from old data."""
        self.push_count.clear()
        print("[LIVELOCK] Push counts reset")

    def handle_livelock(self, i_from):
        """
        Handle livelock by freezing priorities and letting PIBT retry.
        Only use swap if PIBT still fails.
        """
        print(f"[LIVELOCK] Freezing priorities for agents: {self.involved_agents}")
        
        # Freeze priorities for involved agents that are on goals (have highest priority)
        for agent in self.involved_agents:
            if agent not in self.frozen_priorities and i_from[agent] == self.goals[agent]:
                self.frozen_priorities[agent] = math.inf # Highest priority
        
        # Let PIBT handle it naturally - just return False to trigger retry
        # PIBT will use frozen priorities and try backtracking
        return False

    def check_livelock_resolved(self, i_from):
        """
        Check if agents with frozen priorities have made progress.
        If yes, unfreeze them.
        """
        if not self.involved_agents:
            return
        
        # Check if any frozen agent reached their goal
        resolved = False
        for agent in list(self.involved_agents):
            if i_from[agent] != self.goals[agent]:
                resolved = False
                break
        
        if resolved:
            print(f"[LIVELOCK] Resolved! Unfreezing priorities.")
            for agent in self.frozen_priorities:
                print(f"  Agent {agent} wa/;ls at {i_from[agent]}, goal {self.goals[agent]}")

            self.frozen_priorities.clear()
            self.involved_agents.clear()
            self.livelock_detected = False
            self.push_count.clear()

