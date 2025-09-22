from typing import Optional
import numpy as np

from .dist_table import DistTable
from .utils import Config, Configs, Coord, Grid, get_neighbors, is_valid_coord


class PIBT:
    def __init__(self, grid: Grid, starts: Config, goals: Config, seed: int = 0):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)

        # distance table
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        original_best = C[1] if len(C) > 1 else C[0]

        # Check if swap is required and possible ONLY for the best candidate
        swaping_agent = None
        if len(C) > 1:  # Only check if there are actual neighbors
            swaping_agent = self.swap_required_and_possible(i, original_best, Q_from)

        # Line 4: If swap is needed, reverse candidate order
        # This makes agent i try to move away from its goal first
        if swaping_agent is not None:
            C = list(reversed(C))


        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and swaping_agent != j
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
                
            ):
                continue

            # Line 7: Execute swap if conditions are met
            # Only when we successfully take the BEST vertex (from original order) 
            # and we detected a swap was needed
            if (swaping_agent is not None and 
                v == original_best and
                Q_to[swaping_agent] == self.NIL_COORD):  # Agent j not assigned yet

                # print(f"i={i} needs to swap with j={swaping_agent}")
                # This means we're moving away from goal, so pull j to our current position
                Q_to[swaping_agent] = Q_from[i]
                self.occupied_nxt[Q_from[i]] = swaping_agent

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(self, Q_from: Config, priorities: list[float]) -> Config:
        # setup
        N = len(Q_from)
        Q_to: Config = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to

    def run(self, max_timestep: int = 1000) -> Configs:
        # define priorities
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        # main loop, generate sequence of configurations
        configs = [self.starts]
        while len(configs) <= max_timestep:
            # obtain new configuration
            Q = self.step(configs[-1], priorities)
            configs.append(Q)

            # update priorities & goal check
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break  # goal

        return configs
    
    def swap_required_and_possible(self, i: int, target_vertex: Coord, Q_from: Config) -> Optional[int]:
        """
        Pattern detector for swap requirement and possibility.
        Returns agent ID j if swap with agent i is required and possible, None otherwise.
        """
        # Check if there's an agent j at the target vertex
        j = None

        #for pos in get_neighbors(self.grid, i_from[i]):
        if target_vertex in Q_from and self.occupied_now[target_vertex] != self.NIL:
            # j = Q_from.index(target_vertex)
            j = self.occupied_now[target_vertex]

        # assert i!=j, print(f"i={i},j={j},---")

        if j is None:
            return None
            
        # Only consider swap if current vertex has degree <= 2
        if self.get_vertex_degree(Q_from[i]) > 2:
            return None

        # First emulation: Check if swap is required
        if not self.emulate_swap_necessity(i, j, Q_from):
            # print(f"[SWAP_DETECTOR] Swap not required for {i} and {j}")
            return None

        # Second emulation: Check if swap is possible
        if not self.emulate_swap_possibility(i, j, Q_from):
            # print(f"[SWAP_DETECTOR] Swap not possible for {i} and {j}")
            return None
            
        # print(f"[SWAP_DETECTOR] Swap required and possible: {i} <-> {j}")
        return j

    def emulate_swap_necessity(self, i: int, j: int, Q_from: Config) -> bool:
        """
        First emulation: Check if swap is necessary.
        Move i to j's location while moving j away, ignoring other agents.
        """
        current_i = Q_from[i]
        current_j = Q_from[j]
        goal_i = self.goals[i]
        
        # Track states to detect cycles (minimal addition to prevent infinite loops)
        seen_states = set()
        
        while True:
            state = (current_i, current_j)
            if state in seen_states:
                # Cycle detected - paper doesn't specify this case, assume swap not required
                return False
            seen_states.add(state)
            # Move i toward j's current position
            current_i = current_j
            
            # Move j to another vertex (not i's location)
            j_neighbors = [n for n in get_neighbors(self.grid, current_j) 
                        if n != current_i]
            
            if not j_neighbors:
                # j has nowhere to go, treat as degree 1
                return True
                
            # Move j toward its goal among available neighbors
            current_j = min(j_neighbors, key=lambda v: self.dist_tables[j].get(v))
            
            # Check stopping conditions exactly as paper says:
            j_degree = self.get_vertex_degree(current_j)
            
            # (i) Swap not required: j's location has degree > 2
            if j_degree > 2:
                return False
                
            # (ii) Swap required: j's location has degree = 1
            if j_degree == 1:
                return True
                
            # (ii) Swap required: i reaches goal while j's nearest neighbor toward goal is goal_i
            if current_i == goal_i:
                j_all_neighbors = [n for n in get_neighbors(self.grid, current_j)]
                if j_all_neighbors:
                    nearest_to_j_goal = min(j_all_neighbors, key=lambda v: self.dist_tables[j].get(v))
                    if nearest_to_j_goal == goal_i:
                        return True

    def emulate_swap_possibility(self, i: int, j: int, Q_from: Config) -> bool:
        """
        Second emulation: Check if swap is possible.
        Move j to i's location while moving i away.
        """
        current_i = Q_from[i]
        current_j = Q_from[j]
        
        # Track states to detect cycles (minimal addition to prevent infinite loops)
        seen_states = set()
        
        while True:
            state = (current_i, current_j)
            if state in seen_states:
                # Cycle detected - paper doesn't specify this case, assume swap not possible
                return False
            seen_states.add(state)
            # Move j toward i's current position
            current_j = current_i
            
            # Move i to another vertex
            i_neighbors = [n for n in get_neighbors(self.grid, current_i) 
                        if n != current_j]
            
            if not i_neighbors:
                # i has nowhere to go, treat as degree 1
                return False
                
            # Move i toward its goal among available neighbors
            current_i = min(i_neighbors, key=lambda v: self.dist_tables[i].get(v))
            
            # Check stopping conditions exactly as paper says:
            i_degree = self.get_vertex_degree(current_i)
            
            # (i) Swap possible: i's location has degree > 2
            if i_degree > 2:
                return True
                
            # (ii) Swap impossible: i is on vertex with degree = 1
            if i_degree == 1:
                return False
    
    def get_vertex_degree(self, v: Coord) -> int:
        """Get the degree of a vertex (number of valid neighbors)"""
        return len([n for n in get_neighbors(self.grid, v) if is_valid_coord(self.grid, n)])