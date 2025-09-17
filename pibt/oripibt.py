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

        # Check if swap is required and possible ONLY for the best candidate
        swaping_agent = None
        if len(C) > 1:  # Only check if there are actual neighbors
            # pass
            swaping_agent = self.swap_required_and_possible(i, C[0], Q_from)

        # Line 4: If swap is needed, reverse candidate order
        # This makes agent i try to move away from its goal first
        if swaping_agent is not None:
            # print(C)
            print(f"i={i} needs to swap with j={swaping_agent}")
            C = list(reversed(C))
            # print(C)


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
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            # Line 7: Execute swap if conditions are met
            # Only when we successfully take the BEST vertex (from original order) 
            # and we detected a swap was needed
            if (swaping_agent is not None and 
                #j < len(i_from) and  # Valid agent index
                v == C[0] and  
                Q_to[swaping_agent] == self.NIL_COORD):  # Agent j not assigned yet
                print(f"i={i} needs to swap with j={swaping_agent}")
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
    
    def swap_required_and_possible(self, i: int, target_vertex: Coord, i_from: Config) -> Optional[int]:
        """
        Pattern detector for swap requirement and possibility.
        Returns agent ID j if swap with agent i is required and possible, None otherwise.
        """
        # Check if there's an agent j at the target vertex
        j = None

        #for pos in get_neighbors(self.grid, i_from[i]):
        if target_vertex in i_from:
            j = i_from.index(target_vertex)

        assert i!=j, f"i={i},j={j},---"

        if j is None:
            return None
            
        # Only consider swap if current vertex has degree <= 2
        if self.get_vertex_degree(i_from[i]) > 2:
            return None
            
        # print(f"[SWAP_DETECTOR] Checking swap requirement for agents {i} and {j}")

        # First emulation: Check if swap is required
        if not self.emulate_swap_necessity(i, j, i_from):
            # print(f"[SWAP_DETECTOR] Swap not required for {i} and {j}")
            return None

        # Second emulation: Check if swap is possible
        if not self.emulate_swap_possibility(i, j, i_from):
            # print(f"[SWAP_DETECTOR] Swap not possible for {i} and {j}")
            return None
            
        # print(f"[SWAP_DETECTOR] Swap required and possible: {i} <-> {j}")
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
        
        # simulation_steps = 0
        # max_steps = 10  # Prevent infinite loops

        while True:
            # Move i toward j's current position
            current_i = current_j
            
            # Move j to another vertex (not i's location)
            j_neighbors = [n for n in get_neighbors(self.grid, current_j) 
                          if n != current_i]
            
            if not j_neighbors:
                break
                
            # Move j toward its goal among available neighbors
            current_j = min(j_neighbors, key=lambda v: self.dist_tables[j].get(v))
            
            # Check stopping conditions
            # (i) Swap not required: j's location has degree > 2
            if self.get_vertex_degree(current_j) > 2:
                return False
                
            # (ii) Swap required: j's location has degree 1, or i reaches goal while j's nearest neighbor toward goal is i's goal
            #if self.get_vertex_degree(current_j) == 1:
                # return True
            
            if current_i == goal_i:
                # Check if j's nearest neighbor toward its goal is i's goal
                j_neighbors = [n for n in get_neighbors(self.grid, current_j)]
                #if j_neighbors:
                nearest_to_goal = min(j_neighbors, key=lambda v: self.dist_tables[j].get(v))
                if nearest_to_goal == goal_i:
                    return True
                        
            #simulation_steps += 1
            
        return False

    def emulate_swap_possibility(self, i: int, j: int, i_from: Config) -> bool:
        """
        Second emulation: Check if swap is possible.
        Move j to i's location while moving i away.
        """
        current_i = i_from[i]
        current_j = i_from[j]
        
        # simulation_steps = 0
        # max_steps = 10
        
        # while simulation_steps < max_steps:
        while True:
            # Move j toward i's current position
            current_j = current_i
            
            # Move i to another vertex
            i_neighbors = [n for n in get_neighbors(self.grid, current_i) 
                       if n != current_j and is_valid_coord(self.grid, n)]
            
            if not i_neighbors:
                return False
                
            # Move i toward its goal among available neighbors
            current_i = min(i_neighbors, key=lambda v: self.dist_tables[i].get(v))
            
            # Check stopping conditions
            # (i) Swap possible: i's location has degree > 2
            if self.get_vertex_degree(current_i) > 2:
                return True
                
            # (ii) Swap impossible: i is on vertex with degree 1
            if self.get_vertex_degree(current_i) == 1:
                return False
                
            # simulation_steps += 1

    def get_vertex_degree(self, v: Coord) -> int:
        """Get the degree of a vertex (number of valid neighbors)"""
        return len([n for n in get_neighbors(self.grid, v) if is_valid_coord(self.grid, n)])