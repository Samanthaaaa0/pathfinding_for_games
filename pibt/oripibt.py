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

    def funcPIBT(self, i_from: Config, i_moveto: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        candidate = [i_from[i]] + get_neighbors(self.grid, i_from[i])
        self.rng.shuffle(candidate)  # tie-breaking, randomize
        candidate = sorted(candidate, key=lambda u: self.dist_tables[i].get(u))

        # Check if swap is required and possible ONLY for the best candidate
        j = None
        if len(candidate) > 1:  # Only check if there are actual neighbors
            j = self.swap_required_and_possible(i, candidate[0], i_from)
        
        # Line 4: If swap is needed, reverse candidate order
        # This makes agent i try to move away from its goal first
        if j is not None:
            candidate = sorted(candidate, key=lambda u: self.dist_tables[i].get(u), reverse=True)

        # vertex assignment
        for v in candidate:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            k = self.occupied_now[v]

            # avoid edge collision
            if k != self.NIL and i_moveto[k] == i_moveto[i]:
                continue

            # reserve next location
            i_moveto[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (k != i due to the second condition)
            if (
                k != self.NIL
                and (i_moveto[k] == self.NIL_COORD)
                and (not self.funcPIBT(i_from, i_moveto, k))
            ):
                # Failed to move agent k, so we can't take this vertex
                i_moveto[i] = self.NIL_COORD
                self.occupied_nxt[v] = self.NIL
                continue

            # Line 7: Execute swap if conditions are met
            # Only when we successfully take the BEST vertex (from original order) 
            # and we detected a swap was needed
            if (j is not None and 
                j < len(i_from) and  # Valid agent index
                v == candidate[-1] and  # We took the worst vertex (due to reversal)
                i_moveto[j] == self.NIL_COORD):  # Agent j not assigned yet
                
                # This means we're moving away from goal, so pull j to our current position
                i_moveto[j] = i_from[i]
                self.occupied_nxt[i_from[i]] = j

            return True

        # failed to secure node
        i_moveto[i] = i_from[i]
        self.occupied_nxt[i_from[i]] = i
        return False

    def step(self, i_from: Config, priorities: list[float]) -> Config:
        # setup
        N = len(i_from)
        Q_to: Config = []
        for i, v in enumerate(i_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(i_from, Q_to, i)

        # cleanup
        for q_from, q_to in zip(i_from, Q_to):
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
        Conservative pattern detector - only triggers when livelock is likely.
        Returns agent ID j if swap with agent i is required and possible, None otherwise.
        """
        # Check if there's an agent j at the target vertex
        j = None
        for agent_id, pos in enumerate(i_from):
            if pos == target_vertex and agent_id != i:
                j = agent_id
                break
        
        if j is None:
            return None
        
        # Only consider swap in very constrained situations
        # Both positions should be in narrow corridors (degree <= 2)
        if (self.get_vertex_degree(i_from[i]) > 2 or 
            self.get_vertex_degree(target_vertex) > 2):
            return None
        
        # Check if this looks like a potential livelock scenario
        # using simplified emulation that's more conservative
        if not self.is_livelock_scenario(i, j, i_from):
            return None
            
        return j

    def is_livelock_scenario(self, i: int, j: int, i_from: Config) -> bool:
        """
        Conservative check for livelock scenarios.
        Only returns True when swap is really necessary.
        """
        pos_i = i_from[i]
        pos_j = i_from[j] 
        goal_i = self.goals[i]
        goal_j = self.goals[j]
        
        # Quick check: if both agents want to go toward each other's positions
        # and are in a narrow corridor, this could be livelock
        
        # Check if i's goal is in the direction of j's position
        neighbors_i = get_neighbors(self.grid, pos_i)
        if pos_j not in neighbors_i:
            return False
            
        # Check if they're in a "facing" situation in a narrow corridor
        degree_i = self.get_vertex_degree(pos_i)
        degree_j = self.get_vertex_degree(pos_j)
        
        # Only trigger in very narrow situations
        if degree_i > 2 or degree_j > 2:
            return False
            
        # Check if moving normally would create a cycle
        # This is a simplified version - in practice, you might want more sophisticated detection
        dist_i_via_j = self.dist_tables[i].get(pos_j)
        dist_i_direct = self.dist_tables[i].get(pos_i)
        
        # Only swap if going through j's position is actually better for i
        if dist_i_via_j >= dist_i_direct:
            return False
            
        return True

    def get_vertex_degree(self, v: Coord) -> int:
        """Get the degree of a vertex (number of valid neighbors)"""
        return len([n for n in get_neighbors(self.grid, v) if is_valid_coord(self.grid, n)])