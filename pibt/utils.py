from collections import defaultdict
import os
import re
from typing import TypeAlias

# from .dist_table import DistTable

import numpy as np

Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]
Config: TypeAlias = list[Coord]
Configs: TypeAlias = list[Config]

@staticmethod
def parse_map(file_path):
    with open(file_path, 'r') as f:
        # retrieve map size
        lines = f.readlines()
        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])

        # retrieve map
        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in lines[4:]:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1
            # print("y:",y)

        assert y == height, f"map format seems strange, check {file_path}"

    return grid

def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    # coord: y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord

    if x > 0 and grid[y, x - 1]:
        neigh.append((y, x - 1))

    if x < grid.shape[1] - 1 and grid[y, x + 1]:
        neigh.append((y, x + 1))

    if y > 0 and grid[y - 1, x]:
        neigh.append((y - 1, x))

    if y < grid.shape[0] - 1 and grid[y + 1, x]:
        neigh.append((y + 1, x))

    return neigh

def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    y, x = coord
    if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1] or not grid[coord]:
        return False
    return True

# to parse a scenario file - read agents' start and goal positions | cited from
def get_scenario(scen_file: str, N: int | None = None) -> tuple[Config, Config]:
    with open(scen_file, "r") as f:
        starts, goals = [], []
        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s, y_s, x_g, y_g = [int(res.group(k)) for k in range(1, 5)]
                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))

                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break

    return starts, goals

def save_configs_for_visualizer(configs: Configs, filename: str) -> None:
    dirname = os.path.dirname(filename)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)

def validate_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity
            assert v_i_now in [v_i_pre] + get_neighbors(grid, v_i_pre), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"

def is_valid_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    try:
        validate_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False



'''
useless
'''
# def find_subgraphs(grid: Grid, starts: Config, goals: Config, m: int) -> set[Coord]:
#     # find all nontrivial biconnected components in the grid using Tarjan's algorithm
#     bccs = tarjan(grid, starts, goals)
#     print(f"found {len(bccs)} biconnected components")

#     S = [bcc.copy() for bcc in bccs if len(bcc) > 1]
#     all_nodes = set().union(*S)

#     # add all vertices of degree >= 3 that are not in bccs
#     for v in grid:
#         if v not in all_nodes and len(get_neighbors(grid, v)) >= 3:
#             S.append({v})

#     # merge while some pairs are <= m−2
#     merged = True
#     while merged:
#         merged = False
#         # iterate all pairs of subgraphs
#         for i in range(len(S)):
#             for j in range(i + 1, len(S)):
#                 Si, Sj = S[i], S[j]
#                 min_dist = float('inf')
#                 closest_pair = None

#                 # Find closest nodes u ∈ Si and v ∈ Sj
#                 for u in Si:
#                     for v in Sj:
#                         dist = DistTable(grid, u).get(v)
#                         if dist < min_dist:
#                             min_dist = dist
#                             closest_pair = (u, v)

#                 if min_dist <= m - 2:
#                     u, v = closest_pair
#                     path_uv = DistTable(grid, u).get_path(v)
#                     Sk = Si.union(Sj).union(set(path_uv))
#                     S = [S[k] for k in range(len(S)) if k != i and k != j]  # Remove Si, Sj
#                     S.append(Sk)
#                     merged = True
#                     break
#             if merged:
#                 break

#     return S

# def tarjan(grid: Grid, starts: Config, goals: Config) -> list[set[Coord]]:
#     """
#     Find all nontrivial biconnected components in the grid using Tarjan's algorithm - O(|V+E|)
#     - disc      : store discovery times of visited vertices
#     - low       : store the lowest discovery time reachable from the vertex (earliest visited vertex)
#     - visited   : set of visited vertices
#     - current   : current path in the DFS traversal
#     """
#     index = 0
#     disc = {}
#     low = {}
#     stack = []
#     visited_edges = set()
#     bccs = []
#     parent = {}
    
#     def dfs(u: Coord, parent_u: Coord | None):
#         nonlocal index
#         disc[u] = low[u] = index
#         index += 1
#         children = 0

#         for v in get_neighbors(grid, u):
#             if (u, v) in visited_edges or (v, u) in visited_edges:
#                 continue
#             visited_edges.add((u, v))
            
#             if v not in disc:
#                 stack.append((u, v))
#                 parent[v] = u
#                 dfs(v, u)
#                 low[u] = min(low[u], low[v])
#                 children += 1
                
#                 if (parent_u is None and children > 1) or (parent_u is not None and low[v] >= disc[u]):
#                     bcc = set()
#                     while stack and stack[-1] != (u, v):
#                         a, b = stack.pop()
#                         bcc.update([a, b])
#                     a, b = stack.pop()
#                     bcc.update([a, b])
#                     if len(bcc) > 1:
#                         bccs.append(bcc)

#             elif v != parent_u and disc[v] < disc[u]:
#                 stack.append((u, v))
#                 low[u] = min(low[u], disc[v])

#     for node in set(starts + goals):
#         if node not in disc:
#             dfs(node, None)

#     return bccs

# def assign_agents_to_subgraphs(grid: Grid, starts: Config, goals: Config, subgraphs: list[set[Coord]]
# ) -> list[tuple[Config, Config]]:
#     occupied = set(starts)
#     asgn = {}
#     agent_at = {pos: i for i, pos in enumerate(starts)}

#     def bfs(grid, start, blocked):
#         visited = set()
#         queue = [start]
#         while queue:
#             v = queue.pop(0)
#             if v in visited or v in blocked:
#                 continue
#             visited.add(v)
#             for u in get_neighbors(grid, v):
#                 if u not in visited and u not in blocked:
#                     queue.append(u)
#         return visited

#     for s in subgraphs:
#         for v in s:
#             # if v is occupied
#             if v in agent_at:
#                 agent = agent_at[v]

#                 # m'': reachable unoccupied in subgraph with v removed
#                 subgraph_without_v = s - {v}
#                 mpp = len([u for u in bfs(grid, v, blocked=occupied - {v})
#                            if u in subgraph_without_v and u not in occupied])

#                 # look for neighbors outside the subgraph
#                 outside_neighbors = [u for u in get_neighbors(grid, v) if u not in s]

#                 for u in outside_neighbors:
#                     # m′: reachable unoccupied from v in full grid with {u,v} edge removed
#                     mprime = len([x for x in bfs(grid, v, blocked=occupied | {u})
#                                   if x not in occupied])

#                     if ((1 <= mprime < len(s)) or mpp >= 1):
#                         asgn[v] = s
#                         break  # assigned, move on

#                 # If v has no outside neighbors but has an agent
#                 if not outside_neighbors:
#                     asgn[v] = s

#     # Construct sub-configs: group starts and goals based on assignment
#     sub_config_map = defaultdict(lambda: ([], []))  # subgraph -> (starts, goals)

#     for start_pos, sub in asgn.items():
#         sub_config_map[frozenset(sub)][0].append(start_pos)

#     for goal_pos in goals:
#         for sub in subgraphs:
#             if goal_pos in sub:
#                 sub_config_map[frozenset(sub)][1].append(goal_pos)
#                 break

#     # Convert to list of (starts_sub, goals_sub)
#     result = []
#     for (starts_sub, goals_sub) in sub_config_map.values():
#         result.append((starts_sub, goals_sub))

#     return result



    
