import heapq
# resotre- stack
# try_swap()
# 


class Agent:
    def __init__(self, start, goal, epsilon):
        self.start = start
        self.goal = goal
        self.epsilon = epsilon
        self.priority = epsilon
        self.path = [start]
        self.stuck_count = 0  # Track failed local attempts
    
    def get_current_position(self, t):
        """Get the last non-None position up to time t"""
        for i in range(min(t, len(self.path)-1), -1, -1):
            if self.path[i] is not None:
                return self.path[i]
        return self.start

class Map:
    def __init__(self, width, height, grid):
        self.width = width
        self.height = height
        self.grid = grid
        self.distance_cache = {}

    @staticmethod
    def parse_map(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            width = int(lines[1].split()[1])
            height = int(lines[2].split()[1])
            grid = [list(line.strip()) for line in lines[4:]]
        return Map(width, height, grid)

    def is_passable(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] in '.'
        return False

    def neighbors(self, x, y, include_current=False, goal=None):
        """Get neighboring cells with optional prioritization toward a goal"""
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected grid
        neighbors = []

        if include_current:
            neighbors.append((x, y))
        
        # current_distance = shortest_path_length_using_a_star(self, (x, y), goal) if goal else 0
        # current_distance = self.distance((x, y), goal) if goal else (0, 0)
            
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if self.is_passable(nx, ny):
                neighbors.append((nx, ny))
        
        # Sort by distance to goal if provided
        if goal is not None:
            neighbors.sort(key=lambda pos: abs(pos[0]-goal[0]) + abs(pos[1]-goal[1]))
        
        return neighbors

    def distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def a_star(map_obj, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: map_obj.distance(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in map_obj.neighbors(*current):
            tentative_g_score = g_score[current] + map_obj.distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + map_obj.distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found

def shortest_path_length_using_a_star(map_obj, start, goal):
    """
    Find the shortest path from start to goal using A* algorithm.
    Returns the length of the path.
    """
    path = a_star(map_obj, start, goal)
    return len(path) if path else float('inf')

def pibt_hybrid(map_obj, agents, max_time=15):
    
    # init paths and priorities
    for agent in agents:
        agent.path = [agent.start]  # πi[0] ← si
        agent.priority = agent.epsilon  # pi ← εi

    for t in range(max_time):
        print(f"------------------------------------------------------------ TIME STEP {t} ")
        
        # Update priorities
        # pi ← if πi[t] = gi then Ei else pi + 1: for each agent ai ∈ A
        for agent in agents:
            if agent.path[-1] == agent.goal:
                # Reset to epsilon when at goal
                agent.priority = agent.epsilon
            else:
                agent.priority += 1  # Increment priority

        # Sort by priority (highest first)
        agents.sort(key=lambda a: -a.priority)

        print("    >> Highest priority:", agents[0].start,"<<\n")
        
        # Process each agent for this timestep
        for agent in agents:
            
            # Ensure path has enough slots
            while len(agent.path) <= t + 1:
                agent.path.append(None)

            # If this agent hasn't been assigned a position for time t, plan it
            if agent.path[t + 1] is None:
                # standard PIBT recursive planning
                if pibt_recursive(agent, None, map_obj, agents, t):
                    continue  # Success, move to next agent
                
                print(f"    No moves worked for {agent.start}, waiting in place.")
                agent.path[t+1] = agent.path[t]  # Wait in place

        # Early termination if all goals reached
        if all(agent.path[-1] == agent.goal for agent in agents):
            break
    
    # Fallback solution: Replan stuck agents with A*
    for agent in agents:
        if len(agent.path) <= t + 1 or agent.path[t + 1] is None:
            print(f"    Agent {agent.start} needs replanning at time {t+1}")
            ensure_path_length(agent, t + 1)
            replan_with_astar(agent, map_obj, t)

    return [agent.path for agent in agents]

def pibt_recursive(ai, aj, map_obj, agents, t):
    """
    Enhanced PIBT function with push, swap, rotate operations integrated.
    """    

    # Check if agent ai already has a position assigned for time t+1
    # if len(ai.path) > t + 1 and ai.path[t + 1] is not None:
    #     print(f"    Agent {ai.start} already assigned: {ai.path[t + 1]}")
    #     return True

    # use last known position 
    current = ai.get_current_position(t)

    # Skip if already at goal 
    if current == ai.goal:
        print(f"    Already at goal {ai.goal}")
        ai.path[t+1] = current
        return True

    # Get candidate moves (prioritize progress toward goal) 
    candidates = get_candidates(ai, map_obj, current)

    if t-1 >= 0 and len(ai.path) > t-1 and ai.path[t-1] is not None:
        # Avoid moving back to the previous position
        if ai.path[t-1] in candidates:
            candidates.remove(ai.path[t-1])

    for v in candidates:

        # Check for vertex conflict - EXclude nodes that are alredy requested by others
        # question: should we check for higher priority agents?
        if any(
            len(a.path) > t+1 and a.path[t+1] == v
            for a in agents if a != ai
        ): 
            print(f"      Vertex conflict at {v}")
            continue
        
        # Avoid swap conflict - EXCLUDE previous position it inherited from
        # A swap conflict occurs if aj is already at v at time t and ai wants to move into v.
        # If aj is not None, it means we are checking for a swap conflict with another agent.  
        if aj is not None and len(aj.path) > t and aj.path[t] == v:
            print(f"      Swap conflict at {v}")
            continue

        # reserve the vertex
        ai.path[t+1] = v
        # print(f"    Agent {ai.start}----------------: {ai.path[t]} -> {v} ")

        # Handle displaced agents - ak is currently at v where ai wants to move into
        # ∃ak ∈ A s.t. πk[t] = v ∧ πk[t + 1] = ⊥ 
        displaced_agents = [
            ak for ak in agents 
            if (ak != ai and 
                len(ak.path) > t and ak.path[t] == v and 
                (len(ak.path) <= t+1 or ak.path[t+1] is None))
        ]

        success = True
        for ak in displaced_agents:
            print("    Displaced agent:", ak.start, "at", v)
            ensure_path_length(ak, t + 1)
            # ak.path[t+1] = None  # Force re-evaluation

            if not pibt_recursive(ak, ai, map_obj, agents, t):  # invalid
                success = False

                """can you do a push, swap, rotate here?"""
                current = ai.get_current_position(t)                
                operation_success = False
                
                if try_push(ai, ak, v, t, map_obj, agents):
                    print(f"    Push operation successful: A{ai.start} -> {v}")
                    operation_success = True
                    break
                
                # Try swap operation
                if try_swap(ai, ak, t, map_obj, agents):
                    print(f"    Swap operation successful: A{ai.start} <-> {ak.start}")
                    operation_success = True
                    break
                
                # Try rotate operation
                # if try_rotate(ai, ak, current, v, t, map_obj, agents):
                #     print(f"    Rotate operation successful involving A{ai.start}")
                #     operation_success = True
                #     break
                    
                if operation_success:
                    return True
                
                # If no operation worked, this agent will need A* replanning later
                if not operation_success:
                    print(f"    All operations failed for agent {ai.start}")
                break
        
        if success:
            print(f"    Agent {ai.start} successfully moved to {v} at time {t+1}")
            return True
        else:
            # Undo reservation and try next candidate
            ak.path[t+1] = None

    if aj is None: 
        print(f"Agent {ai.start}: invalid")
    return False

def backtrack_and_replan(agents, map_obj, t):
    """
    Backtracking mechanism
    """
    if t < 2:
        return False
    
    print(f"    Backtracking to timestep {t-1}")
    
    # Clear the last timestep for all agents
    for agent in agents:
        if len(agent.path) > t:
            agent.path[t] = None
        if len(agent.path) > t + 1:
            agent.path[t + 1] = None
    
    # Try replanning from t-1
    for agent in agents:
        if agent.path[t] is None:
            current = agent.get_current_position(t - 1)
            candidates = get_candidates(agent, map_obj, current)
            
            for candidate in candidates:
                # Check if this candidate is free
                if not any(len(a.path) > t and a.path[t] == candidate for a in agents if a != agent):
                    agent.path[t] = candidate
                    break
            
            if agent.path[t] is None:
                agent.path[t] = current  # Stay in place
    
    return True

def get_position(agent, t):
    return agent.path[t] if len(agent.path) > t else agent.path[-1]

def ensure_path_length(agent, t):
    while len(agent.path) <= t:
        agent.path.append(None)

def try_push(agent, blocker, target_pos, t, map_obj, agents):
    """
    Try to push another agent out of the way by finding them an alternative position.
    """
    blocker_current = blocker.path[t] if t < len(blocker.path) else blocker.path[-1]
    
    # Get possible positions for the blocker to move to
    blocker_candidates = get_candidates(blocker, map_obj, blocker_current)
    
    for new_pos in blocker_candidates:
        # Check if new position is free
        conflict = any(
            a.path[t] == new_pos
            for a in agents if a != blocker
        )
        
        if not conflict and new_pos != target_pos:
            # Push successful - update blocker's position
            ensure_path_length(blocker, t + 1)
            blocker.path[t+1] = new_pos
            
            # Now agent can take target_pos
            ensure_path_length(agent, t + 1)
            agent.path[t+1] = target_pos
            
            print("Pushed agent", blocker.start, "to", new_pos, "for agent", agent.start)
            return True
    
    return False

def try_swap(agent, blocker, t, map_obj, agents): # 
    a_pos = agent.path[t] if t < len(agent.path) else agent.path[-1]
    b_pos = blocker.path[t] if t < len(blocker.path) else blocker.path[-1]

    # Step 1: Find a valid buffer near b_pos ************ both agents find 3degree location - (if theres lower p agents blocking, able to push) - move- swap
    buffer_candidates = map_obj.neighbors(*b_pos, include_current=False) # find 3-degree neighbors
    buffer_candidates.sort(key=lambda pos: shortest_path_length_using_a_star(map_obj, pos, blocker.goal))

    for buffer in buffer_candidates:
        if buffer == a_pos or buffer == b_pos:
            continue

        # Ensure buffer cell is unoccupied at t, t+1, t+2
        conflict = any(
            a != blocker and (
                (len(a.path) > t and a.path[t] == buffer) or
                (len(a.path) > t+1 and a.path[t+1] == buffer) or
                (len(a.path) > t+2 and a.path[t+2] == buffer)
            )
            for a in agents
        )
        if conflict:
            continue

        # Ensure path is long enough
        ensure_path_length(agent, t + 3)
        ensure_path_length(blocker, t + 3)

        # Assign new paths
        # Step 1: B → buffer
        blocker.path[t+1] = buffer
        # Step 2: A → b_pos
        agent.path[t+1] = b_pos
        # Step 3: B → a_pos
        blocker.path[t+2] = a_pos

        print(f"[SWAP 3-DEGREE] A({agent.start}) moves to {b_pos}, B({blocker.start}) to buffer {buffer} then to {a_pos}")
        return True

    return False


def try_rotate(agent, blocker, agent_current, target_pos, t, map_obj, agents):
    """/
    Try to rotate with multiple agents in a cycle.
    This handles cases where agents form a circular dependency.
    """
    # Simple 3-agent rotation: agent -> target_pos, blocker -> agent_current, third -> blocker_current
    blocker_current = blocker.path[t] if t < len(blocker.path) else blocker.path[-1]
    
    # Find an agent that could move to where blocker currently is
    third_agent = None
    for a in agents:
        if a != agent and a != blocker:
            a_current = a.path[t] if t < len(a.path) else a.path[-1]
            # Check if this agent wants to move to blocker's current position
            a_candidates = get_candidates(a, map_obj, a_current, t)
            if blocker_current in a_candidates:
                third_agent = a
                break
    
    if third_agent is None:
        # Try to find empty space for blocker
        blocker_candidates = get_candidates(blocker, map_obj, blocker_current, t)
        for pos in blocker_candidates:
            conflict = any(
                len(a.path) > t and a.path[t] == pos 
                for a in agents if a != blocker
            )
            if not conflict and pos != target_pos and pos != agent_current:
                # Found empty space - simple push
                while len(blocker.path) <= t:
                    blocker.path.append(None)
                blocker.path[t] = pos
                
                while len(agent.path) <= t:
                    agent.path.append(None)
                agent.path[t] = target_pos

                print("Rotated agent", agent.start, "to", target_pos, "for agent", blocker.start)
                return True
        return False
    
    # Perform 3-way rotation
    third_current = third_agent.path[t] if t < len(third_agent.path) else third_agent.path[-1]
    
    # Check if rotation is valid (no other conflicts)
    positions = [target_pos, agent_current, blocker_current]
    rotation_agents = [agent, blocker, third_agent]
    
    for i, pos in enumerate(positions):
        conflict = any(
            a not in rotation_agents and len(a.path) > t and a.path[t] == pos 
            for a in agents
        )
        if conflict:
            return False
    
    # Execute rotation
    while len(agent.path) <= t:
        agent.path.append(None)
    while len(blocker.path) <= t:
        blocker.path.append(None)
    while len(third_agent.path) <= t:
        third_agent.path.append(None)
        
    agent.path[t] = target_pos
    blocker.path[t] = agent_current  
    third_agent.path[t] = blocker_current
    
    print(f"Rotated agents {agent.start}, {blocker.start}, {third_agent.start} at time {t}")
    return True

def replan_with_astar(agent, map_obj, t):
    start = agent.path[t] if t < len(agent.path) else agent.path[-1]
    new_path = a_star(map_obj, start, agent.goal)
    if new_path:
        agent.path = agent.path[:t] + new_path  # Replace remaining path

def get_candidates(agent, map_obj, current_pos):
    # Get all neighbors including current position
    candidates = map_obj.neighbors(*current_pos, include_current=False, goal=agent.goal)

    candidates.sort(key=lambda pos: shortest_path_length_using_a_star(map_obj, pos, agent.goal))
    return candidates

if __name__ == "__main__":
    # map_obj = Map.parse_map('pibt_rip/maps/small.map')

    # # exp usage with 2 agents
    # agents = [
    #     Agent((0, 0), (4, 4), 1),  # Agent 1: Top-left to bottom-right
    #     Agent((4, 0), (0, 4), 2)   # Agent 2: Top-right to bottom-left
    # ]

    # paths = pibt_hybrid(map_obj, agents)

    # print("\n****************** PIBT Hybrid Results ***************")
    # for i, path in enumerate(paths):
    #     print(f"Agent {i+1} path: {path}")
    # print("Path length:", len(agents[0].path))
    # print("All agents reached their goals!" if all(agent.path[-1] == agent.goal for agent in agents) else "Some agents did not reach their goals.")
    # print("****************** End of Results ***************")

    # print("\n")

    # # Example with 3 agents
    # agents = [
    #     Agent((0, 0), (4, 4), 1),  # Agent 1: Start at (0,0), Goal at (4,4)
    #     Agent((0, 3), (3, 3), 3),  # Agent 2: Start at (0,3), Goal at (3,3)
    #     Agent((4, 0), (0, 4), 2)   # Agent 3: Start at (4,0), Goal at (0,4)
    # ]

    # paths = pibt_hybrid(map_obj, agents)

    # print("\n****************** Pure PIBT Results (3 agents) ***************")
    # for i, path in enumerate(paths):
    #     print(f"Agent {i+1} path: {path}")
    # print("****************** End of Results ***************")
    # print("\n")

    # test push
    agents = [
        Agent((1, 1), (4, 0), 1),  
        Agent((2, 1), (0, 0), 2)   
    ]

    map_obj = Map.parse_map('./assets/pushmap.map')

    paths = pibt_hybrid(map_obj, agents)

    print("\n****************** Pure PIBT Results (push agents) ***************")
    for i, path in enumerate(paths):
        print(f"Agent {i+1} path: {path}")
    print("****************** End of Results ***************")
    print("\n")