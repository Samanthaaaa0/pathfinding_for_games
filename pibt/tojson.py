
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from .utils import Config, Configs, Coord

@dataclass 
class PIBTResults:
    """Container for PIBT algorithm results"""
    configs: Configs
    start_config: Config
    goal_config: Config
    grid_shape: Tuple[int, int]
    num_agents: int
    errors: List = None
    planner_times: List[float] = None

class PIBTJSONConverter:
    """Converts PIBT results to the required JSON format"""
    
    def __init__(self):
        self.action_symbols = {
            'forward': 'F',
            'clockwise': 'R', 
            'counter_clockwise': 'C',
            'wait': 'W',
            'timeout': 'T'
        }
        
    def coords_to_direction(self, from_coord: Coord, to_coord: Coord) -> str:
        """Convert coordinate movement to direction"""
        dx = to_coord[0] - from_coord[0]
        dy = to_coord[1] - from_coord[1]
        
        if dx == 0 and dy == 0:
            return 'wait'
        elif dx == 0 and dy == 1:  # Moving right
            return 'east'
        elif dx == 0 and dy == -1:  # Moving left  
            return 'west'
        elif dx == 1 and dy == 0:  # Moving down
            return 'south'
        elif dx == -1 and dy == 0:  # Moving up
            return 'north'
        else:
            raise ValueError(f"Invalid movement: {from_coord} -> {to_coord}")
    
    def get_turn_action(self, current_dir: str, target_dir: str) -> str:
        """Determine turn action needed to face target direction"""
        directions = ['north', 'east', 'south', 'west']
        
        if target_dir == 'wait':
            return 'wait'
            
        current_idx = directions.index(current_dir)
        target_idx = directions.index(target_dir)
        
        # Calculate turn difference
        diff = (target_idx - current_idx) % 4
        
        if diff == 0:
            return 'forward'
        elif diff == 1:
            return 'clockwise'
        elif diff == 3:
            return 'counter_clockwise' 
        else:  # diff == 2, 180 degree turn
            return 'clockwise'  # Choose clockwise for 180 turns
    
    def convert_path_to_actions(self, agent_configs: List[Coord]) -> str:
        """Convert agent's coordinate path to action sequence"""
        if len(agent_configs) <= 1:
            return ""
            
        actions = []
        current_orientation = 'north'  # Default starting orientation
        
        for i in range(len(agent_configs) - 1):
            from_pos = agent_configs[i]
            to_pos = agent_configs[i + 1]
            
            # Get movement direction
            move_direction = self.coords_to_direction(from_pos, to_pos)
            
            if move_direction == 'wait':
                actions.append(self.action_symbols['wait'])
            else:
                # Determine required turn and movement
                turn_action = self.get_turn_action(current_orientation, move_direction)
                
                if turn_action == 'forward':
                    actions.append(self.action_symbols['forward'])
                elif turn_action == 'clockwise':
                    actions.append(self.action_symbols['clockwise'])
                    actions.append(self.action_symbols['forward'])
                    current_orientation = move_direction
                elif turn_action == 'counter_clockwise':
                    actions.append(self.action_symbols['counter_clockwise'])
                    actions.append(self.action_symbols['forward'])
                    current_orientation = move_direction
                else:  # wait
                    actions.append(self.action_symbols['wait'])
                
                # Update orientation after movement
                if move_direction != 'wait':
                    current_orientation = move_direction
        
        return ','.join(actions)
    
    def extract_agent_paths(self, configs: Configs, num_agents: int) -> List[List[Coord]]:
        """Extract individual agent paths from configuration sequence"""
        agent_paths = [[] for _ in range(num_agents)]
        
        for config in configs:
            for agent_id in range(num_agents):
                agent_paths[agent_id].append(config[agent_id])
                
        return agent_paths
    
    def calculate_metrics(self, configs: Configs, start_config: Config, goal_config: Config) -> Dict:
        """Calculate performance metrics"""
        num_tasks_finished = sum(1 for i, pos in enumerate(configs[-1]) 
                                if pos == goal_config[i])
        
        # Calculate sum of cost (total path length)
        sum_of_cost = sum(len(self.extract_agent_paths(configs, len(start_config))[i]) - 1 
                         for i in range(len(start_config)))
        
        # Makespan is the number of timesteps
        makespan = len(configs) - 1
        
        return {
            'numTaskFinished': num_tasks_finished,
            'sumOfCost': sum_of_cost,
            'makespan': makespan
        }
    
    def convert_to_json(self, results: PIBTResults, output_file: str = None) -> Dict:
        """Convert PIBT results to JSON format"""
        
        # Extract agent paths
        agent_paths = self.extract_agent_paths(results.configs, results.num_agents)
        
        # Convert paths to action sequences
        actual_paths = []
        planner_paths = []  # For PIBT, actual and planner paths are the same
        
        for agent_id in range(results.num_agents):
            action_sequence = self.convert_path_to_actions(agent_paths[agent_id])
            actual_paths.append(action_sequence)
            planner_paths.append(action_sequence)  # Same as actual for PIBT
        
        # Calculate metrics
        metrics = self.calculate_metrics(results.configs, results.start_config, results.goal_config)
        
        # Build JSON structure
        json_output = {
            "actionModel": "MAPF_T",
            "teamSize": results.num_agents,
            "start": [[pos[0], pos[1]] for pos in results.start_config],
            "numTaskFinished": metrics['numTaskFinished'],
            "sumOfCost": metrics['sumOfCost'],
            "makespan": metrics['makespan'],
            "actualPaths": actual_paths,
            "plannerPaths": planner_paths,
            "plannerTimes": results.planner_times or [0.0] * len(results.configs),
            "errors": results.errors or [],
            "actualSchedule": ["0:0"] * results.num_agents,  # Simple single-task schedule
            "plannerSchedule": ["0:0"] * results.num_agents,  # Same as actual
            "events": [],  # Would need task information to populate
            "scheduleErrors": [],
            "tasks": [],  # Would need task definitions
            "numPlannerErrors": 0,
            "numScheduleErrors": 0,
            "numEntryTimeouts": 0
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(json_output, f, indent=2)
                
        return json_output