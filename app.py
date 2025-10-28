import argparse
import os

from pibt.utils import (
    parse_map, 
    get_scenario, 
    is_valid_mapf_solution, 
    save_configs_for_visualizer
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--map-file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "assets", "small.map"
        ),
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "assets", "small-random-1.scen"
        ),
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=10000)

    parser.add_argument(
        "--pibt-version",
        type=str,
        choices=["pibt_new", "oriori", "backup", "lacam"],
        default="pibt_new",
        help="Select which PIBT implementation to use"
    )

    # LaCAM-specific arguments
    parser.add_argument(
        "--time-limit-ms",
        type=int,
        default=10000,
        help="Time limit in milliseconds for LaCAM solver"
    )
    parser.add_argument(
        "--flg-star",
        action="store_true",
        help="Use LaCAM* variant"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level for LaCAM"
    )

    parser.add_argument("--grid", dest="show_grid", action="store_true",
                        help="Show grid on the environment or not")
    parser.add_argument("--aid", dest="show_ag_idx", action="store_true",
                        help="Show agent indices or not")
    parser.add_argument("--tid", dest="show_task_idx", action="store_true",
                        help="Show task indices or not")
    parser.add_argument("--plan", type=str, help="Path to the planned path file")
    args = parser.parse_args()

    if args.pibt_version in ["pibt_new", "oriori", "backup"]:

        if args.pibt_version == "pibt_new":
            from pibt.pibt_new import PIBT
        elif args.pibt_version == "oriori":
            from pibt.oriori import PIBT
        elif args.pibt_version == "backup":
            from pibt.backup import PIBT

        print(f"Using PIBT version: {args.pibt_version}")

        # define problem instance
        grid = parse_map(args.map_file)
        starts, goals = get_scenario(args.scen_file, args.num_agents)

        # solve MAPF
        pibt = PIBT(grid, starts, goals, seed=args.seed)
        plan = pibt.run(max_timestep=args.max_timestep)

        # validation: True -> feasible solution
        print(f"solved: {is_valid_mapf_solution(grid, starts, goals, plan)}")

        # save result
        save_configs_for_visualizer(plan, args.output_file)

    elif args.pibt_version == "lacam":
        from pylacam.src.pycam.lacam import LaCAM
        
        print(f"Using LaCAM")
        
        grid = parse_map(args.map_file)
        starts, goals = get_scenario(args.scen_file, args.num_agents)

        # solve MAPF
        planner = LaCAM()
        solution = planner.solve(
            grid=grid,
            starts=starts,
            goals=goals,
            seed=args.seed,
            time_limit_ms=args.time_limit_ms,
            flg_star=args.flg_star,
            verbose=args.verbose,
        )

        # validation: True -> feasible solution
        print(f"solved: {is_valid_mapf_solution(grid, starts, goals, solution)}")

        # save result
        save_configs_for_visualizer(solution, args.output_file)