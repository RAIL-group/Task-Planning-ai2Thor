import os
import time
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import taskplan
from taskplan.planners.planner import ClosestActionPlanner, \
    LearnedPlanner, KnownPlanner


def evaluate_main(args):
    # Load data for a given seed
    thor_data = taskplan.utilities.ai2thor_helper. \
        ThorInterface(args=args)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid)

    # Intialize logfile
    logfile = os.path.join(args.save_dir, args.logfile_name)
    with open(logfile, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")
    if args.logfile_name == 'naive_logfile.txt':
        planner = ClosestActionPlanner(args, partial_map)
        cost_str = 'naive'
    elif args.logfile_name == 'learned_logfile.txt':
        planner = LearnedPlanner(args, partial_map, verbose=True)
        cost_str = 'learned'
    elif args.logfile_name == 'known_logfile.txt':
        planner = KnownPlanner(args, partial_map)
        cost_str = 'known'
    planning_loop = taskplan.planners.planning_loop.PlanningLoop(
        partial_map=partial_map, robot=init_robot_pose, args=args,
        verbose=True)

    for counter, step_data in enumerate(planning_loop):
        # Update the planner objects
        s_time = time.time()
        planner.update(
            step_data['graph'],
            step_data['subgoals'],
            step_data['robot_pose'])
        print(f"Time taken to update: {time.time() - s_time}")

        # Compute the next subgoal and set to the planning loop
        s_time = time.time()
        chosen_subgoal = planner.compute_selected_subgoal()
        print(f"Time taken to choose subgoal: {time.time() - s_time}")
        planning_loop.set_chosen_subgoal(chosen_subgoal)

    path = planning_loop.robot
    dist, trajectory = taskplan.core.compute_path_cost(partial_map.grid, path)

    print(f"Planning cost: {dist}")
    with open(logfile, "a+") as f:
        # err_str = '' if did_succeed else '[ERR]'
        f.write(f"[Learn] s: {args.current_seed:4d}"
                f" | {cost_str}: {dist:0.3f}\n"
                f"  Steps: {len(path)-1:3d}\n")
    plt.clf()
    plt.figure(figsize=(10, 4))
    what = partial_map.org_node_names[partial_map.target_obj]
    where = [partial_map.org_node_names[goal] for goal in partial_map.target_container]
    plt.suptitle(f"Find {what} from {where} in seed: [{args.current_seed}]", fontsize=9)

    plt.subplot(131)
    # 1 plot the graph overlaied image
    taskplan.plotting.plot_graph_on_grid(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Graph overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(132)
    # 2 plot the grid with trajectory viridis color
    plotting_grid = taskplan.plotting.make_plotting_grid(
        np.transpose(grid)
    )
    plt.imshow(plotting_grid)
    plt.title(f"{cost_str} Cost: {dist:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = path[0]
    plt.text(x, y, '0 - ROBOT', color='brown', size=4)

    for idx, coords in enumerate(path[1:]):
        # find the node_idx for this pose and use it through
        # graph['node_coords']
        pose = taskplan.utilities.utils. \
            get_pose_from_coord(coords, whole_graph)
        x = whole_graph['node_coords'][pose][0]
        y = whole_graph['node_coords'][pose][1]
        name = whole_graph['node_names'][pose]
        plt.text(x, y, f'{idx+1} - {pose}:{name}', color='brown', size=4)

    # Create a Viridis color map
    viridis_cmap = plt.get_cmap('viridis')
    # Generate colors based on the Viridis color map
    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)

    # Plot the points with Viridis color gradient
    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    plt.subplot(133)
    # 3 plot the top-dwon-view
    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)
    plt.title('Top-down view of the map', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    if args.simulate:
        args.cost_str = cost_str
        taskplan.plotting.simulate_plan(
            trajectory=trajectory,
            thor_data=thor_data,
            args=args
        )
    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=1200)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for Object Search"
    )
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--image_filename', type=str, required=True)
    parser.add_argument('--logfile_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True)
    parser.add_argument('--network_file', type=str, required=False)
    parser.add_argument('--simulate', action='store_true', required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
