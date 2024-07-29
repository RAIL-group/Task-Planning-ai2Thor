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

    ################
    # ~~~ Naive ~~ #
    ################
    naive_planner = ClosestActionPlanner(args, partial_map)
    naive_cost_str = 'naive'
    naive_planning_loop = taskplan.planners.planning_loop.PlanningLoop(
        partial_map=partial_map, robot=init_robot_pose, args=args,
        verbose=True)

    for counter, step_data in enumerate(naive_planning_loop):
        # Update the planner objects
        naive_planner.update(
            step_data['graph'],
            step_data['subgoals'],
            step_data['robot_pose'])

        # Compute the next subgoal and set to the planning loop
        chosen_subgoal = naive_planner.compute_selected_subgoal()
        naive_planning_loop.set_chosen_subgoal(chosen_subgoal)

    naive_path = naive_planning_loop.robot
    naive_dist, naive_trajectory = taskplan.core.compute_path_cost(partial_map.grid, naive_path)

    ################
    # ~~ Learned ~ #
    ################
    learned_planner = LearnedPlanner(args, partial_map)
    learned_cost_str = 'learned'
    learned_planning_loop = taskplan.planners.planning_loop.PlanningLoop(
        partial_map=partial_map, robot=init_robot_pose, args=args,
        verbose=True)

    for counter, step_data in enumerate(learned_planning_loop):
        # Update the planner objects
        learned_planner.update(
            step_data['graph'],
            step_data['subgoals'],
            step_data['robot_pose'])

        # Compute the next subgoal and set to the planning loop
        chosen_subgoal = learned_planner.compute_selected_subgoal()
        learned_planning_loop.set_chosen_subgoal(chosen_subgoal)

    learned_path = learned_planning_loop.robot
    learned_dist, learned_trajectory = taskplan.core.compute_path_cost(partial_map.grid, learned_path)
    with open(logfile, "a+") as f:
        f.write(f"[Evaluation] s: {args.current_seed:4d}"
                f" | naive: {naive_dist:0.3f}"
                f" | learned: {learned_dist:0.3f}\n"
                f"  Steps: {len(naive_path)-1:3d}"
                f" | {len(learned_path)-1:3d}\n")

    plt.clf()
    plt.figure(figsize=(10, 5))
    what = partial_map.org_node_names[partial_map.target_obj]
    where = [partial_map.org_node_names[goal] for goal in partial_map.target_container]
    plt.suptitle(f"Find {what} from {where} in seed: [{args.current_seed}]", fontsize=9)

    plt.subplot(231)
    # 1 plot the whole graph
    plt.title('Whole scene graph', fontsize=6)
    graph_image = whole_graph['graph_image']
    plt.imshow(graph_image)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])
    ######################

    plt.subplot(232)
    # 2 plot the graph overlaied image
    taskplan.plotting.plot_graph_on_grid(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Graph overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    ######################

    plt.subplot(233)
    # 3 plot the top-dwon-view
    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)
    plt.title('Top-down view of the map', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    ######################

    plt.subplot(234)
    # 4 plot the grid with naive trajectory viridis color
    plotting_grid = taskplan.plotting.make_plotting_grid(
        np.transpose(grid)
    )
    plt.imshow(plotting_grid)
    plt.title(f"{naive_cost_str} Cost: {naive_dist:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = naive_path[0]
    plt.text(x, y, '0 - ROBOT', color='brown', size=4)

    for idx, coords in enumerate(naive_path[1:]):
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
    colors = np.linspace(0, 1, len(naive_trajectory[0]))
    line_colors = viridis_cmap(colors)

    # Plot the points with Viridis color gradient
    for idx, x in enumerate(naive_trajectory[0]):
        y = naive_trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)
    ######################

    plt.subplot(235)
    # 4 plot the grid with learned trajectory viridis color
    # plotting_grid = taskplan.plotting.make_plotting_grid(
    #     np.transpose(grid)
    # )
    plt.imshow(plotting_grid)
    plt.title(f"{learned_cost_str} Cost: {learned_dist:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = learned_path[0]
    plt.text(x, y, '0 - ROBOT', color='brown', size=4)

    for idx, coords in enumerate(learned_path[1:]):
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
    colors = np.linspace(0, 1, len(learned_trajectory[0]))
    line_colors = viridis_cmap(colors)

    # Plot the points with Viridis color gradient
    for idx, x in enumerate(learned_trajectory[0]):
        y = learned_trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)
    ######################

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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
