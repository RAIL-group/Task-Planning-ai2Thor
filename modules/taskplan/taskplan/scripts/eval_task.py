import os
import time
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl

import taskplan
from taskplan.planners.planner import ClosestActionPlanner, LearnedPlanner
from taskplan.pddl.helper import get_learning_informed_plan


def evaluate_main(args):
    # Load data for a given seed
    thor_data = taskplan.utilities.ai2thor_helper. \
        ThorInterface(args=args)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        seed=args.current_seed
    )

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)

    if args.logfile_name == 'task_learned_logfile.txt':
        plan, cost = get_learning_informed_plan(
            pddl=pddl, partial_map=partial_map,
            subgoals=pddl['subgoals'], init_robot_pose=init_robot_pose,
            learned_net=args.network_file)
        cost_str = 'learned'
    else:
        plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                     planner=pddl['planner'], max_planner_time=300)
        if args.logfile_name == 'task_naive_logfile.txt':
            cost_str = 'naive'
        elif args.logfile_name == 'task_learned_sp_logfile.txt':
            cost_str = 'learned_sp'
    if plan:
        for p in plan:
            print(p)
        print(cost)
    else:
        plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=1200)
        exit()

    find, robot_poses = taskplan.utilities.utils.get_object_to_find_from_plan(
        plan=plan, partial_map=partial_map, init_robot_pose=init_robot_pose)

    # Intialize logfile
    logfile = os.path.join(args.save_dir, args.logfile_name)
    with open(logfile, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")

    search_poses = {}

    for obj_idx in find:
        if args.logfile_name == 'task_naive_logfile.txt':
            planner = ClosestActionPlanner(args, partial_map,
                                           destination=find[obj_idx]['to'])
        elif args.logfile_name == 'task_learned_sp_logfile.txt':
            planner = LearnedPlanner(args, partial_map, verbose=True,
                                     destination=find[obj_idx]['to'])
        elif args.logfile_name == 'task_learned_logfile.txt':
            planner = LearnedPlanner(args, partial_map, verbose=True,
                                     destination=find[obj_idx]['to'])

        partial_map.target_obj = obj_idx
        planning_loop = taskplan.planners.planning_loop.PlanningLoop(
            partial_map=partial_map, robot=find[obj_idx]['from'],
            destination=find[obj_idx]['to'],
            args=args, verbose=True)
        # we set the subgoals from pddl initialization
        # however, for each search we reconsider the subgoals to be unexplored
        # by copying; remove copy to continue from same state but in that
        # case the pddl problem needs to update and pddl solver rerun
        planning_loop.subgoals = pddl['subgoals'].copy()

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

        search_poses[(find[obj_idx]['from'], find[obj_idx]['to'])] = planning_loop.robot

    distances = []
    trajectories = []
    is_find = []
    t = []
    for robot_pose in robot_poses:
        for rp in robot_pose:
            action_type = robot_pose[rp]
        if action_type == 'find':
            robot_pose = search_poses[rp]
            is_find.append(True)
        elif action_type == 'move':
            robot_pose = [rp[0], rp[1]]
            is_find.append(False)
        t.extend(robot_pose)
        dist, traj = taskplan.core.compute_path_cost(partial_map.grid, robot_pose)
        distances.append(dist)
        trajectories.append(traj)
    total_dist, trajectory = taskplan.core.compute_path_cost(partial_map.grid, t)
    # assert dist == total_dist
    dist = sum(distances)
    print(f"Planning cost: {dist}")
    with open(logfile, "a+") as f:
        # err_str = '' if did_succeed else '[ERR]'
        f.write(f"[Learn] s: {args.current_seed:4d}"
                f" | {cost_str}: {dist:0.3f}\n")

    plt.clf()
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"{pddl['goal']} - seed: [{args.current_seed}]", fontsize=9)

    plt.subplot(231)
    # 0 plot the plan
    taskplan.plotting.plot_plan(plan=plan)

    plt.subplot(232)
    # 1 plot the whole graph
    plt.title('Whole scene graph', fontsize=6)
    graph_image = whole_graph['graph_image']
    plt.imshow(graph_image)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

    plt.subplot(233)
    # 2 plot the top-dwon-view
    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)
    plt.title('Top-down view of the map', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(235)
    # 3 plot the graph overlaied image
    taskplan.plotting.plot_graph_on_grid(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Graph overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(236)
    # 4 plot the grid with trajectory viridis color
    plotting_grid = taskplan.plotting.make_blank_grid(
        np.transpose(grid)
    )
    plt.imshow(plotting_grid)
    plt.title(f"{cost_str} Cost: {dist:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    viridis_cmap = plt.get_cmap('viridis')

    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)

    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    # Hide box and ticks
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=1200)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for Task Planning under uncertainty"
    )
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--image_filename', type=str, required=True)
    parser.add_argument('--logfile_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True)
    parser.add_argument('--network_file', type=str, required=False)
    # parser.add_argument('--simulate', action='store_true', required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
