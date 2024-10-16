import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl

import taskplan
from taskplan.planners.planner import ClosestActionPlanner
from taskplan.pddl.helper import get_learning_informed_plan
from taskplan.utilities.utils import get_container_pose


def get_args():
    args = lambda: None
    args.current_seed = 7000
    args.resolution = 0.05
    args.save_dir = '/data/test_logs/'
    args.image_filename = 'tester.png'
    args.network_file = '/data/taskplan/logs/00_test/gnn.pt'

    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)

    return args


def test_learned_plan():
    args = get_args()

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

    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        seed=args.current_seed
    )

    plan, cost = get_learning_informed_plan(
            pddl=pddl, partial_map=partial_map,
            subgoals=pddl['subgoals'], init_robot_pose=init_robot_pose,
            learned_net=args.network_file)

    if plan:
        for idx, p in enumerate(plan):
            print(idx, p)
    raise NotImplementedError


def test_place_task():
    args = get_args()

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
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
    if plan:
        for p in plan:
            print(p)
        print(cost)
    raise NotImplementedError


def test_replan():
    args = get_args()

    # Load data for a given seed
    thor_data = taskplan.utilities.ai2thor_helper. \
        ThorInterface(args=args)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)

    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        seed=args.current_seed
    )

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
    if plan:
        for p in plan:
            print(p)
        print(cost)

    for action in plan[:4]:
        if action.name == 'move':
            move_start = action.args[0]
            ms_pose = get_container_pose(move_start, partial_map)
            if ms_pose is None:
                ms_pose = init_robot_pose
            move_end = action.args[1]
            me_pose = get_container_pose(move_end, partial_map)
            if me_pose is None:
                me_pose = init_robot_pose

            # Update problem for move action.
            # (rob-at move_end)
            pddl['problem'] = taskplan.pddl.helper.update_problem_move(
                pddl['problem'], move_end)
            # robot_poses.append({(ms_pose, me_pose): 'move'})
        elif action.name == 'pick':
            object_name = action.args[0]
            pick_at = action.args[1]
            pick_pose = get_container_pose(pick_at, partial_map)
            if pick_pose is None:
                pick_pose = init_robot_pose
            # Update problem for pick action.
            # (not (hand-is-free))
            # (not (is-at object location))
            # (is holding object)
            pddl['problem'] = taskplan.pddl.helper.update_problem_pick(
                pddl['problem'], object_name, pick_at)
        elif action.name == 'place':
            object_name = action.args[0]
            place_at = action.args[1]
            place_pose = get_container_pose(place_at, partial_map)
            if place_pose is None:
                place_pose = init_robot_pose
            # Update problem for place action.
            # (hand-is-free)
            # (is-at object location)
            # (not (is holding object))
            pddl['problem'] = taskplan.pddl.helper.update_problem_place(
                pddl['problem'], object_name, place_at)
        elif action.name == 'find':
            obj_name = action.args[0]
            find_start = action.args[1]
            fs_pose = get_container_pose(find_start, partial_map)
            if fs_pose is None:
                fs_pose = init_robot_pose
            find_end = action.args[2]
            fe_pose = get_container_pose(find_end, partial_map)
            if fe_pose is None:
                fe_pose = init_robot_pose
    with open("/data/test_logs/problem.txt", "w") as file:
        file.write(pddl['problem'])

    # execute the first action from the plan
        # if the action is find (only run the subroutine for a step to reveal an unseen location)
    # save the state of the world
    # get new problem file from the updated state of the world
    # replan, until (no actions are in the plan, either of the goals have been reached)
    plt.imshow(whole_graph['graph_image'])
    plt.savefig(f'/data/test_logs/graph_{args.current_seed}.png', dpi=400)
    raise NotImplementedError
