import random
from pddlstream.algorithms.search import solve_from_pddl

import taskplan
from taskplan.planners.planner import LearnedPlanner


def generate_pddl_problem(domain_name, problem_name, objects, init_states,
                          goal_states):
    """
    Generates a PDDL problem file content.

    :param domain_name: Name of the PDDL domain.
    :param problem_name: Name of the PDDL problem.
    :param objects: Dictionary of objects, where keys are types and values are lists of object names.
    :param init_states: List of strings representing the initial states.
    :param goal_states: List of strings representing the goal states.
    :return: A string representing the content of a PDDL problem file.
    """
    # Start the problem definition
    problem_str = f"(define (problem {problem_name})\n"
    problem_str += f"    (:domain {domain_name})\n"

    # Define objects
    problem_str += "    (:objects\n"
    for obj_type, obj_names in objects.items():
        problem_str += "        " + " ".join(obj_names) + " - " + obj_type + "\n"
    problem_str += "    )\n"

    # Define initial state
    problem_str += "    (:init\n"
    for state in init_states:
        problem_str += "        " + state + "\n"
    problem_str += "    )\n"

    # Define goal state
    problem_str += "    (:goal\n"
    problem_str += "        (and\n"
    for state in goal_states:
        problem_str += "            " + state + "\n"
    problem_str += "        )\n"
    problem_str += "    )\n"

    problem_str += "    (:metric minimize (total-cost))\n"

    # Close the problem definition
    problem_str += ")\n"

    return problem_str


def get_pddl_instance(whole_graph, map_data, seed=0):
    # Initialize the environment setting which containers are undiscovered
    init_subgoals_idx = taskplan.utilities.utils.initialize_environment(
        whole_graph['cnt_node_idx'], seed)
    subgoal_IDs = taskplan.utilities.utils.get_container_ID(
        whole_graph['nodes'], init_subgoals_idx)

    # initialize pddl related contents
    pddl = {}
    pddl['domain'] = taskplan.pddl.domain.get_domain(whole_graph)
    pddl['problem'], pddl['goal'] = taskplan.pddl.problem.get_problem(
        map_data=map_data, unvisited=subgoal_IDs, seed=seed)
    pddl['planner'] = 'ff-astar2'  # 'max-astar'
    pddl['subgoals'] = init_subgoals_idx
    return pddl


def get_expected_cost_of_finding(partial_map, subgoals, obj_name,
                                 robot_pose, destination,
                                 learned_net='/data/taskplan/logs/01_cost_fix/gnn.pt'):
    ''' This function calculates and returns the expected cost of finding an object
    given the partial map, initial subgoals, object name, initial robot pose, and a
    learned network path
    '''
    obj_idx = partial_map.idx_map[obj_name]
    partial_map.target_obj = obj_idx
    graph, subgoals = partial_map.update_graph_and_subgoals(subgoals)

    args = lambda: None
    args.network_file = learned_net
    planner = LearnedPlanner(args, partial_map,
                             verbose=False, destination=destination)
    planner.update(graph, subgoals, robot_pose)
    exp_cost, _ = planner.compute_selected_subgoal(return_cost=True)
    return round(exp_cost, 4)


def update_problem(problem, obj, from_loc, to_loc, distance):
    x = f'(= (find-cost {obj} {from_loc} {to_loc}) '
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if x in line:
            line = '        ' + x + f'{distance})'
            lines[line_idx] = line
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def get_learning_informed_plan_old(pddl, partial_map, subgoals, robot_pose, args):
    ''' This function takes input of a PDDL instance, the partial map,
    the subgoals, and the current robot pose, args for nn-file.
    '''
    expected_cost = {}

    prev_plan = None
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                 planner=pddl['planner'])
    while prev_plan != plan:
        prev_plan = plan
        for action in plan:
            if action.name == 'find':
                obj_name = action.args[0]
                curr_rob = action.args[1]
                if obj_name not in expected_cost:
                    expected_cost[obj_name] = {}
                if curr_rob not in expected_cost[obj_name]:
                    expected_cost[obj_name][curr_rob] = get_expected_cost_of_finding(
                        partial_map, subgoals, obj_name, robot_pose, args)
                distance_update = expected_cost[obj_name][curr_rob]
                pddl['problem'] = update_problem(
                    pddl['problem'], obj_name, curr_rob, distance_update)
            elif action.name == 'move':
                container_name = action.args[1]
                container_pose = taskplan.utilities.utils.get_container_pose(
                    container_name, partial_map)
                robot_pose = container_pose

        plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                     planner=pddl['planner'], max_planner_time=300)
        # print(pddl['problem'])
        # print('Plan:', plan)
        # print('Cost:', cost)
        print('Exp cost: ', expected_cost)
    return plan, cost


def get_learning_informed_plan(pddl, partial_map, subgoals, init_robot_pose, learned_net):
    ''' This function takes input of a PDDL instance, the partial map,
    the subgoals, and the current robot pose, args for nn-file.
    '''
    idx2assetID = {partial_map.idx_map[assetID]: assetID for assetID in partial_map.idx_map}

    for obj_idx in partial_map.obj_node_idx:
        obj_cnt_idx = partial_map.org_edge_index[0][partial_map.org_edge_index[1].index(obj_idx)]
        obj_name = idx2assetID[obj_idx]

        cnt_names = ['initial_robot_pose']
        cnt_names += [idx2assetID[cnt_idx] for cnt_idx in partial_map.cnt_node_idx]

        cnt_coords = [init_robot_pose]
        cnt_coords += [partial_map.node_coords[cnt_idx] for cnt_idx in partial_map.cnt_node_idx]

        if obj_cnt_idx in subgoals:
            # Object whereabout is unknwon
            for from_idx, from_loc in enumerate(cnt_names):
                robot_pose = cnt_coords[from_idx]
                for to_idx, to_loc in enumerate(cnt_names):
                    destination_pose = cnt_coords[to_idx]
                    distance_update = get_expected_cost_of_finding(
                        partial_map, subgoals, obj_name, robot_pose, destination_pose, learned_net)
                    pddl['problem'] = update_problem(
                        pddl['problem'], obj_name, from_loc, to_loc, distance_update)

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                 planner=pddl['planner'], max_planner_time=300)

    return plan, cost


def get_goals(seed, cnt_of_interest, obj_of_interest):
    random.seed(seed)
    goal_cnt = random.sample(cnt_of_interest, 2)
    goal_obj = random.sample(obj_of_interest, 2)
    task1 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj)

    goal_cnt = random.sample(cnt_of_interest, 2)
    goal_obj = random.sample(obj_of_interest, 2)
    task2 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj)

    goal_cnt = random.sample(cnt_of_interest, 2)
    goal_obj = random.sample(obj_of_interest, 2)
    task3 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj)
    task = [task3, task2, task1]
    task = taskplan.pddl.task.multiple_goal(task)
    return task
