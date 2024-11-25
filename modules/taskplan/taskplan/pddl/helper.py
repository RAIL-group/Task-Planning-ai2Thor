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


def get_learning_informed_pddl(pddl, partial_map, subgoals, init_robot_pose, learned_net):
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

    return pddl


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


def update_problem_move(problem, end):
    x = '(rob-at '
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if x in line:
            line = '        ' + x + f'{end})'
            lines[line_idx] = line
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_pick(problem, obj, loc):
    x = f'        (is-holding {obj})'
    insert_x = None
    y = '(hand-is-free)'
    z = f'(is-at {obj} {loc})'
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if y in line:
            line = '        ' + f'(not {y})'
            lines[line_idx] = line
        elif z in line:
            line = '        ' + f'(not {z})'
            lines[line_idx] = line
            insert_x = line_idx + 1
    if insert_x:
        lines.insert(insert_x, x)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_place(problem, obj, loc):
    x = '(not (hand-is-free))'
    y = f'(not (is-at {obj} '
    z = f'(is-holding {obj})'
    delete_z = None
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if x in line:
            line = '        ' + '(hand-is-free)'
            lines[line_idx] = line
        elif y in line:
            line = '        ' + f'(is-at {obj} {loc})'
            lines[line_idx] = line
        elif z in line:
            line = '        ' + f'(not {z})'
            delete_z = line_idx
    if delete_z:
        del lines[delete_z]
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_find(problem, obj, loc):
    y = f'(not (is-located {obj}))'
    z = f'        (is-at {obj} {loc})'
    insert_z = None
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if y in line:
            line = f'        (is-located {obj})'
            lines[line_idx] = line
            insert_z = line_idx + 1
    if insert_z:
        lines.insert(insert_z, z)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def get_goals2(seed, cnt_of_interest, obj_of_interest, objects):
    random.seed(seed)
    object_relations = {
        'plate': ['mug', 'bowl', 'apple'],
        'bowl': ['mug', 'egg'],
        # 'toiletpaper': ['soapbottle'],
        'cellphone': ['pillow']
    }
    pairs = []
    for object in object_relations:
        if object in objects:
            choice1 = random.choice(objects[object])
            for object2 in object_relations[object]:
                if object2 in objects:
                    choice2 = random.choice(objects[object2])
                    pair = (choice1, choice2)
                    pairs.append(pair)

    preferred_containers = ['diningtable', 'chair', 'sofa', 'bed', 'countertop']
    compatible_containers = [cnt for cnt in cnt_of_interest
                             if cnt.split('|')[0] in preferred_containers]

    if compatible_containers == [] or len(pairs) < 3:
        return None

    goal_cnt = random.choice(compatible_containers)
    goal_cnt = [goal_cnt, goal_cnt]
    goal_obj = random.sample(pairs, 3)

    task1 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj[0])

    task2 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj[1])

    task3 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj[2])
    task = [task3, task2, task1]
    task = taskplan.pddl.task.multiple_goal(task)
    return task
