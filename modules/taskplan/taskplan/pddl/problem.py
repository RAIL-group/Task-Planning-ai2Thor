import random

import taskplan
from taskplan.pddl.helper import generate_pddl_problem
from taskplan.utilities.ai2thor_helper import get_generic_name


def get_problem(map_data, unvisited, seed=0):
    obj_of_interest = []
    cnt_of_interest = []
    containers = map_data.containers
    objects = {
         'init_r': ['initial_robot_pose']
    }
    init_states = [
        '(= (total-cost) 0)',
        '(restrict-move-to initial_robot_pose)',
        '(hand-is-free)',
        '(rob-at initial_robot_pose)'  # , '(is-fillable coffeemachine)'
    ]
    for container in containers:
        cnt_name = container['id']
        cnt_of_interest.append(cnt_name)
        gen_name = get_generic_name(cnt_name)
        if gen_name not in objects:
            objects[gen_name] = [cnt_name]
        else:
            objects[gen_name].append(cnt_name)
        children = container.get('children')
        if children is not None:
            for child in children:
                child_name = child['id']
                obj_of_interest.append(child_name)
                gen_name_child = get_generic_name(child_name)

                if gen_name_child not in objects:
                    objects[gen_name_child] = [child_name]
                else:
                    objects[gen_name_child].append(child_name)

                if cnt_name in unvisited:
                    init_states.append(f"(not (is-located {child_name}))")
                    init_states.append(f"(= (find-cost {child_name}) 0)")
                else:
                    init_states.append(f"(is-located {child_name})")
                    init_states.append(f"(is-at {child_name} {cnt_name})")
                    init_states.append(f"(= (find-cost {child_name}) 0)")
    #             if 'isLiquid' in child and child['isLiquid'] == 1:
    #                 init_states.append(f"(is-liquid {chld_name})")
    #             if 'pickable' in child and child['pickable'] == 1:
                init_states.append(f"(is-pickable {child_name})")
    #             if 'spreadable' in child and child['spreadable'] == 1:
    #                 init_states.append(f"(is-spreadable {chld_name})")
    #             if 'washable' in child and child['washable'] == 1:
    #                 init_states.append(f"(is-washable {chld_name})")
    #             if 'dirty' in child and child['dirty'] == 1:
    #                 init_states.append(f"(is-dirty {chld_name})")
    #             if 'spread' in child and child['spread'] == 1:
    #                 init_states.append(f"(is-spread {chld_name})")
    #             if 'fillable' in child and child['fillable'] == 1:
    #                 init_states.append(f"(is-fillable {chld_name})")
    #             if 'folded' in child and child['folded'] == 1:
    #                 init_states.append(f"(is-folded {chld_name})")
    #             if 'foldable' in child and child['foldable'] == 1:
    #                 init_states.append(f"(is-foldable {chld_name})")

    for c1 in map_data.known_cost:
        for c2 in map_data.known_cost[c1]:
            if c1 == c2:
                continue
            val = map_data.known_cost[c1][c2]
            init_states.append(
                f"(= (known-cost {c1} {c2}) {val})"
            )

    random.seed(seed)
    goal_cnt = random.sample(cnt_of_interest, 2)
    goal_obj = random.sample(obj_of_interest, 2)
    task = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj)
    print(f'Goal: {task}')
    goal = [task]
    PROBLEM_PDDL = generate_pddl_problem(
        domain_name='indoor',
        problem_name='pick-place-problem',
        objects=objects,
        init_states=init_states,
        goal_states=goal
    )
    return PROBLEM_PDDL, task
