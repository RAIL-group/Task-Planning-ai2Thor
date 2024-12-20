from taskplan.pddl.helper import generate_pddl_problem, get_goals
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

                cnt_names = ['initial_robot_pose']
                cnt_names += [loc['id'] for loc in containers]

                if cnt_name in unvisited:
                    # Object is in the unknown space
                    init_states.append(f"(not (is-located {child_name}))")

                    # The expected find cost needs to be computed via the
                    # model later on. But here we use the optimistic find cost

                    # --- ROOM FOR IMPROVEMENT --- #
                    # if either of the from-loc/to-loc is in subgoals then
                    # the optimistic assumtion would be the missing object can
                    # be found in either. So, taking the distance of from-loc
                    # to to-loc is sufficient
                    for from_loc in cnt_names:
                        for to_loc in cnt_names:
                            d = map_data.known_cost[from_loc][to_loc]
                            init_states.append(f"(= (find-cost {child_name} {from_loc} {to_loc}) {d})")
                    # or else we can optimistically assume the object is in the nearest
                    # undiscovered location from the to-loc [WILL work on it later!!]
                else:
                    # Object is in the known space
                    init_states.append(f"(is-located {child_name})")
                    init_states.append(f"(is-at {child_name} {cnt_name})")

                    # The expected find cost should be sum of the cost to
                    # cnt_name from the from_loc and then the cost to to_loc
                    # from the cnt_name
                    for from_loc in cnt_names:
                        for to_loc in cnt_names:
                            d1 = map_data.known_cost[from_loc][cnt_name]
                            d2 = map_data.known_cost[cnt_name][to_loc]
                            d = d1 + d2
                            init_states.append(f"(= (find-cost {child_name} {from_loc} {to_loc}) {d})")

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

    task = get_goals(seed, cnt_of_interest, obj_of_interest)
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
