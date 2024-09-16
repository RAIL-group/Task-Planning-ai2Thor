import math
import random
import itertools
import numpy as np

import gridmap
# import taskplan
import lsp_accel


IS_FROM_LAST_CHOSEN_REWARD = 0 * 10.0


class Subgoal:
    def __init__(self, value) -> None:
        self.value = value
        self.props_set = False
        self.is_from_last_chosen = False
        self.is_obstructed = False
        self.prob_feasible = 1.0
        self.delta_success_cost = 0.0
        self.exploration_cost = 0.0
        self.negative_weighting = 0.0
        self.positive_weighting = 0.0

        self.counter = 0
        # Compute and cache the hash
        self.hash = hash(self.value)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def set_props(self,
                  prob_feasible,
                  is_obstructed=False,
                  delta_success_cost=0,
                  exploration_cost=0,
                  positive_weighting=0,
                  negative_weighting=0,
                  counter=0,
                  last_observed_pose=None,
                  did_set=True):
        self.props_set = did_set
        self.just_set = did_set
        self.prob_feasible = prob_feasible
        self.is_obstructed = is_obstructed
        self.delta_success_cost = delta_success_cost
        self.exploration_cost = exploration_cost
        self.positive_weighting = positive_weighting
        self.negative_weighting = negative_weighting
        self.counter = counter
        self.last_observed_pose = last_observed_pose


class PartialMap:
    ''' This class is responsible for creating the core capabilites like
    creating partial graph from the full graph, getting subgoals, performing
    change is graph based on action, etc.
    container_nodes carries the information about container poses
    '''
    def __init__(self, graph, grid=None, distinct=False):
        self.org_node_feats = graph['graph_nodes']
        self.org_edge_index = graph['graph_edge_index']
        self.org_node_names = graph['node_names']
        self.cnt_node_idx = graph['cnt_node_idx']
        self.obj_node_idx = graph['obj_node_idx']
        self.node_coords = graph['node_coords']
        self.idx_map = graph['idx_map']
        # self.distances = graph['distances']
        self.distinct = distinct  # when true looks for specific object instance

        self.target_obj = random.sample(self.obj_node_idx, 1)[0]
        self.container_poses = self._get_container_poses()

        if grid is not None:
            self.grid = grid

    def _get_container_poses(self):
        return {container_idx: tuple(self.node_coords[container_idx][0:2])
                for container_idx in self.cnt_node_idx}

    def _get_object_free_graph(self):
        # Trimdown all object nodes to get an object node free graph
        obj_count = len(self.obj_node_idx)

        temp = self.org_edge_index.copy()
        temp[0] = temp[0][:-obj_count:]
        temp[1] = temp[1][:-obj_count:]

        return {
            'node_feats': self.org_node_feats[:-obj_count:],
            'edge_index': temp
        }

    def initialize_graph_and_subgoals(self, seed=0):
        random.seed(seed)
        if self.distinct:
            target_obj_container_idx = [self.org_edge_index[0][self.org_edge_index[1].index(self.target_obj)]]
        else:
            # Find the container index containing the target object
            all_target = []
            for obj_idx in self.obj_node_idx:
                if self.org_node_names[obj_idx] == self.org_node_names[self.target_obj]:
                    all_target.append(obj_idx)
            target_obj_container_idx = {self.org_edge_index[0][self.org_edge_index[1].index(obj_idx)]
                                        for obj_idx in all_target}
        self.target_container = target_obj_container_idx

        # select 50% or above nodes as subgoals from the original
        # container nodes, but no less than 2
        cnt_count = len(self.cnt_node_idx)
        lb_sample = min(cnt_count, 2)
        num_of_val_to_choose = max(lb_sample, random.sample(list(range(
            cnt_count // 2, cnt_count)), 1)[0])
        subgoal_idx = random.sample(self.cnt_node_idx, num_of_val_to_choose)
        for target_obj_cnt in target_obj_container_idx:
            if target_obj_cnt not in subgoal_idx:
                subgoal_idx.append(target_obj_cnt)
        subgoal_idx = sorted(subgoal_idx)

        # Extract the container nodes' index that were not chosen
        # as subgoal nodes
        cnt_to_reveal_idx = [xx
                             for xx in self.cnt_node_idx
                             if xx not in subgoal_idx]

        # Check if the cnt_nodes to reveal had any connections in original
        # graph and update the initial graph adding those connection and nodes
        # they were connected to
        graph = self._get_object_free_graph().copy()

        for node_idx in cnt_to_reveal_idx:
            # if the node has connections in original
            # update graph: add edge to object node
            # append revealed object's feature to node_feats
            connected_obj_idx = [
                self.org_edge_index[1][idx]
                for idx, node in enumerate(self.org_edge_index[0])
                if node == node_idx
            ]

            for obj_idx in connected_obj_idx:
                graph['edge_index'][0].append(node_idx)
                graph['edge_index'][1].append(len(graph['node_feats']))
                graph['node_feats'].append(self.org_node_feats[obj_idx])

        return graph, subgoal_idx

    def update_graph_and_subgoals(self, subgoals, chosen_subgoal=None):
        if chosen_subgoal is not None:
            subgoals.remove(chosen_subgoal)

        subgoal_idx = subgoals
        cnt_to_reveal_idx = [xx
                             for xx in self.cnt_node_idx
                             if xx not in subgoal_idx]

        graph = self._get_object_free_graph()

        for node_idx in cnt_to_reveal_idx:
            # if the node has connections in original
            # update graph: add edge to object node
            # append revealed object's feature to node_feats
            connected_obj_idx = [
                self.org_edge_index[1][idx]
                for idx, node in enumerate(self.org_edge_index[0])
                if node == node_idx
            ]

            for obj_idx in connected_obj_idx:
                graph['edge_index'][0].append(node_idx)
                graph['edge_index'][1].append(len(graph['node_feats']))
                graph['node_feats'].append(self.org_node_feats[obj_idx])

        return graph, subgoal_idx

    def prepare_gcn_input(self, curr_graph, subgoals):
        # add the target node and connect it to all the subgoal nodes
        # with edges
        graph = curr_graph.copy()

        for subgoal in subgoals:
            graph['edge_index'][0].append(subgoal)
            graph['edge_index'][1].append(len(graph['node_feats']))
        graph['node_feats'].append(self.org_node_feats[self.target_obj])
        is_subgoal = [0] * len(graph['node_feats'])
        is_target = [0] * len(graph['node_feats'])
        is_target[-1] = 1
        for subgoal_idx in subgoals:
            is_subgoal[subgoal_idx] = 1
        graph['is_subgoal'] = is_subgoal
        graph['is_target'] = is_target
        return graph

    def get_training_data(self):
        current_graph, subgoals = self.initialize_graph_and_subgoals()
        input_graph = self.prepare_gcn_input(current_graph, subgoals)

        label = [0] * len(input_graph['node_feats'])
        for target_container in self.target_container:
            label[target_container] = 1
        input_graph['labels'] = label

        return input_graph


class FState(object):
    """Used to conviently store the 'state' during recursive cost search.
    """
    def __init__(self, new_frontier, distances, old_state=None):
        nf = new_frontier
        p = nf.prob_feasible
        # Success cost
        try:
            sc = nf.delta_success_cost + distances['goal'][nf]
        except KeyError:
            sc = nf.delta_success_cost + distances['goal'][nf.id]
        # Exploration cost
        ec = nf.exploration_cost

        if old_state is not None:
            self.frontier_list = old_state.frontier_list + [nf]
            # Store the old frontier
            of = old_state.frontier_list[-1]
            # Known cost (travel between frontiers)
            try:
                kc = distances['frontier'][frozenset([nf, of])]
            except KeyError:
                kc = distances['frontier'][frozenset([nf.id, of.id])]
            self.cost = old_state.cost + old_state.prob * (kc + p * sc +
                                                           (1 - p) * ec)
            self.prob = old_state.prob * (1 - p)
        else:
            # This is the first frontier, so the robot must accumulate a cost of getting to the frontier
            self.frontier_list = [nf]
            # Known cost (travel to frontier)
            try:
                kc = distances['robot'][nf]
            except KeyError:
                kc = distances['robot'][nf.id]

            if nf.is_from_last_chosen:
                kc -= IS_FROM_LAST_CHOSEN_REWARD
            self.cost = kc + p * sc + (1 - p) * ec
            self.prob = (1 - p)

    def __lt__(self, other):
        return self.cost < other.cost


def get_ordering_cost(subgoals, distances):
    """A helper function to compute the expected cost of a particular ordering.
    The function takes an ordered list of subgoals (the order in which the robot
    aims to explore beyond them). Consistent with the subgoal planning API,
    'distances' is a dictionary with three keys: 'robot' (a dict of the
    robot-subgoal distances), 'goal' (a dict of the goal-subgoal distances), and
    'frontier' (a dict of the frontier-frontier distances)."""
    fstate = None
    for s in subgoals:
        fstate = FState(s, distances, fstate)

    return fstate.cost


def get_lowest_cost_ordering(subgoals, distances, do_sort=True):
    """This computes the lowest cost ordering (the policy) the robot will follow
    for navigation under uncertainty. It wraps a branch-and-bound search
    function implemented in C++ in 'lsp_accel'. As is typical of
    branch-and-bound functions, function evaluation is fastest if the high-cost
    plans can be ruled out quickly: i.e., if the first expansion is already of
    relatively low cost, many of the other branches can be pruned. When
    'do_sort' is True, a handful of common-sense heuristics are run to find an
    initial ordering that is of low cost to take advantage of this property. The
    subgoals are sorted by the various heuristics and the ordering that
    minimizes the expected cost is chosen. That ordering is used as an input to
    the search function, which searches it first."""

    if len(subgoals) == 0:
        return None, None

    if do_sort:
        order_heuristics = []
        order_heuristics.append({
            s: ii for ii, s in enumerate(subgoals)
        })
        order_heuristics.append({
            s: 1 - s.prob_feasible for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.prob_feasible * s.delta_success_cost +
            (1 - s.prob_feasible) * s.exploration_cost
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s]
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.delta_success_cost
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.exploration_cost
            for s in subgoals
        })

        heuristic_ordering_dat = []
        for heuristic in order_heuristics:
            ordered_subgoals = sorted(subgoals, reverse=False, key=lambda s: heuristic[s])
            ordering_cost = get_ordering_cost(ordered_subgoals, distances)
            heuristic_ordering_dat.append((ordering_cost, ordered_subgoals))

        subgoals = min(heuristic_ordering_dat, key=lambda hod: hod[0])[1]

    s_dict = {hash(s): s for s in subgoals}
    rd_cpp = {hash(s): distances['robot'][s] for s in subgoals}
    gd_cpp = {hash(s): distances['goal'][s] for s in subgoals}
    fd_cpp = {(hash(sp[0]), hash(sp[1])): distances['frontier'][frozenset(sp)]
              for sp in itertools.permutations(subgoals, 2)}
    s_cpp = [
        lsp_accel.FrontierData(s.prob_feasible, s.delta_success_cost,
                               s.exploration_cost, hash(s),
                               s.is_from_last_chosen) for s in subgoals
    ]

    cost, ordering = lsp_accel.get_lowest_cost_ordering(
        s_cpp, rd_cpp, gd_cpp, fd_cpp)
    ordering = [s_dict[sid] for sid in ordering]

    return cost, ordering


def get_top_n_frontiers(frontiers, goal_dist, robot_dist, n):
    """This heuristic is for retrieving the 'best' N frontiers"""

    # This sorts the frontiers by (1) any frontiers that "derive their
    # properties" from the last chosen frontier and (2) the probablity that the
    # frontiers lead to the goal.
    frontiers = [f for f in frontiers if f.prob_feasible > 0]

    h_prob = {s: s.prob_feasible for s in frontiers}
    try:
        h_dist = {s: goal_dist[s] + robot_dist[s] for s in frontiers}
    except KeyError:
        h_dist = {s: goal_dist[s.id] + robot_dist[s.id] for s in frontiers}

    fs_prob = sorted(list(frontiers), key=lambda s: h_prob[s], reverse=True)
    fs_dist = sorted(list(frontiers), key=lambda s: h_dist[s], reverse=False)

    seen = set()
    fs_collated = []

    for front_d in fs_dist[:2]:
        if front_d not in seen:
            seen.add(front_d)
            fs_collated.append(front_d)

    for front_p in fs_prob:
        if front_p not in seen:
            seen.add(front_p)
            fs_collated.append(front_p)

    assert len(fs_collated) == len(seen)
    assert len(fs_collated) == len(fs_prob)
    assert len(fs_collated) == len(fs_dist)

    return fs_collated[0:n]


def get_best_expected_cost_and_frontier_list(
        subgoals, partial_map, robot_pose, destination, num_frontiers_max):

    # Get robot distances
    robot_distances = get_robot_distances(
        partial_map.grid, robot_pose, subgoals)

    # Get goal distances
    if destination is None:
        goal_distances = {subgoal: robot_distances[subgoal]
                          for subgoal in subgoals}
    else:
        goal_distances = get_robot_distances(
            partial_map.grid, destination, subgoals)

    # Calculate top n subgoals
    subgoals = get_top_n_frontiers(
        subgoals, goal_distances, robot_distances, num_frontiers_max)

    # Get subgoal pair distances
    subgoal_distances = get_subgoal_distances(partial_map.grid, subgoals)

    distances = {
        'frontier': subgoal_distances,
        'robot': robot_distances,
        'goal': goal_distances,
    }

    out = get_lowest_cost_ordering(subgoals, distances)
    return out


def get_robot_distances(grid, robot_pose, subgoals):
    ''' This function returns distance from the robot to the subgoals
    where poses are stored in grid cell coordinates.'''
    robot_distances = dict()

    occ_grid = np.copy(grid)
    occ_grid[int(robot_pose[0])][int(robot_pose[1])] = 0

    for subgoal in subgoals:
        occ_grid[int(subgoal.pos[0]), int(subgoal.pos[1])] = 0

    cost_grid = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[
            robot_pose[0],
            robot_pose[1]
        ],
        use_soft_cost=True,
        only_return_cost_grid=True)

    # Compute the cost for each frontier
    for subgoal in subgoals:
        f_pt = subgoal.pos
        cost = cost_grid[int(f_pt[0]), int(f_pt[1])]

        if math.isinf(cost):
            cost = 100000000000
            subgoal.set_props(prob_feasible=0.0, is_obstructed=True)
            subgoal.just_set = False

        robot_distances[subgoal] = cost

    return robot_distances


def get_subgoal_distances(grid, subgoals):
    ''' This function returns distance from any subgoal to other subgoals
    where poses are stored in grid cell coordinates.'''
    subgoal_distances = {}
    occ_grid = np.copy(grid)
    for subgoal in subgoals:
        occ_grid[int(subgoal.pos[0]), int(subgoal.pos[1])] = 0
    for idx, sg_1 in enumerate(subgoals[:-1]):
        start = sg_1.pos
        cost_grid = gridmap.planning.compute_cost_grid_from_position(
            occ_grid,
            start=start,
            use_soft_cost=True,
            only_return_cost_grid=True)
        for sg_2 in subgoals[idx + 1:]:
            fsg_set = frozenset([sg_1, sg_2])
            fpoints = sg_2.pos
            cost = cost_grid[int(fpoints[0]), int(fpoints[1])]
            subgoal_distances[fsg_set] = cost

    return subgoal_distances


def compute_path_cost(grid, path):
    ''' This function returns the total path and path cost
    given the occupancy grid and the trjectory as poses, the
    robot has visited througout the object search process,
    where poses are stored in grid cell coordinates.'''
    total_cost = 0
    total_path = None
    occ_grid = np.copy(grid)

    for point in path:
        occ_grid[int(point[0]), int(point[1])] = 0

    for idx, point in enumerate(path[:-1]):
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            occ_grid,
            start=point,
            use_soft_cost=True,
            only_return_cost_grid=False)
        next_point = path[idx + 1]

        cost = cost_grid[int(next_point[0]), int(next_point[1])]

        total_cost += cost
        did_plan, robot_path = get_path([next_point[0], next_point[1]],
                                        do_sparsify=False,
                                        do_flip=False)
        if total_path is None:
            total_path = robot_path
        else:
            total_path = np.concatenate((total_path, robot_path), axis=1)

    return total_cost, total_path
