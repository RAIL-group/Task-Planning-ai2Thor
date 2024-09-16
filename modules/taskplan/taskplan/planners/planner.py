import copy
import torch

import taskplan
from taskplan.core import Subgoal
from taskplan.learning.models.gnn import Gnn


NUM_MAX_FRONTIERS = 8


class Planner():
    def __init__(self, args, partial_map=None, device=None, verbose=True):
        self.args = args
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.verbose = verbose
        self.partial_map = partial_map

    def update(self, graph, subgoals, robot_pose):
        self.graph = graph
        self.robot_pose = robot_pose
        self.new_subgoals = [s for s in subgoals]
        # Convert into `Subgoal class' in an attempt to match cost calculation
        # API's input format
        self.subgoals = [copy.copy(Subgoal(value)) for value in self.new_subgoals]
        for subgoal in self.subgoals:
            subgoal.pos = self.partial_map.container_poses[subgoal]
        self._update_subgoal_properties()

    def _update_subgoal_properties():
        raise NotImplementedError


class KnownPlanner(Planner):
    ''' This planner has access to the target object location and can
    find the object in 1 step
    '''
    def __init__(self, args, partial_map, device=None, verbose=False,
                 destination=None):
        super(KnownPlanner, self).__init__(
            args, partial_map, device, verbose)
        self.destination = destination

    def _update_subgoal_properties(self):
        for subgoal in self.subgoals:
            if subgoal.value in self.partial_map.target_container:
                subgoal.set_props(prob_feasible=1.0)
            else:
                subgoal.set_props(prob_feasible=0.0)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.partial_map.org_node_names[subgoal.value]}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.partial_map,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS))
        return frontier_ordering[0]


class ClosestActionPlanner(Planner):
    ''' This planner naively looks in the nearest container to find any object
    '''
    def __init__(self, args, partial_map, device=None, verbose=False,
                 destination=None):
        super(ClosestActionPlanner, self).__init__(
            args, partial_map, device, verbose)
        self.destination = destination

    def _update_subgoal_properties(self):
        for subgoal in self.subgoals:
            subgoal.set_props(prob_feasible=1.0)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.partial_map.org_node_names[subgoal.value]}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.partial_map,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS))
        return frontier_ordering[0]


class LearnedPlanner(Planner):
    ''' This planner calculates subgoal properties using the learned network
    and then uses LSP approach to pick the best available action (subgoal).
    '''
    def __init__(self, args, partial_map, device=None, verbose=True,
                 destination=None):
        super(LearnedPlanner, self).__init__(
            args, partial_map, device, verbose)
        self.destination = destination
        self.subgoal_property_net = Gnn.get_net_eval_fn(
            args.network_file, device=self.device)

    def _update_subgoal_properties(self):
        self.gcn_graph_input = self.partial_map.prepare_gcn_input(
            curr_graph=self.graph,
            subgoals=self.new_subgoals
        )
        prob_feasible_dict = self.subgoal_property_net(
            datum=self.gcn_graph_input,
            subgoals=self.subgoals
        )
        for subgoal in self.subgoals:
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal])
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.partial_map.org_node_names[subgoal.value]}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self, return_cost=False):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.partial_map,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS))
        if return_cost:
            return min_cost, frontier_ordering[0]
        return frontier_ordering[0]
