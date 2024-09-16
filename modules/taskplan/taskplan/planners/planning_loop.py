import time


class PlanningLoop():
    def __init__(self, partial_map, robot, destination, args,
                 verbose=False, close_loop=False):
        self.partial_map = partial_map
        self.graph, self.subgoals = self.partial_map. \
            initialize_graph_and_subgoals(args.current_seed)
        self.goal = partial_map.target_container
        self.robot = []
        self.robot.append(robot)
        self.destination = destination
        self.args = args
        self.did_succeed = True
        self.verbose = verbose
        self.chosen_subgoal = None
        self.close_loop = close_loop

    def __iter__(self):
        counter = 0
        count_since_last_turnaround = 100
        fn_start_time = time.time()

        # Main planning loop
        while (self.chosen_subgoal not in self.goal):

            if self.verbose:
                print(f"Need (WHAT): {self.partial_map.org_node_names[self.partial_map.target_obj]}")
                goals = [self.partial_map.org_node_names[goal] for goal in self.goal]
                print(f"From (WHERE): {goals}")

                print(f"Counter: {counter} | Count since last turnaround: "
                      f"{count_since_last_turnaround}")

            self.graph, self.subgoals = self.partial_map. \
                update_graph_and_subgoals(subgoals=self.subgoals,
                                          chosen_subgoal=self.chosen_subgoal)

            yield {
                'graph': self.graph,
                'subgoals': self.subgoals,
                'robot_pose': self.robot[-1]
            }

            # update robot_pose with current action pose for next iteration of action
            self.robot.append(self.chosen_subgoal.pos)

            counter += 1
            count_since_last_turnaround += 1
            if self.verbose:
                print("")

        # add initial robot pose at the end to close the search loop
        if self.close_loop:
            self.robot.append(self.robot[0])
        elif self.robot[-1] != self.destination:
            self.robot.append(self.destination)

        if self.verbose:
            print("TOTAL TIME:", time.time() - fn_start_time)

    def set_chosen_subgoal(self, new_chosen_subgoal):
        self.chosen_subgoal = new_chosen_subgoal
        print(f"Finding in (WHERE): {self.partial_map.org_node_names[self.chosen_subgoal]}")
