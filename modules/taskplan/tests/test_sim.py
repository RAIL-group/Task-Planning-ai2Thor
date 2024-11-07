import os
import numpy as np
import matplotlib.pyplot as plt

import taskplan
import taskplan.plotting


def get_args():
    create_dir()
    args = lambda key: None
    args.current_seed = 0
    args.resolution = 0.05
    return args


def create_dir():
    main_path = '/data'
    sub_dir = 'test_logs'
    sub_path = os.path.join(main_path, sub_dir)
    if not os.path.exists(sub_path):
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        os.makedirs(sub_path)


def test_reachable_grid():
    ''' This test plots occupancy grid and the original top-view image of the same procTHOR map.
    '''
    args = get_args()
    save_file = '/data/test_logs/grid-scene-' \
        + str(args.current_seed) + '.png'

    # # Get data for a send and extract initial object states
    thor_data = taskplan.utilities.ai2thor_helper. \
        ThorInterface(args=args, preprocess=True)

    # Get the occupancy grid from thor_data
    grid = thor_data.occupancy_grid.copy()
    plt.subplot(121)
    img = np.transpose(grid)
    plt.imshow(img)

    top_down_frame = thor_data.get_top_down_frame()

    plt.subplot(122)
    plt.imshow(top_down_frame)
    plt.savefig(save_file, dpi=1200)

    assert 0 == 0


def test_graph_on_grid():
    ''' This test plots occupancy grid and the original top-view image of the same procTHOR map.
    '''
    args = get_args()
    save_file = '/data/test_logs/graph-on-grid-' \
        + str(args.current_seed) + '.png'

    # Get thor data for a send and extract initial object states
    thor_data = taskplan.utilities.ai2thor_helper. \
        ThorInterface(args=args)
    # Get the occupancy grid from proc_data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from ProcTHOR data
    whole_graph = thor_data.get_graph()

    plt.subplot(121)
    taskplan.plotting.plot_graph_on_grid(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)

    # plot top-down view from simulator
    plt.subplot(122)
    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)

    plt.savefig(save_file, dpi=1200)

    assert 0 == 0


def test_plot_trajectory():
    args = get_args()
    args.current_seed = 7004
    save_file = '/data/test_logs/trajectory-' \
        + str(args.current_seed) + '.png'

    # Get thor data for a send and extract initial object states
    thor_data = taskplan.utilities.ai2thor_helper. \
        ThorInterface(args=args)
    # Get the occupancy grid from proc_data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from ProcTHOR data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)

    # print(whole_graph['node_names'])
    # print(whole_graph['node_coords'])

    # # unaudited trajectory
    # cost_str = 'Requires auditing'
    # robot_poses = [init_robot_pose, (106, 101), (182, 112), (196, 144), (176, 184)]
    # # dining table (106, 101)
    # # bed (182, 112)
    # # arm-chair (196, 144)
    # # fridge (176, 184)

    # audited trajectory
    cost_str = 'With auditing'
    robot_poses = [init_robot_pose, (176, 184)]

    distance, trajectory = taskplan.core.compute_path_cost(partial_map.grid, robot_poses)

    # 1 plot the grid with trajectory viridis color
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plotting_grid = taskplan.plotting.make_blank_grid(
        np.transpose(grid)
    )
    plt.imshow(plotting_grid)
    plt.title(f"{cost_str} Cost: {distance:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    viridis_cmap = plt.get_cmap('viridis')

    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)

    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    plt.subplot(122)
    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)

    # Hide box and ticks
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{save_file}', dpi=1200)
