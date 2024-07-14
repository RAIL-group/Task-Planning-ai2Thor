import os
import random
import numpy as np
import matplotlib.pyplot as plt

import taskplan


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
    random.seed(args.current_seed)
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
