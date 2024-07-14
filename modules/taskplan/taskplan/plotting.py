import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion


COLLISION_VAL = 1
FREE_VAL = 0
UNOBSERVED_VAL = -1
assert (COLLISION_VAL > FREE_VAL)
assert (FREE_VAL > UNOBSERVED_VAL)
OBSTACLE_THRESHOLD = 0.5 * (FREE_VAL + COLLISION_VAL)


FOOT_PRINT = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])


def make_plotting_grid(grid_map):
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= OBSTACLE_THRESHOLD
    # Take one pixel boundary of the region collision
    thinned = erosion(collision, footprint=FOOT_PRINT)
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < OBSTACLE_THRESHOLD, grid_map >= FREE_VAL)
    grid[:, :, 0][free] = 1
    grid[:, :, 1][free] = 1
    grid[:, :, 2][free] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid


def plot_graph_on_grid(grid, graph):
    '''Plot the scene graph on the occupancy grid to scale'''
    plotting_grid = make_plotting_grid(np.transpose(grid))
    plt.imshow(plotting_grid)

    # find the room nodes
    room_node_idx = [idx for idx in range(1, graph['cnt_node_idx'][0])]

    rc_idx = room_node_idx + graph['cnt_node_idx']

    # plot the edge connectivity between rooms and their containers only
    filtered_edges = [
        edge
        for edge in graph['edge_index']
        if edge[1] in rc_idx and edge[0] != 0
    ]

    for (start, end) in filtered_edges:
        p1 = graph['nodes'][start]['pos']
        p2 = graph['nodes'][end]['pos']
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

    # plot room nodes
    for room in rc_idx:
        room_pos = graph['nodes'][room]['pos']
        room_name = graph['nodes'][room]['name']
        plt.text(room_pos[0], room_pos[1], room_name, color='brown', size=6, rotation=40)
