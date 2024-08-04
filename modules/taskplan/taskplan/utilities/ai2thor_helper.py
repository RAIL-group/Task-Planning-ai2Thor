import io
import os
import json
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from shapely import geometry
from ai2thor.controller import Controller
from sentence_transformers import SentenceTransformer

import gridmap

IGNORE_CONTAINERS = [
    'baseballbat', 'basketBall', 'boots', 'desklamp', 'painting',
    'floorlamp', 'houseplant', 'roomdecor', 'showercurtain',
    'showerhead', 'television', 'vacuumcleaner', 'photo', 'plunger',
    'basketball', 'box'
]


class ThorInterface:
    def __init__(self, args, preprocess=True):
        self.args = args
        self.seed = args.current_seed
        self.grid_resolution = args.resolution

        self.scene = self.load_scene()

        self.rooms = self.scene['rooms']
        self.agent = self.scene['metadata']['agent']

        self.containers = self.scene['objects']
        if preprocess:
            self.containers = [
                container for container in self.containers
                if container['id'].split('|')[0].lower() not in IGNORE_CONTAINERS
            ]

        self.controller = Controller(scene=self.scene,
                                     gridSize=self.grid_resolution,
                                     width=480, height=480)
        self.occupancy_grid = self.get_occupancy_grid()

        self.known_cost = self.get_known_costs()

    def load_scene(self, path='/data/procthor-data'):
        with open(
            f'{path}/data.jsonl',
            "r",
        ) as json_file:
            json_list = list(json_file)
        return json.loads(json_list[self.seed])

    def set_grid_offset(self, min_x, min_y):
        self.grid_offset = np.array([min_x, min_y])

    def scale_to_grid(self, point):
        x = round((point[0] - self.grid_offset[0]) / self.grid_resolution)
        y = round((point[1] - self.grid_offset[1]) / self.grid_resolution)
        return x, y

    def get_robot_pose(self):
        position = self.agent['position']
        position = np.array([position['x'], position['z']])
        return self.scale_to_grid(position)

    def get_occupancy_grid(self):
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
        RPs = reachable_positions

        xs = [rp["x"] for rp in reachable_positions]
        zs = [rp["z"] for rp in reachable_positions]

        # Calculate the mins and maxs
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        x_offset = min_x - self.grid_resolution if min_x < 0 else 0
        z_offset = min_z - self.grid_resolution if min_z < 0 else 0
        self.set_grid_offset(x_offset, z_offset)

        # Create list of free points
        points = list(zip(xs, zs))
        grid_to_points_map = {self.scale_to_grid(point): RPs[idx]
                              for idx, point in enumerate(points)}
        height, width = self.scale_to_grid([max_x, max_z])
        occupancy_grid = np.ones((height+2, width+2), dtype=int)
        free_positions = grid_to_points_map.keys()
        for pos in free_positions:
            occupancy_grid[pos] = 0

        # store the mapping from grid coordinates to simulator positions
        self.g2p_map = grid_to_points_map

        # set the nearest freespace container positions
        for container in self.containers:
            position = container['position']
            if position is not None:
                # get nearest free space pose
                nearest_fp = get_nearest_free_point(position, points)
                # then scale the free space pose to grid
                scaled_position = self.scale_to_grid(np.array([nearest_fp[0], nearest_fp[1]]))  # noqa: E501
                # finally set the scaled grid pose as the container position
                container['position'] = scaled_position  # 2d only
                container['id'] = container['id'].lower()  # 2d only

                # next do the same if there is any children of this container
                if 'children' in container:
                    children = container['children']
                    for child in children:
                        child['position'] = container['position']
                        child['id'] = child['id'].lower()

        for room in self.rooms:
            floor = [(rp["x"], rp["z"]) for rp in room["floorPolygon"]]
            room_poly = geometry.Polygon(floor)
            point = room_poly.centroid
            point = {'x': point.x, 'z': point.y}
            nearest_fp = get_nearest_free_point(point, points)
            scaled_position = self.scale_to_grid(np.array([nearest_fp[0], nearest_fp[1]]))  # noqa: E501
            room['position'] = scaled_position  # 2d only

        return occupancy_grid

    def get_top_down_frame(self):
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = Image.fromarray(event.third_party_camera_frames[-1])
        top_down_frame = top_down_frame.transpose(Image.FLIP_TOP_BOTTOM)
        return top_down_frame

    def get_graph(self):
        ''' This method creates graph data from the procthor-10k data'''
        # Create dummy apartment node
        node_count = 0
        nodes = {}
        assetId_idx_map = {}
        edges = []
        nodes[node_count] = {
            'id': 'Apartment|0',
            'name': 'apartment',
            'pos': (0, 0),
            'type': [1, 0, 0, 0]
        }
        node_count += 1

        # Iterate over rooms but skip position coordinate scaling since not
        # required in distance calculations
        for room in self.rooms:
            nodes[node_count] = {
                'id': room['id'],
                'name': room['roomType'].lower(),
                'pos': room['position'],
                'type': [0, 1, 0, 0]
            }
            edges.append(tuple([0, node_count]))
            node_count += 1

        # add an edge between two rooms adjacent by a passable shared door
        room_edges = set()
        for i in range(1, len(nodes)):
            for j in range(i+1, len(nodes)):
                node_1, node_2 = nodes[i], nodes[j]
                if has_edge(self.scene['doors'], node_1['id'], node_2['id']):
                    room_edges.add(tuple(sorted((i, j))))
        edges.extend(room_edges)

        node_keys = list(nodes.keys())
        node_ids = [get_room_id(nodes[key]['id']) for key in node_keys]
        cnt_node_idx = []

        for container in self.containers:
            cnt_id = get_room_id(container['id'])
            src = node_ids.index(cnt_id)
            assetId = container['id']
            assetId_idx_map[assetId] = node_count
            name = get_generic_name(container['id'])
            nodes[node_count] = {
                'id': container['id'],
                'name': name,
                'pos': container['position'],
                'type': [0, 0, 1, 0]
            }
            edges.append(tuple([src, node_count]))
            cnt_node_idx.append(node_count)
            node_count += 1

        node_keys = list(nodes.keys())
        node_ids = [nodes[key]['id'] for key in node_keys]
        obj_node_idx = []

        for container in self.containers:
            connected_objects = container.get('children')
            if connected_objects is not None:
                src = node_ids.index(container['id'])
                for object in connected_objects:
                    assetId = object['id']
                    assetId_idx_map[assetId] = node_count
                    name = get_generic_name(object['id'])
                    nodes[node_count] = {
                        'id': object['id'],
                        'name': name,
                        'pos': object['position'],
                        'type': [0, 0, 0, 1]
                    }
                    edges.append(tuple([src, node_count]))
                    obj_node_idx.append(node_count)
                    node_count += 1

        graph = {
            'nodes': nodes,  # dictionary {id, name, pos, type}
            'edge_index': edges,  # pairwise edge list
            'cnt_node_idx': cnt_node_idx,  # indices of contianers
            'obj_node_idx': obj_node_idx,  # indices of objects
            'idx_map': assetId_idx_map  # mapping from assedId to graph index position
        }

        # Add edges to get a connected graph if not already connected
        req_edges = get_edges_for_connected_graph(self.occupancy_grid, graph)
        graph['edge_index'] = req_edges + graph['edge_index']

        # perform some more formatting for the graph, then return
        node_coords = {}
        node_names = {}
        graph_nodes = []
        node_color_list = []

        for count, node_key in enumerate(graph['nodes']):
            node_coords[node_key] = graph['nodes'][node_key]['pos']
            node_names[node_key] = graph['nodes'][node_key]['name']
            node_feature = np.concatenate((
                get_sentence_embedding(graph['nodes'][node_key]['name']),
                graph['nodes'][node_key]['type']
            ))
            assert count == node_key
            graph_nodes.append(node_feature)
            node_color_list.append(get_object_color_from_type(
                graph['nodes'][node_key]['type']))

        graph['node_coords'] = node_coords
        graph['node_names'] = node_names
        graph['graph_nodes'] = graph_nodes  # node features
        src = []
        dst = []
        for edge in graph['edge_index']:
            src.append(edge[0])
            dst.append(edge[1])
        graph['graph_edge_index'] = [src, dst]

        graph['graph_image'] = get_graph_image(
            graph['edge_index'],
            node_names, node_color_list
        )

        return graph

    def get_known_costs(self):
        known_cost = {'initial_robot_pose': {}}
        init_r = self.get_robot_pose()

        # get cost from initial robot pose to all containers
        for container in self.containers:
            container_id = container['id']
            known_cost[container_id] = {}
            container_position = container['position']
            cost = get_cost(grid=self.occupancy_grid,
                            robot_pose=init_r,
                            end=container_position)
            known_cost['initial_robot_pose'][container_id] = round(cost, 4)

        # get cost from a container to every other container
        for index, container1 in enumerate(self.containers):
            cnt1_id = container1['id']
            cnt1_position = container1['position']
            for container2 in self.containers[index+1:]:
                cnt2_id = container2['id']
                cnt2_position = container2['position']
                cost = get_cost(grid=self.occupancy_grid,
                                robot_pose=cnt1_position,
                                end=cnt2_position)
                known_cost[cnt1_id][cnt2_id] = round(cost, 4)
                known_cost[cnt2_id][cnt1_id] = round(cost, 4)

        return known_cost


def get_nearest_free_point(point, free_points):
    _min = 1000000000
    tp = point
    fp_idx = 0
    for idx, rp in enumerate(free_points):
        dist = (rp[0]-tp['x'])**2 + (rp[1]-tp['z'])**2
        if dist < _min:
            _min = dist
            fp_idx = idx
    return free_points[fp_idx]


def has_edge(doors, room_0, room_1):
    for door in doors:
        if (door['room0'] == room_0 and door['room1'] == room_1) or \
           (door['room1'] == room_0 and door['room0'] == room_1):
            return True
    return False


def get_generic_name(name):
    return name.split('|')[0].lower()


def get_room_id(name):
    return int(name.split('|')[1])


def get_cost(grid, robot_pose, end):
    occ_grid = np.copy(grid)
    occ_grid[int(robot_pose[0])][int(robot_pose[1])] = 0

    occ_grid[end[0], end[1]] = 0

    cost_grid = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[
            robot_pose[0],
            robot_pose[1]
        ],
        use_soft_cost=True,
        only_return_cost_grid=True)

    cost = cost_grid[end[0], end[1]]
    return cost


def get_edges_for_connected_graph(grid, graph):
    """ This function finds edges that needs to exist to have a connected graph """
    edges_to_add = []

    # find the room nodes
    room_node_idx = [idx for idx in range(1, graph['cnt_node_idx'][0])]

    # extract the edges only for the rooms
    filtered_edges = [
        edge
        for edge in graph['edge_index']
        if edge[1] in room_node_idx and edge[0] != 0
    ]

    # Get a list (sorted by length) of disconnected components
    sorted_dc = get_dc_comps(room_node_idx, filtered_edges)

    length_of_dc = len(sorted_dc)
    while length_of_dc > 1:
        comps = sorted_dc[0]
        merged_set = set()
        min_cost = 9999
        min_index = -9999
        for s in sorted_dc[1:]:
            merged_set |= s
        for comp in comps:
            for idx, target in enumerate(merged_set):
                cost = get_cost(grid, graph['nodes'][comp]['pos'],
                                graph['nodes'][target]['pos'])
                if cost < min_cost:
                    min_cost = cost
                    min_index = list(merged_set)[idx]

        edge_to_add = (comp, min_index)
        edges_to_add.append(edge_to_add)
        filtered_edges = filtered_edges + [edge_to_add]
        sorted_dc = get_dc_comps(room_node_idx, filtered_edges)
        length_of_dc = len(sorted_dc)

    return edges_to_add


def get_dc_comps(room_idxs, edges):
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(room_idxs)
    G.add_edges_from(edges)

    # Find disconnected components
    disconnected_components = list(nx.connected_components(G))
    sorted_dc = sorted(disconnected_components, key=lambda x: len(x))

    return sorted_dc


def load_sentence_embedding(target_file_name):
    target_dir = '/data/sentence_transformers/cache/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Walk through all directories and files in target_dir
    for root, dirs, files in os.walk(target_dir):
        if target_file_name in files:
            file_path = os.path.join(root, target_file_name)
            if os.path.exists(file_path):
                return np.load(file_path)
    return None


def get_sentence_embedding(sentence):
    loaded_embedding = load_sentence_embedding(sentence + '.npy')
    if loaded_embedding is None:
        model_path = "/data/sentence_transformers/"
        model = SentenceTransformer(model_path)
        sentence_embedding = model.encode([sentence])[0]
        file_name = '/data/sentence_transformers/cache/' + sentence + '.npy'
        np.save(file_name, sentence_embedding)
        return sentence_embedding
    else:
        return loaded_embedding


def get_object_color_from_type(encoding):
    if encoding[0] == 1:
        return "red"
    if encoding[1] == 1:
        return "blue"
    if encoding[2] == 1:
        return "green"
    if encoding[3] == 1:
        return "orange"
    return "violet"


def get_graph_image(edge_index, node_names, color_map):
    # Create a graph object
    G = nx.Graph()

    # Add nodes to the graph with labels
    for idx, _ in enumerate(node_names):
        G.add_node(idx)

    # Add edges to the graph
    G.add_edges_from(edge_index)

    # Draw the graph
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=150,
            labels=node_names, font_size=8, font_weight='regular', edge_color='black')

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    img = Image.open(buf)
    return img
