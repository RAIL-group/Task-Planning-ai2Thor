import json
import copy
import numpy as np
from PIL import Image
from shapely import geometry
from ai2thor.controller import Controller


IGNORE_CONTAINERS = [
    'baseballbat', 'basketBall', 'boots', 'desklamp', 'painting'
    'floorlamp', 'houseplant', 'roomdecor', 'showercurtain',
    'showerhead', 'television', 'vacuumcleaner', 'photo',
]


class ThorInterface:
    def __init__(self, args, preprocess=True):
        self.args = args
        self.seed = args.current_seed
        self.grid_resolution = args.resolution

        self.scene = self.load_scene()
        self.controller = Controller(scene=self.scene,
                                     gridSize=self.grid_resolution,
                                     width=480, height=480)

        self.rooms = self.scene['rooms']
        self.agent = self.scene['metadata']['agent']

        self.containers = self.scene['objects']
        if preprocess:
            self.containers = [
                container for container in self.containers
                if container['id'].split('|')[0].lower() not in IGNORE_CONTAINERS
            ]

        self.occupancy_grid = self.get_occupancy_grid()

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

    def get_occupancy_grid(self):
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
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
        grid_to_points_map = {self.scale_to_grid(point): point for point in points}
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

                # next do the same if there is any children of this container
                if 'children' in container:
                    children = container['children']
                    for child in children:
                        child['position'] = container['position']

        for room in self.rooms:
            floor = [(rp["x"], rp["z"]) for rp in room["floorPolygon"]]
            room_poly = geometry.Polygon(floor)
            point = room_poly.centroid
            scaled_position = self.scale_to_grid(np.array([point.x, point.y]))  # noqa: E501
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
