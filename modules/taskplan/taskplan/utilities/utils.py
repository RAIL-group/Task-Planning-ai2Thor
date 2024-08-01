import os
import glob
import torch
import random
import numpy as np
from torch_geometric.data import Data

import learning


def write_datum_to_file(args, datum, counter):
    data_filename = os.path.join('pickles', f'dat_{args.current_seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(args.save_dir, data_filename), datum)
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.csv'
    with open(os.path.join(args.save_dir, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')


def get_data_path_names(args):
    training_data_files = glob.glob(os.path.join(args.data_csv_dir, "*train*.csv"))
    testing_data_files = glob.glob(os.path.join(args.data_csv_dir, "*test*.csv"))
    return training_data_files, testing_data_files


def preprocess_training_data(args=None):
    def make_graph(data):
        data['node_feats'] = torch.tensor(
            np.array(data['node_feats']), dtype=torch.float)
        temp = [[x[0], x[1]] for x in data['edge_index'] if x[0] != x[1]]
        data['edge_index'] = torch.tensor(list(zip(*temp)), dtype=torch.long)
        data['is_subgoal'] = torch.tensor(data['is_subgoal'], dtype=torch.long)
        data['is_target'] = torch.tensor(data['is_target'], dtype=torch.long)
        data['labels'] = torch.tensor(data['labels'], dtype=torch.float)

        tg_GCN_format = Data(x=data['node_feats'],
                             edge_index=data['edge_index'],
                             is_subgoal=data['is_subgoal'],
                             is_target=data['is_target'],
                             # edge_features=data['edge_features'],
                             y=data['labels'])

        result = tg_GCN_format
        return result
    return make_graph


def preprocess_gcn_data(datum):
    data = datum.copy()
    data['edge_data'] = torch.tensor(data['edge_index'], dtype=torch.long)
    data['latent_features'] = torch.tensor(np.array(
        data['node_feats']), dtype=torch.float)
    data['is_subgoal'] = torch.tensor(data['is_subgoal'], dtype=torch.long)
    data['is_target'] = torch.tensor(data['is_target'], dtype=torch.long)
    return data


def get_pose_from_coord(coords, whole_graph):
    coords_list = []
    for node in whole_graph['node_coords']:
        coords_list.append(tuple(
            [whole_graph['node_coords'][node][0],
             whole_graph['node_coords'][node][1]]))
    if coords in coords_list:
        pos = coords_list.index(coords)
        return pos
    return None


def initialize_environment(cnt_node_idx, seed=0):
    random.seed(seed)
    sample_count = random.randint(1, len(cnt_node_idx))
    undiscovered_cnts = random.sample(cnt_node_idx, sample_count)
    srtd_und_cnts = sorted(undiscovered_cnts)
    return srtd_und_cnts


def get_container_ID(nodes, cnts):
    set_of_ID = set()
    for cnt in cnts:
        set_of_ID.add(nodes[cnt]['id'])
    return set_of_ID


def get_container_pose(cnt_name, partial_map):
    '''This function takes in a container name and the
    partial map as input to return the container pose on the grid'''
    if cnt_name in partial_map.idx_map:
        return partial_map.container_poses[partial_map.idx_map[cnt_name]]
    raise ValueError('The container could not be located on the grid!')


def get_object_to_find_from_plan(plan, partial_map, init_robot_pose):
    '''This function takes in a plan and the partial map as
    input to return the object index to find; limited to finding
    single object for now.'''
    find_from = {}
    find_at = []
    robot_poses = [init_robot_pose]
    for action in plan:
        if action.name == 'move':
            container_name = action.args[1]
            container_pose = get_container_pose(container_name, partial_map)
            robot_poses.append(container_pose)
        else:
            robot_poses.append(robot_poses[-1])
        if action.name == 'find':
            obj_name = action.args[0]
            if obj_name in partial_map.idx_map:
                obj_idx = partial_map.idx_map[obj_name]
                find_from[obj_idx] = robot_poses[-1]
            find_at.append(len(robot_poses))
            # raise ValueError('The object could not be found!')
    return find_from, find_at, robot_poses


def get_pose_action_log(robot_poses):
    pose_action_log = {}
    for idx, p in enumerate(robot_poses):
        if p in pose_action_log:
            pose_action_log[p].append(idx)
        else:
            pose_action_log[p] = [idx]
    return pose_action_log
