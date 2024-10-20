import os
import glob
import torch
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
