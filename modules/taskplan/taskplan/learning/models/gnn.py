import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
# from torch_geometric.nn import GCNConv, GATv2Conv

import taskplan


class Gnn(nn.Module):
    name = 'GNNforTaskPlan'

    def __init__(self, args=None):
        super(Gnn, self).__init__()
        torch.manual_seed(8616)
        self._args = args

        self.fc1 = nn.Linear(772 + 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8)
        self.conv1 = SAGEConv(8, 8)
        self.conv2 = SAGEConv(8, 8)
        self.conv3 = SAGEConv(8, 8)
        self.conv4 = SAGEConv(8, 4)
        self.classifier = nn.Linear(4, 1)

        self.fc1bn = nn.BatchNorm1d(512)
        self.fc2bn = nn.BatchNorm1d(256)
        self.fc3bn = nn.BatchNorm1d(128)
        self.fc4bn = nn.BatchNorm1d(64)
        self.fc5bn = nn.BatchNorm1d(32)
        self.fc6bn = nn.BatchNorm1d(16)
        self.fc7bn = nn.BatchNorm1d(8)
        self.conv1bn = nn.BatchNorm1d(8)
        self.conv2bn = nn.BatchNorm1d(8)
        self.conv3bn = nn.BatchNorm1d(8)
        self.conv4bn = nn.BatchNorm1d(4)

        # Following class weighting factors have
        # been calculated after observing the data
        self.pos = 1
        self.neg = 1

    def forward(self, data, device):
        lf = data['latent_features'].type(torch.float).to(device)
        edge_data = data['edge_data']
        x = torch.cat((edge_data[0], edge_data[1]), 0)
        y = torch.cat((edge_data[1], edge_data[0]), 0)
        edge_data = torch.reshape(torch.cat((x, y), 0), (2, -1))
        edge_index = edge_data.to(device)
        # edge_features = data['edge_features'].type(torch.float).to(device)
        # edge_features = edge_features.repeat(2, 1).to(device) / 200

        is_subgoal = data['is_subgoal'].view(-1, 1).to(device)
        is_target = data['is_target'].view(-1, 1).to(device)
        h = torch.cat((lf, is_subgoal), 1)
        h = torch.cat((h, is_target), 1)

        h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        h = F.leaky_relu(self.fc4bn(self.fc4(h)), 0.1)
        h = F.leaky_relu(self.fc5bn(self.fc5(h)), 0.1)
        h = F.leaky_relu(self.fc6bn(self.fc6(h)), 0.1)
        h = F.leaky_relu(self.fc7bn(self.fc7(h)), 0.1)
        h = F.leaky_relu(self.conv1bn(self.conv1(h, edge_index)), 0.1)
        h = F.leaky_relu(self.conv2bn(self.conv2(h, edge_index)), 0.1)
        h = F.leaky_relu(self.conv3bn(self.conv3(h, edge_index)), 0.1)
        h = F.leaky_relu(self.conv4bn(self.conv4(h, edge_index)), 0.1)
        props = self.classifier(h)
        return props

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):
        # Separate outputs.
        is_feasible_logits = nn_out[:, 0]
        # Extract & load the data to device
        is_subgoal = data.is_subgoal.to(device)
        is_feasible_label = data.y.to(device)

        # Compute the contribution from the is_feasible_label
        is_feasible_xentropy = self.pos * \
            is_feasible_label * -F.logsigmoid(is_feasible_logits) + \
            (1 - is_feasible_label) * -F.logsigmoid(-is_feasible_logits)
        is_feasible_xentropy = torch.sum(is_subgoal * is_feasible_xentropy)
        is_feasible_xentropy /= torch.sum(is_subgoal) + 0.000001

        # Sum the contributions
        loss = is_feasible_xentropy

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/is_feasible_xentropy",
                              is_feasible_xentropy.item(),
                              index)
            writer.add_scalar("Loss/total_loss",
                              loss.item(),
                              index)

        return loss

    @classmethod
    def get_net_eval_fn(_, network_file,
                        device=None):
        model = Gnn()
        model.load_state_dict(torch.load(network_file,
                                         map_location=device))
        model.eval()
        model.to(device)

        def frontier_net(datum, subgoals):
            graph = taskplan.utilities.utils.preprocess_gcn_data(datum)
            prob_feasible_dict = {}
            with torch.no_grad():
                out = model.forward(graph, device)
                out[:, 0] = torch.sigmoid(out[:, 0])
                out = out.detach().cpu().numpy()
                for subgoal in subgoals:
                    # Extract subgoal properties for a subgoal
                    subgoal_props = out[subgoal.value]
                    prob_feasible_dict[subgoal] = subgoal_props[0]
                return prob_feasible_dict
        return frontier_net
