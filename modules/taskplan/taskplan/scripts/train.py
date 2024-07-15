import os
import torch
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

import taskplan
from learning.data import CSVPickleDataset
from taskplan.learning.models.gnn import Gnn


def get_model_prep_fn_and_training_strs(args):
    print("Training LSPGNN for Task Plan... ...")
    model = Gnn(args)
    prep_fn = taskplan.utilities.utils.preprocess_training_data(args)
    train_writer_str = 'train_gnn'
    test_writer_str = 'test_gnn'
    lr_writer_str = 'learning_rate/gnn'
    model_name_str = 'gnn.pt'

    return {
        'model': model,
        'prep_fn': prep_fn,
        'train_writer_str': train_writer_str,
        'test_writer_str': test_writer_str,
        'lr_writer_str': lr_writer_str,
        'model_name_str': model_name_str
    }


def train(args, train_path, test_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get the model and other training info
    model_and_training_info = get_model_prep_fn_and_training_strs(args)
    model = model_and_training_info['model']
    prep_fn = model_and_training_info['prep_fn']
    train_writer_str = model_and_training_info['train_writer_str']
    test_writer_str = model_and_training_info['test_writer_str']
    lr_writer_str = model_and_training_info['lr_writer_str']
    model_name_str = model_and_training_info['model_name_str']

    # Create the datasets and loaders
    train_dataset = CSVPickleDataset(train_path, prep_fn)
    print("Number of training graphs:", len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, train_writer_str))

    test_dataset = CSVPickleDataset(test_path, prep_fn)
    print("Number of testing graphs:", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    test_iter = iter(test_loader)
    test_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, test_writer_str))

    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1,
        gamma=args.learning_rate_decay_factor)

    tot_index = 0
    for epoch in range(args.num_epochs):
        for index, train_batch in enumerate(train_loader):
            out = model.forward({
                'edge_data': train_batch.edge_index,
                'is_subgoal': train_batch.is_subgoal,
                'is_target': train_batch.is_target,
                'latent_features': train_batch.x
            }, device)

            loss = model.loss(out,
                              train_batch,
                              device=device,
                              writer=train_writer,
                              index=tot_index)

            if index % args.train_log_frequency == 0:
                print(f"Train Loss({epoch}.{index}, {tot_index}): {loss}")

            if index % args.test_log_frequency == 0:
                with torch.no_grad():
                    try:
                        test_batch = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_batch = next(test_iter)

                    tout = model.forward({
                        'edge_data': test_batch.edge_index,
                        'is_subgoal': test_batch.is_subgoal,
                        'is_target': test_batch.is_target,
                        'latent_features': test_batch.x
                    }, device)
                    tloss = model.loss(tout,
                                       data=test_batch,
                                       device=device,
                                       writer=test_writer,
                                       index=tot_index)
                    print(f"Test Loss({epoch}.{index}, {tot_index}): {tloss}")

            # Perform update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_index += 1

            # Log the learning rate
            test_writer.add_scalar(lr_writer_str,
                                   scheduler.get_last_lr()[-1],
                                   tot_index)
        # Step the learning rate scheduler
        scheduler.step()

    # Saving the model after training
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, model_name_str))


def get_parser():
    # Add new arguments
    parser = argparse.ArgumentParser(
        description="Train LSPGNN net with PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Logging
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--data_csv_dir', type=str)
    parser.add_argument(
        '--train_log_frequency',
        default=10,
        help='Frequency (in steps) train logs printed to the terminal',
        type=int)
    parser.add_argument(
        '--test_log_frequency',
        default=100,
        help='Frequency (in steps) test logs printed to the terminal',
        type=int)

    # Training
    parser.add_argument('--num_epochs',
                        default=8,
                        help='Number of epochs to run training',
                        type=int)
    parser.add_argument('--learning_rate',
                        default=0.001,
                        help='Initial learning rate',
                        type=float)
    parser.add_argument('--learning_rate_decay_factor',
                        default=0.6,
                        help='How much learning rate decreases between epochs.',
                        type=float)
    parser.add_argument('--batch_size',
                        default=32,
                        help='Number of data per training iteration batch',
                        type=int)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Always freeze your random seeds
    torch.manual_seed(8616)
    random.seed(8616)
    train_path, test_path = taskplan.utilities.utils.get_data_path_names(args)
    # Train the neural network
    train(args=args, train_path=train_path, test_path=test_path)
