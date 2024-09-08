import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def process_learned_data(args):
    """Preprocessing function for learned lsp"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . learned: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'LEARNED_LSP']
    )


def process_naive_data(args):
    """Preprocessing function for naive (closest action planner) lsp"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . naive: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'NAIVE_LSP']
    )


def process_learned_sp_data(args):
    """Preprocessing function for planner with Learned Search Policy"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . learned_sp: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'LEARNED_SP']
    )


def process_data(args):
    """Preprocessing function for all planner"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . naive: (.*?) . learned: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1]), float(d[2]), float(d[3])])

    return pd.DataFrame(
        data,
        columns=['seed', 'Naive Cost', 'Learned Cost']
    )


def gjs_scatter_plot(ax, cost_x, cost_y, max_val, fail_val):
    common_seeds = set.intersection(set(cost_x.keys()), set(cost_y.keys()))
    y_axis = [cost_y[seed] for seed in common_seeds]
    x_axis = [cost_x[seed] for seed in common_seeds]
    # Calculate the point density
    xy = np.vstack([x_axis, y_axis])
    z = gaussian_kde(xy)(xy)
    colors = plt.get_cmap("Blues")((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.50)

    ax.scatter(x_axis, y_axis, c=colors)

    # x axis and y axis should be the same
    # max_val = max(max(x_axis), max(y_axis))
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    # draw a center line
    ax.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=0.5, alpha=0.2)

    # Plot the failed seeds
    y_fail_seed = set.difference(set(cost_x.keys()), set(cost_y.keys()))
    y_fail_cost = [cost_x[seed] for seed in y_fail_seed]
    x_fail_fill_cost = [fail_val for _ in y_fail_seed]
    if y_fail_cost:
        print(y_fail_cost)
        ax.plot([fail_val, fail_val], [min(y_fail_cost)-30, max(y_fail_cost)+30], color='black', linestyle='--', linewidth=0.5, alpha=0.2)
        ax.scatter(x_fail_fill_cost, y_fail_cost, color='red', marker='x')

    x_fail_seed = set.difference(set(cost_y.keys()), set(cost_x.keys()))
    x_fail_cost = [cost_y[seed] for seed in x_fail_seed]
    y_fail_fill_cost = [fail_val for _ in x_fail_seed]
    if x_fail_cost:
        print(x_fail_cost)
        ax.plot([min(x_fail_cost)-30, max(x_fail_cost)+30], [fail_val, fail_val], color='black', linestyle='--', linewidth=0.5, alpha=0.2)
        ax.scatter(x_fail_cost, y_fail_fill_cost, color='red', marker='x')


def make_scatter_with_box(data_x, data_y, max_val=None):
    if max_val is None:
        max_val = 1.1 * max(max(data_x.values()), max(data_y.values()))
    fail_val = max_val - 20
    fig8 = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs1 = fig8.add_gridspec(nrows=8, ncols=8, left=0.1, right=0.9, wspace=0.00, hspace=0)
    f8_ax_mid = fig8.add_subplot(gs1[:, :])
    f8_ax_bot = fig8.add_subplot(gs1[-1, :])
    f8_ax_lft = fig8.add_subplot(gs1[:, 0])

    gjs_scatter_plot(f8_ax_mid, data_x, data_y, max_val, fail_val)
    # f8_ax_mid.set_xlabel(bot_name)
    # f8_ax_mid.set_ylabel(lft_name)
    # f8_ax_mid.set_title(title)

    f8_ax_lft.boxplot(list(data_y.values()), vert=True, showmeans=True)
    f8_ax_lft.set_ylim([0, max_val])
    f8_ax_lft.set_axis_off()

    # Bottom
    f8_ax_bot.boxplot(list(data_x.values()), vert=False, showmeans=True)
    f8_ax_bot.set_xlim([0, max_val])
    f8_ax_bot.set_axis_off()

    return f8_ax_mid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a figure (and write to file) for results from the interpretability project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('--output_image_file',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('--output_file2', type=str,
                        required=False, default=None)
    parser.add_argument('--learned', action='store_true')
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--learned_sp', action='store_true')
    args = parser.parse_args()

    if args.learned:
        data = process_learned_data(args)
        print(data.describe())
    elif args.naive:
        data = process_naive_data(args)
        print(data.describe())
    elif args.learned_sp:
        data = process_learned_sp_data(args)
        print(data.describe())
    else:
        data = process_data(args)
        print(data.describe())
        result_dict = data.set_index('seed').T.to_dict()

        Known_dict = {k: result_dict[k]['Known Cost'] for k in result_dict}
        Naive_dict = {k: result_dict[k]['Naive Cost'] for k in result_dict}
        Learned_dict = {k: result_dict[k]['Learned Cost'] for k in result_dict}
        make_scatter_with_box(Known_dict, Learned_dict)
        plt.savefig(args.output_image_file, dpi=600)
        plt.clf()
        make_scatter_with_box(Naive_dict, Learned_dict)
        plt.savefig(args.output_file2, dpi=600)
