import argparse
import matplotlib.pyplot as plt

import taskplan


def generate_map_info(args):
    # Load data for a given seed
    thor_data = taskplan.utilities.ai2thor_helper. \
        ThorInterface(args=args)

    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)
    plt.title('Top-down view of the map', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=400)

    with open(f'{args.save_dir}/{args.info_filename}', "w+") as file:
        for cnt in thor_data.containers:
            cnt_id = cnt['id']
            connected_objects = cnt.get('children')
            children = [obj['id'] for obj in connected_objects] if connected_objects else []
            file.write(f"{cnt_id}: {children}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation using ProcTHOR for Task Planning"
    )
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--image_filename', type=str, required=False)
    parser.add_argument('--info_filename', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    parser.add_argument('--resolution', type=float, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    generate_map_info(args)
