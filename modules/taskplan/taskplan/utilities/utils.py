import os
import learning


def write_datum_to_file(args, datum, counter):
    data_filename = os.path.join('pickles', f'dat_{args.current_seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(args.save_dir, data_filename), datum)
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.csv'
    with open(os.path.join(args.save_dir, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')
