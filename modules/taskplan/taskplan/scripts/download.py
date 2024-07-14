import prior
import json
import argparse
from sentence_transformers import SentenceTransformer


def save_model(args):
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    model.save(args.save_dir)


def load_procthor_data(args):
    dataset = prior.load_dataset("procthor-10k")
    train_data = dataset['train']

    # File path where you want to save the JSON Lines file
    file_path = f'{args.save_dir}/data.jsonl'

    # Writing data to a JSONL file
    with open(file_path, 'w') as file:
        for entry in train_data:
            # Assuming 'entry' is a dictionary-like object; adjust if necessary
            json_string = json.dumps(entry)
            file.write(json_string + '\n')


if __name__ == "__main__":
    # Load the procthor-10k data and save it to data/procthor-data/ directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    if args.save_dir == '/data/procthor-data':
        load_procthor_data(args)
    else:
        save_model(args)
