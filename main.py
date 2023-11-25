# Author: Alex Schofield
# Title: main.py (WIP)

import argparse
from torch.utils.data import DataLoader
from data_utils import import_fasta, tokenized_sequences, pp_dataset, collate_batch
from training_utils import start


def main():
    # parser = argparse.ArgumentParser(description="Process FASTA sequences")
    # parser.add_argument("fasta_file", type=str, help="Path to FASTA file")
    # args = parser.parse_args()

    try:
        # sequences = list(import_fasta(args.fasta_file))
        sequences = list(import_fasta("uniprot-28proteome-smalltest.fasta"))
        print("✅ Loaded FASTA file")
    except Exception as e:
        print("❌ Failed to load FASTA file")
        print(e)
        return

    try:
        tokenized_seqs, aa_idx, _ = tokenized_sequences(sequences)
        print("✅ Tokenized sequences")
    except Exception as e:
        print("❌ Failed to tokenize sequences")
        print(e)
        return

    input_size = len(aa_idx)
    hidden_size = 128
    output_size = input_size
    num_layers = 2
    batch_size = 32
    num_epochs = 10

    dataset = pp_dataset(tokenized_seqs)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    start(input_size, hidden_size, output_size, num_layers, num_epochs, train_loader)


if __name__ == "__main__":
    main()
