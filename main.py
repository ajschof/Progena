# Author: Alex Schofield
# Title: main.py (WIP)

import argparse
from data_utils import importFASTA, tokenizedSequences, PP_Dataset, collate_batch
from training_utils import run_training_process
from torch.utils.data import DataLoader


# Redundant code, but I'm keeping it here for now...

def main():
    parser = argparse.ArgumentParser(description="Process FASTA sequences")
    parser.add_argument("fasta_file", type=str, help="Path to FASTA file")
    args = parser.parse_args()
    
    try:
        sequences = list(importFASTA(args.fasta_file))
        print("✅ Loaded FASTA file")
    except Exception as e:
        print("❌ Failed to load FASTA file")
        print(e)
        return

    try:
        tokenized_seqs, aa_idx, _ = tokenizedSequences(sequences)
        print("✅ Tokenized sequences")
    except Exception as e:
        print("❌ Failed to tokenize sequences")
        print(e)
        return
    
    # Define model parameters
    input_size = len(aa_idx)  # Number of unique amino acids
    hidden_size = 128
    output_size = input_size
    num_layers = 2
    batch_size = 32
    num_epochs = 10

    # Initialize dataset and dataloaders
    dataset = PP_Dataset(tokenized_seqs)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    # Define val_loader if you have a validation set

    # Run the training process
    run_training_process(input_size, hidden_size, output_size, num_layers, batch_size, num_epochs, train_loader)
    # Include val_loader as an argument if it's defined

if __name__ == "__main__":
    main()