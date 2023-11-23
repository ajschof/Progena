# Author: Alex Schofield
# Title: main.py (WIP)

import argparse
from data_utils import importFASTA, tokenizedSequences

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

    print("First few tokenized sequences:")
    for seq in tokenized_seqs[:5]:  # Print the first few tokenized sequences
        print(seq)

if __name__ == "__main__":
    main()
