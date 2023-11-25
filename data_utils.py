# Author: Alex Schofield
# Title: data_utils.py
# Description: Define functions for loading and processing data

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from Bio import SeqIO


def import_fasta(file_path):
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequence = str(record.seq)
            # print(f"Read sequence: {sequence}")
            yield sequence


def tokenize_sequences(sequences):
    amino_acids = sorted(set("".join(sequences)))
    aa_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    idx_aa = {idx: aa for aa, idx in aa_idx.items()}
    tokenized_sequences = [
        torch.tensor([aa_idx[aa] for aa in seq], dtype=torch.long) for seq in sequences
    ]
    # print(f"Tokenized Sequences: {tokenized_sequences}")
    # print(f"Amino Acid to Index Mapping: {aa_idx}")
    # print(f"Index to Amino Acid Mapping: {idx_aa}")
    return tokenized_sequences, aa_idx, idx_aa


class pp_dataset(Dataset):
    def __init__(self, tokenize_sequences):
        self.tokenize_sequences = tokenize_sequences

    def __len__(self):
        return len(self.tokenize_sequences)

    def __getitem__(self, index):
        sequence = self.tokenize_sequences[index]
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]
        print(
            f"__getitem__ index: {index}, input: {input_sequence.size()}, target: {target_sequence.size()}"
        )
        return input_sequence, target_sequence


def collate_batch(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    print(
        f"collate_batch: input shape {input_seqs_padded.shape}, target shape {target_seqs_padded.shape}"
    )
    return input_seqs_padded, target_seqs_padded
