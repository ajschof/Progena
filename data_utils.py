# Author: Alex Schofield
# Title: data_utils.py
# Description: Define functions for loading and processing data

import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence

def importFASTA(file_path):
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequence = str(record.seq)
            #print(f"Read sequence: {sequence}")
            yield sequence

def tokenizedSequences(sequences):
    aminoAcids = sorted(set("".join(sequences)))
    aa_idx = {aa: idx for idx, aa in enumerate(aminoAcids)}
    idx_aa = {idx: aa for aa, idx in aa_idx.items()}
    tokenizedSequences = [torch.tensor([aa_idx[aa] for aa in seq], dtype=torch.long) for seq in sequences]
    #print(f"Tokenized Sequences: {tokenizedSequences}")
    #print(f"Amino Acid to Index Mapping: {aa_idx}")
    #print(f"Index to Amino Acid Mapping: {idx_aa}")
    return tokenizedSequences, aa_idx, idx_aa

class PP_Dataset(Dataset):
    def __init__ (self, tokenizedSequences):
        self.tokenizedSequences = tokenizedSequences

    def __len__(self):
        return len(self.tokenizedSequences)

    def __getitem__(self, index):
        sequence = self.tokenizedSequences[index]
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]
        print(f"__getitem__ index: {index}, input: {input_sequence.size()}, target: {target_sequence.size()}")
        return input_sequence, target_sequence

def collate_batch(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    print(f"collate_batch: input shape {input_seqs_padded.shape}, target shape {target_seqs_padded.shape}")
    return input_seqs_padded, target_seqs_padded
