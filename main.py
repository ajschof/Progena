import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from Bio import SeqIO
from colorama import Fore, Style


def import_fasta(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequence = str(record.seq)
            yield sequence


def tokenize_sequences(sequences):
    amino_acids = sorted(set("".join(sequences)))
    aa_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    idx_aa = {idx: aa for aa, idx in aa_idx.items()}
    tokenized_sequences = [
        torch.tensor([aa_idx[aa] for aa in seq], dtype=torch.long) for seq in sequences
    ]
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
        return input_sequence, target_sequence


def collate_batch(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    return input_seqs_padded, target_seqs_padded


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size, x.device)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        output, hidden = self.lstm(x, hidden)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        )
        return hidden


# def save_checkpoint(state, filepath):
#     torch.save(state, filepath)


# def load_checkpoint(filepath, model, optimizer, device):
#     checkpoint = torch.load(filepath, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     return checkpoint["epoch"], checkpoint["train_loss"], checkpoint["val_loss"]


def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    for batch, (inputs, targets) in enumerate(train_loader):
        # One-hot encode inputs and move to device
        inputs_one_hot = torch.nn.functional.one_hot(
            inputs, num_classes=model.input_size
        ).float()
        inputs_one_hot, targets = inputs_one_hot.to(device), targets.to(device)

        # Forward pass and loss computation
        optimizer.zero_grad()
        output, _ = model(inputs_one_hot)
        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Calculate and display percentage completed
        percentage_completed = 100 * (batch + 1) / num_batches
        sys.stdout.write(
            f"\r{Fore.GREEN}{Style.BRIGHT}Epoch {epoch+1}/{num_epochs}, Batch {batch+1}/{num_batches}, {percentage_completed:.2f}% completed{Style.RESET_ALL}"
        )
        sys.stdout.flush()

    print()  # To move to the next line after the last batch
    return total_loss / num_batches



def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output, _ = model(inputs)
            loss = criterion(output, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def start(
    input_size,
    hidden_size,
    output_size,
    num_layers,
    num_epochs,
    train_loader,
    val_loader=None,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available")
    else:
        device = torch.device("cpu")
        print("CUDA and MPS are not available, falling back to CPU")
    model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        train_loss = train(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )

        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")

        # checkpoint = {
        #     "epoch": epoch + 1,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "train_loss": train_loss,
        #     "val_loss": val_loss,
        # }
        # save_checkpoint(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")


def main():
    parser = argparse.ArgumentParser(description="Process FASTA sequences")
    parser.add_argument("fasta_file", type=str, help="Path to FASTA file")
    # parser.add_argument(
    #     "--checkpoint", type=str, help="Path to a saved checkpoint", default=None
    # )
    args = parser.parse_args()

    try:
        sequences = list(import_fasta(args.fasta_file))
        print("✅ Loaded FASTA file")
    except Exception as e:
        print("❌ Failed to load FASTA file")
        print(e)
        return

    try:
        tokenized_seqs, aa_idx, _ = tokenize_sequences(sequences)
        print("✅ Tokenized sequences")
    except Exception as e:
        print("❌ Failed to tokenize sequences")
        print(e)
        return

    input_size = len(aa_idx)
    output_size = len(aa_idx)
    hidden_size = 128
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
