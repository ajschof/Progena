import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
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


def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
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

        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        correct_predictions += (predicted == targets.view(-1)).sum().item()
        total_predictions += targets.numel()

        # Calculate and display percentage completed along with loss and accuracy
        percentage_completed = 100 * (batch + 1) / num_batches
        average_loss = total_loss / (batch + 1)
        accuracy = correct_predictions / total_predictions

        sys.stdout.write(
            f"\r{Fore.YELLOW}{Style.BRIGHT}Training... {percentage_completed:.2f}% | Epoch {epoch+1}/{num_epochs}, Batch {batch+1}/{num_batches}, Train Loss: {average_loss:.4f}, Train Acc: {accuracy * 100:.2f}%{Style.RESET_ALL}"
        )

        sys.stdout.flush()

    print(f"🚂 Finished training for Epoch {epoch+1}/{num_epochs}")
    return total_loss / num_batches, accuracy


def validate(model, val_loader, criterion, device, epoch, num_epochs):
    print("🔍 Starting validation phase...")  # Debug print
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(val_loader):
            inputs_one_hot = torch.nn.functional.one_hot(
                inputs, num_classes=model.input_size
            ).float()
            inputs_one_hot, targets = inputs_one_hot.to(device), targets.to(device)

            output, _ = model(inputs_one_hot)
            loss = criterion(output, targets.view(-1))
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            correct_predictions += (predicted == targets.view(-1)).sum().item()
            total_predictions += targets.numel()

            # Calculate and display percentage completed along with loss and accuracy
            percentage_completed = 100 * (batch + 1) / num_batches
            average_loss = total_loss / (batch + 1)
            accuracy = correct_predictions / total_predictions

            sys.stdout.write(
                f"\r{Fore.CYAN}{Style.BRIGHT}Validating... {percentage_completed:.2f}% | Epoch {epoch+1}/{num_epochs}, Batch {batch+1}/{num_batches}, Val Loss: {average_loss:.4f}, Val Acc: {accuracy * 100:.2f}%{Style.RESET_ALL}"
            )
            sys.stdout.flush()

    print(f"\n🚂 Finished training for Epoch {epoch+1}/{num_epochs}")
    return total_loss / num_batches, accuracy


def start(
    input_size,
    hidden_size,
    output_size,
    num_layers,
    num_epochs,
    train_loader,
    val_loader,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA is available")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS is available")
    else:
        device = torch.device("cpu")
        print("❌ CUDA and MPS are not available, falling back to CPU")
    model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_accuracy = train(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )

        # Validation phase
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device, epoch, num_epochs
        )

        # Print training and validation results
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Process FASTA sequences")
    parser.add_argument("fasta_file", type=str, help="Path to FASTA file")
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

    full_dataset = pp_dataset(tokenized_seqs)
    train_size = int(0.8 * len(full_dataset))  # e.g., 80% for training
    val_size = len(full_dataset) - train_size  # remainder for validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    start(
        input_size,
        hidden_size,
        output_size,
        num_layers,
        num_epochs,
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    main()
