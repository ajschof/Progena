import torch
import torch.nn as nn
import torch.optim as optim


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
    for batch, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch}: inputs {inputs.shape}, targets {targets.shape}")

        inputs_one_hot = torch.nn.functional.one_hot(
            inputs, num_classes=model.input_size
        ).float()
        inputs_one_hot = inputs_one_hot.to(device)

        targets = targets.view(-1).to(device)

        optimizer.zero_grad()

        output, _ = model(inputs_one_hot)

        # targets = targets[:, -1]

        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch}, Loss: {loss.item()}")


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
        train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")
