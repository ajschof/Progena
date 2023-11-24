import torch
import torch.nn as nn
import torch.optim as optim

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size, x.device)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output[:, -1, :])
        return output, hidden

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        return hidden

# Training Function
def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    for inputs, targets in train_loader:
        seq_len = inputs.shape[1]  # assuming inputs are of shape (batch_size, seq_len)
        inputs = inputs.view(seq_len, -1, 21).to(device, dtype=torch.float32)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



# Validation Function
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

def run_training_process(input_size, hidden_size, output_size, num_layers, batch_size, num_epochs, train_loader, val_loader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")
