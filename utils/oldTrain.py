import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.metrics import accuracy_score

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model

def evaluate(model, dataloader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total
