import torch
import torch.nn as nn
import torch.optim as optim

MODEL_PATH = "saved_models/model_v2.pth"

def train(device, model, X_train, y_train, batch_size, lr, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {device}")
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            # Convert numpy arrays to torch tensors and move to device
            inputs = torch.from_numpy(X_train[i:i+batch_size]).float().to(device)
            targets = torch.from_numpy(y_train[i:i+batch_size]).float().to(device).unsqueeze(1) # unsqueeze for (batch_size, 1) target shape

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Print loss once per epoch for cleaner output
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("Training complete")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
