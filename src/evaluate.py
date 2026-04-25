import numpy as np
import torch
import torch.nn as nn

def evaluate(model ,X_test, y_test):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    model.eval()

    criterion = nn.MSELoss()

    with torch.no_grad():
        predictions = model(X_test)
        mse_loss = criterion(predictions, y_test).item()

    predictions_np = predictions.cpu().numpy().flatten()
    y_test_np = y_test.cpu().numpy().flatten()

    mae = np.mean(np.abs(predictions_np - y_test_np))
    rmse = np.sqrt(mse_loss)

    print(f"Test MSE: {mse_loss:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
