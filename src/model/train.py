#%% LIB
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader

#%% MAIN
def train(
        model, 
        X, 
        y, 
        max_epoch,
        batch_size=64,
        lr=1e-3,
        device='mps',
        print_epoch_loss=True
        ):
    model = model.to(device)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()  # Define loss: Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Optimizer
    
    training_loss = []
    for epoch in range(max_epoch):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(dataset)
        training_loss.append(epoch_loss)
        if len(training_loss)>2:
            epoch_loss_reduction = abs(
                training_loss[-1] - training_loss[-2]
                ) / training_loss[-1]
            if epoch_loss_reduction <= 1e-3:
                print(f"Stop at epoch: {epoch}")
                break
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {loss.item():.4f}")
            
    return model, training_loss