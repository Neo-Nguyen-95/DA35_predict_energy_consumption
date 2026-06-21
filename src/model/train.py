#%% LIB
import copy

from torch import nn
import torch


#%% MAIN
def train(
        model, 
        loader,
        max_epoch=100,
        lr=1e-3,
        device='mps',
        print_epoch_loss=True,
        early_stopping=True,
        min_delta=1e-3,
        patience=2,
        ):
    model = model.to(device)
    
    criterion = nn.MSELoss()  # Define loss: Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Optimizer
    
    training_loss = []
    best_loss = None
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    for epoch in range(max_epoch):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if X_batch.ndim == 2:
                X_batch = X_batch.unsqueeze(dim=-1)
            if y_batch.ndim == 1:
                y_batch = y_batch.unsqueeze(dim=-1)
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(loader.dataset)
        training_loss.append(epoch_loss)

        is_best_loss = (
            best_loss is None
            or epoch_loss < best_loss * (1 - min_delta)
        )
        if is_best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping:
            if epochs_without_improvement >= patience:
                print(f"Stop at epoch: {epoch}; best epoch: {best_epoch}; best loss: {best_loss:.6f}")
                break

        if print_epoch_loss and epoch % 10 == 0:
            print(f"Epoch {epoch}: {epoch_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model, training_loss, best_epoch
