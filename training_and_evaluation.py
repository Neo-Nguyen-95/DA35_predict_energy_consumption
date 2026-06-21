#%% LIB
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model.train import train

from src.service.input_data import fetch_and_transform_data
from src.service.transform_data import get_daily_data, get_weekly_data
from src.model.gru import GRURegressor

import matplotlib.pyplot as plt
import seaborn as sns


def make_windows(series, window_size):
    if len(series) <= window_size:
        raise ValueError("Series length must be greater than window_size.")
    windows = series.unfold(
        0,  # unfold along dimension 0, the series is 1D
        window_size + 1,  # +1 is for the y value
        1  # moving 1 step
        )
    return (
        windows[:, :-1].contiguous(),  # X
        windows[:, -1].contiguous()  # y
        )


#%% CONFIG
device = "mps" if torch.backends.mps.is_available() else "cpu"
    
WINDOW_SIZE = 12
BATCH_SIZE = 32
#%% DATA
#%%% Full data
df = fetch_and_transform_data()
# df_daily = get_daily_data(df)
df_weekly = get_weekly_data(df)

series_data = torch.tensor(
    # df_daily['Daily_power_consumption'],
    df_weekly['Weekly_energy_consumption'], 
    dtype=torch.float32
    )
split_point = int(0.8 * len(series_data))
series_data_train = series_data[:split_point]
series_data_test = series_data[split_point:]

train_mean = series_data_train.mean()
train_std = series_data_train.std()
series_data_train_scaled = (series_data_train - train_mean) / train_std
series_data_test_scaled = (series_data_test - train_mean) / train_std


#%%% Training dataset
X_train, y_train = make_windows(series_data_train_scaled, WINDOW_SIZE)

print(X_train.shape)
print(y_train.shape)

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
    )

#%%% Testing dataset
X_test, y_test = make_windows(series_data_test_scaled, WINDOW_SIZE)

print(X_test.shape)
print(y_test.shape)


#%% MODEL

#%%% Training
model = GRURegressor(
    input_size=1,
    hidden_size=32,
    output_size=1
    )
model, training_loss, best_epoch = train(
    model, 
    loader, 
    max_epoch=300,
    lr=5e-3,
    device=device,
    print_epoch_loss=True,
    patience=10
    )
print(f"Best epoch: {best_epoch}, best loss: {training_loss[best_epoch]:.6f}")

#%%% Evaluation
# ---------------------------------------------------
# Eval 0: Training losss
# ---------------------------------------------------
plt.figure()
sns.lineplot(training_loss)
plt.title('Loss plot')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# ---------------------------------------------------
# Eval 1: Train set
# ---------------------------------------------------
model.eval()
with torch.no_grad():
    pred = model(X_train.to(device).unsqueeze(-1))

pred = pred.detach().cpu().flatten() * train_std + train_mean
real = y_train.detach().cpu().flatten() * train_std + train_mean

pred_numpy = pred.numpy()
real_numpy = real.numpy()

mae = torch.mean(torch.abs(pred - real))
rmse = torch.sqrt(torch.mean((pred - real) ** 2))
mape = torch.mean(torch.abs((real - pred) / real)) * 100  # Mean Absolute Percentage Error

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

plt.figure()
ax = sns.lineplot(real_numpy, label='True')
ax = sns.lineplot(pred_numpy, label='Predict')
ax.legend(loc="upper right")
plt.title('Compare true with predicted value in Training Dataset')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.show()

# ---------------------------------------------------
# Eval 2: Test set
# ---------------------------------------------------
model.eval()
with torch.no_grad():
    pred = model(X_test.to(device).unsqueeze(-1))

pred = pred.detach().cpu().flatten() * train_std + train_mean
real = y_test.detach().cpu().flatten() * train_std + train_mean

pred_numpy = pred.numpy()
real_numpy = real.numpy()

mae = torch.mean(torch.abs(pred - real))
rmse = torch.sqrt(torch.mean((pred - real) ** 2))
mape = torch.mean(torch.abs((real - pred) / real)) * 100  # Mean Absolute Percentage Error

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

plt.figure()
ax = sns.lineplot(real_numpy, label='True')
ax = sns.lineplot(pred_numpy, label='Predict')
ax.legend(loc="upper right")
plt.title('Compare true with predicted value in Testing Dataset')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.show()

# ---------------------------------------------------
# Eval 3: Both set
# ---------------------------------------------------
X_all = torch.cat([X_train[-len(X_test):], X_test], dim=0)
y_all = torch.cat([y_train[-len(X_test):], y_test], dim=0)
split_index = len(y_train[-len(X_test):])

model.eval()
with torch.no_grad():
    pred = model(X_all.to(device).unsqueeze(-1))

pred = pred.detach().cpu().flatten() * train_std + train_mean
real = y_all.detach().cpu().flatten() * train_std + train_mean

pred_numpy = pred.numpy()
real_numpy = real.numpy()

mae = torch.mean(torch.abs(pred - real))
rmse = torch.sqrt(torch.mean((pred - real) ** 2))
mape = torch.mean(torch.abs((real - pred) / real)) * 100  # Mean Absolute Percentage Error

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

error_numpy = (real_numpy - pred_numpy) / real_numpy * 100

fig, (ax, ax_error) = plt.subplots(
    2, 1,
    figsize=(10, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
    )
ax.plot(
        real_numpy, 
        label="True", 
        color="tab:blue", 
        linewidth=1.5, 
        alpha=0.9
        )
ax.plot(
        pred_numpy, 
        label="Predict", 
        color="tab:orange", 
        linewidth=1.0, 
        alpha=0.9
        )
ax.axvline(
    split_index, 
    color="red", 
    linestyle="--", 
    linewidth=1.2, 
    label="Train/Test split"
    )
ax.set_title("True vs Predicted Value in Train and Test Dataset")
ax.set_ylabel("Value")
ax.legend(loc="upper right", ncol=2)
ax.grid(alpha=0.25)

ax_error.axhline(
    0, 
    color="black", 
    linewidth=1
    )
ax_error.plot(error_numpy, color="tab:red", linewidth=1.0, alpha=0.75)
ax_error.axvline(split_index, color="red", linestyle="--", linewidth=1.2)
ax_error.set_title("Prediction Error")
ax_error.set_xlabel("Timestep")
ax_error.set_ylabel("Error (%)")
ax_error.set_ylim(-100, 100)
ax_error.grid(alpha=0.25)

plt.tight_layout()
plt.show()
