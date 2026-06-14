#%% LIB
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from src.model.gru import GRURegressor
from src.model.train import train

#%% CONFIG
if torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available

#%% DATA
# ---------------------------------------------------
# Create sin series
# ---------------------------------------------------
t = torch.arange(0, 1500, dtype=torch.float32)
series = torch.sin(0.05 * t)
noise = 0.1 * torch.randn_like(series)
noise_series = series + noise

sns.lineplot(x=t, y=noise_series)
plt.show()

# ---------------------------------------------------
# Create window input & output
# ---------------------------------------------------
window_size = 20
X = []
y = []
for i in range(len(noise_series) - window_size):
    X.append(noise_series[i: i + window_size])
    y.append(noise_series[i + window_size])
X = torch.stack(X)
y = torch.stack(y)

print(X.shape)
print(y.shape)

# ---------------------------------------------------
# Add feature dimension
# ---------------------------------------------------
# This returns: X with torch.Size([980, 20, 1])
# Mean: 980 samples, 20 timesteps, 1 feature
# Corresponding to: (batch_size, sequence_length, input_size)

X = X.unsqueeze(dim=-1)
y = y.unsqueeze(dim=-1)

print(X.shape)  
print(y.shape)

#%% MODEL
# ---------------------------------------------------
# Training
# ---------------------------------------------------
model = GRURegressor(
    input_size=1,
    hidden_size=32,
    output_size=1
    )
model, training_loss = train(
    model, 
    X, 
    y, 
    100,
    device=device
    )

#%% EVALUATION
sns.lineplot(training_loss)
plt.show()

t = torch.arange(0, 20, dtype=torch.float32)
series_sample = torch.sin(0.05 * t)

for t in range(200):
    series_sample_stack = torch.stack([series_sample])
    series_sample_unsqueeze = series_sample_stack.unsqueeze(-1)
    with torch.no_grad():
        pred = model(series_sample_unsqueeze.to(device))
        series_sample = torch.cat([series_sample, pred.flatten().cpu()])

pred_np = series_sample.detach().cpu().numpy().flatten()

# sns.lineplot(x=t[1020:], y=noise_series[1020:])
# sns.lineplot(x=t[1020:], y=series[1020:])
sns.lineplot(x=torch.arange(0, len(pred_np), dtype=torch.float32), y=pred_np)
plt.show()


