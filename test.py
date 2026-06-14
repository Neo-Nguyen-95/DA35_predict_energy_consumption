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
t = torch.arange(0, 1000, dtype=torch.float32)
series = torch.sin(0.05 * t)
noise = 0.1 * torch.randn_like(series)
noise_series = series + noise

sns.set_theme(style='whitegrid', context='notebook')
sns.lineplot(x=t, y=noise_series)
plt.title('Sine wave with noise')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.show()

# ---------------------------------------------------
# Create window input & output
# ---------------------------------------------------
# This step return: X with torch.Size([980, 20])
# Mean: 980 sample, 20 timestep
# Corresponding to: (batch_size, sequence_length)

window_size = 20
X = []  # X will contain list of tensor
y = []
for i in range(len(noise_series) - window_size):
    X.append(noise_series[i: i + window_size])
    y.append(noise_series[i + window_size])
     
X = torch.stack(X)  # Stack all tensor 
y = torch.stack(y)

print(X.shape)
print(y.shape)

# ---------------------------------------------------
# Add feature dimension
# ---------------------------------------------------
# This step returns: X with torch.Size([980, 20, 1])
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
# Eval 1: Test again with sin wave
# ---------------------------------------------------
with torch.no_grad():
    pred = model(X.to(device))

pred_numpy = pred.detach().cpu().numpy().flatten()

plt.figure()
ax = sns.lineplot(x=t[20:], y=series[20:], label='True sin wave')
ax = sns.lineplot(x=t[20:], y=pred_numpy, label='Predict value')
ax.legend(loc="upper right")
plt.title('Compare true sin wave with predicted wave from GRU')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.show()

# ---------------------------------------------------
# Eval 2: Give 1 window size value => predict 300
# ---------------------------------------------------
# t = torch.arange(0, 20, dtype=torch.float32)
# series_sample = torch.sin(0.05 * t)

# for t in range(300):
#     series_sample_stack = torch.stack([series_sample])
#     series_sample_unsqueeze = series_sample_stack.unsqueeze(-1)
#     with torch.no_grad():
#         pred = model(series_sample_unsqueeze.to(device))
#         series_sample = torch.cat([series_sample, pred.flatten().cpu()])

# pred_np = series_sample.detach().cpu().numpy().flatten()

# sns.lineplot(x=t[1020:], y=noise_series[1020:])
# t_test = torch.arange(0, len(pred_np), dtype=torch.float32)
# sns.lineplot(x=t_test, y=torch.sin(0.05 * t_test))
# sns.lineplot(x=t_test, y=pred_np)
# plt.show()


