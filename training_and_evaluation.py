#%% LIB
import torch
from model import train


#%% CONFIG
if torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
    
#%% MAIN