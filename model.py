import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Candidate gate
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x, h_prev):
        z = torch.sigmoid(
            self.W_z(x) + self.U_z(h_prev)
            )
        r = torch.sigmoid(
            self.W_r(x) + self.U_r(h_prev)
            )
        h_tilde = torch.tanh(
            self.W_h(x) + self.U_h(r * h_prev)
            )
        h = (1 - z) * h_prev + z * h_tilde
        return h
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        
    def forward(self, x, h0=None):
        batch_size, sequence_length, _ = x.shape
        
        if h0 is None:
            h = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device
                )
        else:
            h = h0
        
        outputs = []
        for t in range(sequence_length):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)
        return outputs, h
        




my_gru = GRU(
    input_size=10,
    hidden_size=20
)
x = torch.randn(8, 15, 10)

out, h = my_gru(x)

print(out.shape)
print(h.shape)
