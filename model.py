import torch
import torch.nn as nn

class PaymentRiskDNN(nn.Module):
    def __init__(self, input_dim):
        super(PaymentRiskDNN, self).__init__()
        # Deep Neural Network / Multilayer Perceptron (MLP)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Menghasilkan probabilitas [0, 1]
        )
        
    def forward(self, x):
        return self.network(x)
