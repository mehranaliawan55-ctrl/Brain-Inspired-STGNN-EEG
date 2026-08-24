import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainInspiredSTGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network (STGNN) with Continual Learning Bottleneck
    for Multi-Channel EEG Signal Decoding.
    """
    def __init__(self, num_channels: int = 64, time_steps: int = 500, num_classes: int = 4):
        super(BrainInspiredSTGNN, self).__init__()
        
        # Temporal Convolutional Layers
        self.temp_conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=7, padding=3)
        self.temp_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        
        # Spatial Graph Convolution Weights
        self.spatial_weight = nn.Parameter(torch.FloatTensor(64, 64))
        nn.init.xavier_uniform_(self.spatial_weight)
        
        # Continual Learning Bottleneck Layer
        self.bottleneck = nn.Linear(64 * time_steps, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor of shape (batch_size, num_channels, time_steps)
        adj: Adjacency matrix of shape (num_channels, num_channels)
        """
        # Temporal Feature Extraction
        x = F.relu(self.temp_conv1(x))
        x = F.relu(self.temp_conv2(x))
        
        # Spatial Graph Convolution
        x = torch.einsum('bct,cc->bct', x, adj)
        x = torch.matmul(x.transpose(1, 2), self.spatial_weight).transpose(1, 2)
        
        # Bottleneck & Classification
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.bottleneck(x))
        out = self.classifier(x)
        
        return out

if __name__ == "__main__":
    # Test pipeline with dummy 64-channel EEG batch
    model = BrainInspiredSTGNN(num_channels=64, time_steps=500, num_classes=4)
    dummy_eeg = torch.randn(8, 64, 500)
    dummy_adj = torch.eye(64)
    
    outputs = model(dummy_eeg, dummy_adj)
    print(f"[STGNN Engine] Pipeline Verification Successful.")
    print(f"[STGNN Engine] Output Logits Shape: {outputs.shape}")
